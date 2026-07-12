from collections import deque

import numpy as np
import torch
from tensordict import TensorDict
from torchrl.data import Bounded, Composite, Unbounded
from torchrl.envs import EnvBase

import lidar
from drone import Drone
from spray_tracker import SprayTracker
from open3d_sim_env import label_path_regions, fill_regions_top_materials

GOAL_RADIUS = 5.0


class DroneEnv(EnvBase):
    def __init__(self, values, start, end, n_rays=100, sight_range=10, spray_n_rays=50, max_steps=400, prob4=0.8):
        super().__init__()
        self.values = values
        self.og_values = values.copy()
        self.current_step = 0
        self.max_steps = max_steps
        self.start = np.array(start, dtype=np.float32)
        self.end = np.array(end, dtype=np.float32)
        self.n_rays = n_rays
        self.sight_range = sight_range
        # NEW: fraction of regions painted "needs water" (material 4) vs
        # "doesn't need water" (material 5). Passed in from outside (e.g. a
        # training-progress curriculum in rl_model.py) instead of always
        # using fill_regions_top_materials' own hardcoded default.
        self.prob4 = prob4
        self.MAX_MOVE_AMOUNT = 3.0  # upper bound of the learned move-scale action; matches the old hardcoded amount=3
        self.drone = Drone(0, self.sight_range, self.values, self.start, self.end)
        self.drone.set_spray_params(n_rays=spray_n_rays)
        self.path_history = []
        self.record = False

        self._last_scan = np.ones(self.n_rays)
        self._last_hit = False

        # --- NEW: "stuck" detection -- if the drone hasn't displaced more
        # than STUCK_DISPLACEMENT_THRESHOLD blocks over the last STUCK_WINDOW
        # moves (e.g. circling/oscillating in place instead of progressing),
        # treat it like a crash: same penalty, same episode-ending behavior.
        # History holds STUCK_WINDOW+1 positions so it spans exactly
        # STUCK_WINDOW moves (the +1 is the position *before* the first of
        # those moves).
        self._pos_history = deque(maxlen=self.STUCK_WINDOW + 1)
        self._pos_history.append(self.drone.pos.copy())
        self._last_stuck = False

        # --- NEW: region labels + SprayTracker, built once per terrain ---
        # label_path_regions/fill_regions_top_materials come from open3d_sim_env.py
        # and are what turn the raw voxel grid into "which (x,y) cells need
        # water vs shouldn't be sprayed" -- SprayTracker consumes exactly this.
        self.labels, _ = label_path_regions(self.values)
        self.values, self.region_materials = fill_regions_top_materials(
            self.values, self.labels, prob4=self.prob4
        )
        self.spray_tracker = SprayTracker(
            self.values, self.labels, self.region_materials, n_rays=spray_n_rays
        )
        # reward-tracking state, all reset fresh each episode in _reset()
        self._prev_completeness = {}          # {region_id: fraction} from last step, for progress delta
        self._prev_completed_regions = set()  # region_ids that already earned their completion bonus
        self._prev_all_done = False           # whether all_needs_water_done() was already True
        self._prev_useful_hits = 0.0          # cumulative on-target spray hits, for per-step spray reward
        self._prev_total_completeness = 0.0   # area-weighted (block-count) completeness, for progress delta

        # flat obs = lidar scan + goal direction(3) + goal distance(1)
        flat_size = n_rays + 4
        map_h, map_w = self.spray_tracker.shape  # (x, y) footprint of the terrain

        # --- CHANGED: observation_spec is now a Composite with two keys,
        # not a single flat "observation" key. This is what the CNN's
        # in_keys=["coverage_map", "flat"] on actor_net/value_net expect.
        self.observation_spec = Composite(
            coverage_map=Unbounded(shape=(2, map_h, map_w)),   # channel 0: material, channel 1: completion
            flat=Unbounded(shape=(flat_size,)),
            # NEW: explicit crash flag, separate from "done" (which also fires on
            # successful completion). Not consumed by the CNN (in_keys=["coverage_map",
            # "flat"] only) -- this exists purely so the training loop can log a real
            # crash rate instead of inferring it from reward thresholds or step counts.
            crashed=Unbounded(shape=(1,)),
        )

        # --- CHANGED: action is now movement (3) + spray toggle (1) + move
        # scale (1) = 5. Last-but-one component > 0 means "spray this step".
        # The final component controls HOW FAR this step moves: it's mapped
        # from [-1, 1] to [0, MAX_MOVE_AMOUNT] and passed straight to
        # Drone.move()'s `amount`, so the network can choose small cautious
        # steps as well as full-speed ones instead of always moving a fixed
        # distance regardless of what it outputs.
        self.action_spec = Bounded(low=-1, high=1, shape=(5,))

        self.reward_spec = Unbounded(shape=(1,))

    def _step(self, tensordict):
        self.current_step += 1
        action = tensordict["action"].cpu().numpy()
        move_action, spray_action, scale_action = action[:3], action[3], action[4]

        # scale_action is in [-1, 1] (from the Tanh-squashed policy output);
        # map it to [0, MAX_MOVE_AMOUNT] so the network can choose anywhere
        # from "barely move" to a full-speed step, instead of always moving
        # a fixed hardcoded distance regardless of what it outputs.
        move_amount = (scale_action + 1.0) / 2.0 * self.MAX_MOVE_AMOUNT

        prev_pos = self.drone.pos.copy()
        hit = self.drone.move(move_action, amount=move_amount)
        self._last_hit = hit

        # --- NEW: update stuck-detection history and flag. Record position
        # regardless of whether this move was a collision -- a drone that
        # keeps bouncing off the same wall in place should also count as stuck.
        self._pos_history.append(self.drone.pos.copy())
        self._last_stuck = self._check_stuck()

        # --- NEW: only spray (and register hits) if the agent chose to and
        # didn't just crash. This is what lets the agent learn to withhold
        # spray over no-water regions instead of spraying unconditionally.
        if not hit and spray_action > 0:
            hit_xyz = self.drone.spray()
            self.spray_tracker.register_hits(hit_xyz)

        if self.record and not hit:
            self.path_history.append(self.drone.pos.copy())

        obs = self._get_obs()
        reward = self._get_reward(prev_pos, current_step=self.current_step)
        terminated = self._is_terminated()
        truncated = self._is_truncated()
        done = terminated or truncated

        return TensorDict(
            {
                "coverage_map": torch.tensor(obs["coverage_map"], device=self.device),
                "flat": torch.tensor(obs["flat"], device=self.device),
                "reward": torch.tensor([reward], dtype=torch.float32, device=self.device),
                # "done" = episode over for ANY reason (terminated OR truncated).
                "done": torch.tensor([done], device=self.device),
                # "terminated" = a real end state: crashed, or finished all
                # needs-water regions. Distinct from truncation so a training
                # loop (e.g. GAE bootstrapping) can tell the difference.
                "terminated": torch.tensor([terminated], device=self.device),
                # "truncated" = ran out of time (hit max_steps) without a
                # real terminal outcome.
                "truncated": torch.tensor([truncated], device=self.device),
                # NEW: real crash flag (see observation_spec comment above), not
                # inferred from reward value -- this is self._last_hit OR
                # self._last_stuck, since getting stuck ends the episode and
                # is penalized identically to a real collision. If you later
                # want to log "collided" vs "stalled out" separately, split
                # this into two tensors instead.
                "crashed": torch.tensor([float(self._last_hit or self._last_stuck)], dtype=torch.float32, device=self.device),
            },
            batch_size=[],
        )

    def _reset(self, tensordict=None, **kwargs):
        self.values = self.og_values.copy()
        self.current_step = 0
        self.drone = Drone(0, self.sight_range, self.values, self.start, self.end)

        # --- NEW: rebuild region labels/materials for the fresh terrain and
        # reset the tracker's hit counts for the new episode.
        self.labels, _ = label_path_regions(self.values)
        self.values, self.region_materials = fill_regions_top_materials(
            self.values, self.labels, prob4=self.prob4
        )
        self.spray_tracker = SprayTracker(
            self.values, self.labels, self.region_materials, n_rays=self.drone.spray_n_rays
        )
        self._prev_completeness = {}
        self._prev_completed_regions = set()
        self._prev_all_done = False
        self._prev_useful_hits = 0.0
        self._prev_total_completeness = 0.0

        self._last_hit = False

        # --- NEW: reset stuck-detection history for the fresh episode,
        # seeded with the drone's starting position.
        self._pos_history = deque(maxlen=self.STUCK_WINDOW + 1)
        self._pos_history.append(self.drone.pos.copy())
        self._last_stuck = False

        obs = self._get_obs()
        return TensorDict(
            {
                "coverage_map": torch.tensor(obs["coverage_map"], device=self.device),
                "flat": torch.tensor(obs["flat"], device=self.device),
                "crashed": torch.tensor([0.0], dtype=torch.float32, device=self.device),
            },
            batch_size=[],
        )

    def _get_obs(self):
        self._last_scan = lidar.get_lidar_surroundings(
            self.values, self.drone.pos, self.sight_range
        )
        direction, distance = lidar.get_goal_vector(self.drone.pos, self.end, self.sight_range)
        flat = np.concatenate([self._last_scan, direction, [distance]]).astype(np.float32)

        coverage_map = self._build_coverage_map()

        return {"coverage_map": coverage_map, "flat": flat}

    def _build_coverage_map(self):
        """(2, H, W) tensor: channel 0 = region material (1.0 needs-water,
        0.5 no-water, 0.0 path/other), channel 1 = completion fraction
        (hit_counts / required_hits_per_cell, clipped to [0, 1])."""
        st = self.spray_tracker
        mat_channel = np.zeros(st.shape, dtype=np.float32)
        for rid in st.needs_water_regions:
            mat_channel[st.labels == rid] = 1.0
        for rid in st.no_water_regions:
            mat_channel[st.labels == rid] = 0.5

        comp_channel = np.clip(
            st.hit_counts / max(st.required_hits_per_cell, 1), 0.0, 1.0
        ).astype(np.float32)

        return np.stack([mat_channel, comp_channel], axis=0)

    # --- reward tuning constants -- all in one place so they're easy to find/adjust ---
    CRASH_PENALTY = -1.5                 # hitting terrain/trees/out-of-bounds
    STEP_COST = 0.0023                    # tiny per-step cost, discourages stalling
    OVERSPRAY_PENALTY_PER_HIT = 0.0004    # per spray-ray hit landed on a no-water region
    SPRAY_HIT_REWARD_PER_HIT = 0.02      # per spray-ray hit landed on a needs-water region (up to its cap)
    PROGRESS_REWARD_SCALE = 0.5           # multiplier on step-to-step completeness delta
    REGION_COMPLETE_BONUS = 0.75          # one-time bonus, paid the step a single region finishes
    ALL_REGIONS_COMPLETE_BONUS = 1.5     # one-time bonus, paid the step EVERY region finishes

    # --- "stuck" detection: no real displacement over a window of moves ---
    STUCK_WINDOW = 15                     # look back this many moves
    STUCK_DISPLACEMENT_THRESHOLD = 4.0    # if bounding-box diagonal over the window is under this many blocks, it's "stuck"

    def _get_reward(self, prev_pos, current_step=0, max_steps=400):
        # Stuck (no real displacement over STUCK_WINDOW moves) is treated
        # exactly like a terrain/bounds crash: same flat penalty, and it
        # skips the rest of the reward calc just like a real crash does.
        if self._hit_something() or self._last_stuck:
            return self.CRASH_PENALTY

        # --- 1. progress reward: did overall needs-water completeness improve? ---
        # completeness_now is still per-region (used below for the per-region
        # completion bonus), but the progress reward itself now uses an
        # area-weighted total across ALL needs-water cells combined, rather
        # than an unweighted mean of per-region fractions -- so a big region
        # counts proportionally more than a tiny one, matching total ground
        # actually watered instead of "average region completion".
        completeness_now = self.spray_tracker.needs_water_completeness()  # {region_id: fraction 0-1}
        total_completeness_now = self.spray_tracker.needs_water_total_completeness()
        progress_reward = self.PROGRESS_REWARD_SCALE * (total_completeness_now - self._prev_total_completeness)

        # --- 2. per-region completion bonus: did any region cross 100% THIS step? ---
        newly_completed_regions = {
            rid for rid, frac in completeness_now.items()
            if frac >= 0.95 and rid not in self._prev_completed_regions
        }
        region_complete_bonus = self.REGION_COMPLETE_BONUS * len(newly_completed_regions)

        # --- 3. all-regions-done bonus: one-time, paid only the step everything finishes ---
        all_done_now = self.spray_tracker.all_needs_water_done(threshold=0.90)
        all_done_bonus = self.ALL_REGIONS_COMPLETE_BONUS if (all_done_now and not self._prev_all_done) else 0.0

        # --- 4. overspray penalty: discourage spraying no-water regions ---
        overspray_hits_total = sum(self.spray_tracker.overspray_counts().values())
        overspray_penalty = self.OVERSPRAY_PENALTY_PER_HIT * overspray_hits_total

        # --- 5. spray-hit reward: direct, immediate credit for choosing to spray
        # a needs-water cell, rather than waiting for progress_reward (which is
        # a mean across all regions and barely moves for a single hit) or the
        # one-time completion bonus. Only counts hits up to each cell's required
        # amount, so it can't be farmed by oversaturating an already-done cell.
        useful_hits_now = self._useful_hit_total()
        new_useful_hits = max(useful_hits_now - self._prev_useful_hits, 0.0)
        spray_hit_reward = self.SPRAY_HIT_REWARD_PER_HIT * new_useful_hits

        # --- save state for next step's deltas ---
        self._prev_completeness = completeness_now
        self._prev_total_completeness = total_completeness_now
        self._prev_completed_regions |= newly_completed_regions
        self._prev_all_done = all_done_now
        self._prev_useful_hits = useful_hits_now

        reward = (
            progress_reward
            + region_complete_bonus
            + all_done_bonus
            + spray_hit_reward
            - overspray_penalty
            - self.STEP_COST
        )
        # return float(np.clip(reward, -1.0, 1.0))
        return float(reward)

    def _useful_hit_total(self):
        """Sum of spray-ray hits landed on needs-water cells, each capped at
        that cell's required_hits_per_cell so extra spraying past completion
        doesn't keep paying out."""
        st = self.spray_tracker
        if not st.needs_water_regions:
            return 0.0
        needs_water_mask = np.isin(st.labels, list(st.needs_water_regions))
        capped_hits = np.clip(st.hit_counts, 0, st.required_hits_per_cell)
        return float(capped_hits[needs_water_mask].sum())

    @staticmethod
    def _mean_or(d, default):
        """Mean of a dict's values, or `default` if the dict is empty."""
        return float(np.mean(list(d.values()))) if d else default

    def _hit_something(self):
        return self._last_hit

    def _check_stuck(self):
        """True if the drone's position over the last STUCK_WINDOW moves has
        stayed within a STUCK_DISPLACEMENT_THRESHOLD-block bounding box --
        i.e. it's been circling/oscillating in place instead of making real
        progress. Uses the diagonal of the axis-aligned bounding box of the
        recent position history as the "how far did it actually roam" measure.
        """
        if len(self._pos_history) < self._pos_history.maxlen:
            return False  # not enough history yet (e.g. start of episode)

        pts = np.array(self._pos_history)
        spread = np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))
        return spread < self.STUCK_DISPLACEMENT_THRESHOLD

    def _is_terminated(self):
        """Real terminal outcomes: crashed into something, got stuck (no
        real displacement over STUCK_WINDOW moves), or finished watering
        everything that needs it (90% overall, same threshold used in
        _get_reward). Nothing to do with step count."""
        return (
            self._hit_something()
            or self._last_stuck
            or self.spray_tracker.all_needs_water_done(threshold=0.90)
        )

    def _is_truncated(self):
        """Ran out of time: hit max_steps without crashing or finishing."""
        return self.current_step >= self.max_steps

    def _set_seed(self, seed):
        np.random.seed(seed)