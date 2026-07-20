from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torchrl.data import Bounded, Composite, Unbounded
from torchrl.envs import EnvBase

import lidar
from drone import Drone
from spray_tracker import SprayTracker
from open3d_sim_env import label_path_regions, fill_regions_top_materials

GOAL_RADIUS = 5.0


class DroneEnv(EnvBase):
    @staticmethod
    def _stride2_conv_out(n, kernel=3, stride=2, padding=1):
        """Output size of a single Conv2d(kernel=3, stride=2, padding=1)
        applied to an `n`-long spatial dimension -- matches the two
        stride-2 conv layers the old CNN used to downsample coverage_map
        with, so the new precomputed downscaled map ends up the same size
        the network used to see after the conv stack."""
        return (n + 2 * padding - kernel) // stride + 1

    def __init__(self, values, start, end, n_rays=100, sight_range=10, spray_n_rays=50, max_steps=400, prob4=0.8,
                 region_done_threshold=None):
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
        # NEW: "how much of a region must be watered before it counts as
        # done" threshold. Passed in from outside the same way prob4 is --
        # e.g. rl_model.py's training loop computes a STEPPED (staircase)
        # curriculum value via region_done_threshold_for_progress() and
        # threads it through here. Defaults to the fully-ramped END value
        # so standalone use (e.g. run_model_test.py, which never passes
        # this) evaluates against the hardest, final-stage threshold
        # rather than the easy early-training one.
        self.REGION_DONE_THRESHOLD = (
            self.REGION_DONE_THRESHOLD_END if region_done_threshold is None else float(region_done_threshold)
        )
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

        # --- NEW: one-region-at-a-time targeting. Instead of getting reward
        # for progress across every needs-water region simultaneously, the
        # drone is assigned a single "active" region -- the needs-water
        # region closest to where it spawned -- and only earns reward for
        # that one. Once it's done, the closest remaining needs-water region
        # becomes the new active one, and so on until none are left.
        # Distance is measured used the (x, y) footprint centroid cached by
        # SprayTracker, since regions are ground-plane footprints.
        self._region_norm = float(np.linalg.norm(self.spray_tracker.shape))  # map diagonal, for normalising region distances
        self._completed_regions_order = []  # region_ids already finished and moved past, in order
        self.active_region_id, _ = self.spray_tracker.closest_region(
            self.spray_tracker.needs_water_regions, self.drone.pos[:2]
        )
        # reward-tracking state, all reset fresh each episode in _reset()
        self._prev_active_completeness = 0.0  # active region's fraction last step, for progress delta
        self._prev_active_useful_hits = 0.0   # active region's on-target spray hits last step
        self._prev_active_distance = self._region_vector(self.active_region_id)[1]  # normalised distance to active region, for approach-reward delta

        # flat obs = lidar scan + goal direction(3)+distance(1)
        #
        # CHANGED: coverage_map is no longer fed through a CNN (see
        # _build_coverage_map below) -- it's now two precomputed, already-
        # downscaled channels: channel 0 is the active-region spray-need
        # map, channel 1 is the drone's position (a soft one-hot spike,
        # downscaled the same way as channel 0). Since the position channel
        # now puts the drone directly on the same grid as what needs
        # spraying, there's no separate hand-computed direction/distance to
        # the active region in `flat` anymore -- the map alone is meant to
        # carry that relationship, for the network to learn to read off
        # itself instead of being handed it as a shortcut.
        flat_size = n_rays + 4
        map_h, map_w = self.spray_tracker.shape  # (x, y) footprint of the terrain

        # --- NEW: matches the spatial size the old CNN's two stride-2,
        # kernel-3, padding-1 conv layers reduced the map down to
        # internally, so the downscaled map below carries the same amount
        # of spatial detail the network used to end up working with anyway
        # -- just computed directly instead of learned.
        self._map_out_hw = (
            self._stride2_conv_out(self._stride2_conv_out(map_h)),
            self._stride2_conv_out(self._stride2_conv_out(map_w)),
        )

        # --- CHANGED: observation_spec is now a Composite with two keys,
        # not a single flat "observation" key. This is what the actor/value
        # net's in_keys=["coverage_map", "flat"] expect. coverage_map is now
        # a 2-channel downscaled map (see _build_coverage_map) instead
        # of the old 3-channel full-resolution map a CNN used to chew on.
        #
        # NEW: density_map_full + drone_xy. coverage_map's channel 0 is
        # only the DOWNSCALED spray-need map (e.g. 8x8) that the network's
        # density_direction head reads to pick a t in [0, 1]. Resolving
        # that t into an actual target block should happen against the
        # REAL full-resolution grid (e.g. 30x30), not the coarse one --
        # but pick_block.pick_block() needs that real grid, plus the
        # drone's real (x, y), as plain per-step data. Since training
        # batches steps from many parallel envs AND replays old steps from
        # the buffer (whose envs have long since moved on), there's no
        # "current env" to query at pick time the way a single live env in
        # run_model_test.py has -- these two have to travel through the
        # tensordict/replay buffer like any other observation so each
        # historical step still carries its OWN correct state.
        self.observation_spec = Composite(
            coverage_map=Unbounded(shape=(2, *self._map_out_hw)),  # channel 0: active-region spray-need, channel 1: drone position -- both downscaled
            flat=Unbounded(shape=(flat_size,)),
            density_map_full=Unbounded(shape=(map_h, map_w)),  # real full-resolution active-region spray-need map
            drone_xy=Unbounded(shape=(2,)),  # drone's real (x, y), same coordinate frame as density_map_full
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
                "density_map_full": torch.tensor(obs["density_map_full"], device=self.device),
                "drone_xy": torch.tensor(obs["drone_xy"], device=self.device),
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

        # --- NEW: fresh one-region-at-a-time targeting for the new episode.
        self._completed_regions_order = []
        self.active_region_id, _ = self.spray_tracker.closest_region(
            self.spray_tracker.needs_water_regions, self.drone.pos[:2]
        )
        self._prev_active_completeness = 0.0
        self._prev_active_useful_hits = 0.0
        self._prev_active_distance = self._region_vector(self.active_region_id)[1]

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
                "density_map_full": torch.tensor(obs["density_map_full"], device=self.device),
                "drone_xy": torch.tensor(obs["drone_xy"], device=self.device),
                "crashed": torch.tensor([0.0], dtype=torch.float32, device=self.device),
            },
            batch_size=[],
        )

    def _get_obs(self):
        self._last_scan = lidar.get_lidar_surroundings(
            self.values, self.drone.pos, self.sight_range
        )
        direction, distance = lidar.get_goal_vector(self.drone.pos, self.end, self.sight_range)

        flat = np.concatenate([
            self._last_scan, direction, [distance],
        ]).astype(np.float32)

        coverage_map = self._build_coverage_map()

        return {
            "coverage_map": coverage_map,
            "flat": flat,
            # NEW: real full-resolution grid + drone position, for
            # pick_block.pick_block() to resolve target blocks against
            # (see observation_spec comment above).
            "density_map_full": self.full_res_active_need_map(),
            "drone_xy": self.drone.pos[:2].astype(np.float32),
        }

    def _remaining_regions(self):
        """Needs-water regions not yet completed/passed and not currently
        active -- i.e. everything still waiting in the queue."""
        return (
            self.spray_tracker.needs_water_regions
            - set(self._completed_regions_order)
            - {self.active_region_id}
        )

    def _region_vector(self, region_id):
        """(direction(3,), normalised distance) from the drone's current
        position to a region's footprint centroid. Returns a zero direction
        and max distance (1.0) if `region_id` is None -- e.g. no active
        region left, or nothing queued after the active one."""
        if region_id is None:
            return np.zeros(3, dtype=np.float32), 1.0
        centroid = self.spray_tracker.region_centroid(region_id)
        return lidar.get_target_vector(self.drone.pos, centroid, self._region_norm)

    def _is_in_region(self, region_id):
        """True if the drone's current (x, y) footprint cell belongs to
        `region_id`. Used for the in-active-region presence reward -- being
        physically inside the region it's supposed to be working, not just
        close to its centroid (a large or oddly-shaped region can have its
        centroid sit outside the region itself)."""
        if region_id is None:
            return False
        x, y = int(self.drone.pos[0]), int(self.drone.pos[1])
        labels = self.spray_tracker.labels
        if not (0 <= x < labels.shape[0] and 0 <= y < labels.shape[1]):
            return False
        return bool(labels[x, y] == region_id)

    def full_res_active_need_map(self):
        """(H, W) full-resolution "active-region spray need" map, at the
        real terrain resolution (self.spray_tracker.shape -- e.g. 30x30),
        NOT the coarser downscaled coverage_map channel 0 (e.g. 8x8) the
        network's density_direction head reads. Per cell:
          - 0.0  outside the active region entirely (any other region, or
                 no active region at all)
          - 0.0  inside the active region but already fully sprayed
                 (completeness >= 1) -- done cells stop mattering, same as
                 cells outside the region
          - fraction in (0, 1]  inside the active region and not yet fully
                 sprayed: 1.0 - completeness, i.e. how much spraying is
                 still needed there (1.0 = untouched, near 0 = nearly done)

        Exposed separately (rather than only living inline in
        _build_coverage_map) so callers like run_model_test.py can hand
        pick_block.pick_block() the REAL grid to choose a target block
        from, instead of the coarser one the network itself sees.
        """
        st = self.spray_tracker
        completeness = np.clip(
            st.hit_counts / max(st.required_hits_per_cell, 1), 0.0, 1.0
        ).astype(np.float32)
        remaining_need = 1.0 - completeness

        need_channel = np.zeros(st.shape, dtype=np.float32)
        if self.active_region_id is not None:
            active_mask = (st.labels == self.active_region_id)
            need_channel[active_mask] = remaining_need[active_mask]
        return need_channel

    def _build_coverage_map(self):
        """(2, h_out, w_out) tensor: two channels, each downscaled/averaged
        from the full (H, W) footprint down to h_out x w_out -- the same
        spatial size the old CNN used to reduce the map to internally via
        its two stride-2 conv layers (see _stride2_conv_out / self._map_out_hw).

        Channel 0 is full_res_active_need_map() above, downscaled.

        Channel 1 -- drone position. A single 1.0 at the drone's current
        (x, y) cell, 0 everywhere else, BEFORE downscaling -- i.e. a soft
        one-hot spike once pooled, since averaging spreads that single 1.0
        across however many full-res cells land in its output bin (so its
        peak value shrinks the coarser h_out/w_out is, but it stays the
        only nonzero region on the map). This puts the drone directly on
        the same grid as what needs spraying, so the network can read the
        relationship off the map itself instead of being handed a
        hand-computed direction/distance vector in `flat`.

        The downscaling itself is average pooling (torch's
        adaptive_avg_pool2d), which does exactly what plain reshaping
        can't when the map size doesn't divide evenly into h_out/w_out:
        it splits the input into h_out x w_out roughly-equal blocks and
        averages each one, so a block straddling both 0s and partial
        fractions (or straddling the position spike) comes out as their
        true mean rather than being rounded or truncated.
        """
        need_channel = self.full_res_active_need_map()

        # Drone position is a float (self.drone.pos can sit mid-cell) so
        # this clamps/rounds to the nearest valid cell rather than assuming
        # int(pos) is always safe to index.
        pos_channel = np.zeros(need_channel.shape, dtype=np.float32)
        px = int(np.clip(round(float(self.drone.pos[0])), 0, need_channel.shape[0] - 1))
        py = int(np.clip(round(float(self.drone.pos[1])), 0, need_channel.shape[1] - 1))
        pos_channel[px, py] = 1.0

        # (2, H, W) -> average-pool down to (2, h_out, w_out), each channel independently
        t = torch.from_numpy(np.stack([need_channel, pos_channel], axis=0)).unsqueeze(0)
        downscaled = F.adaptive_avg_pool2d(t, output_size=self._map_out_hw)
        return downscaled.squeeze(0).numpy()

    # --- reward tuning constants -- all in one place so they're easy to find/adjust ---
    CRASH_PENALTY = -8                 # hitting terrain/trees/out-of-bounds
    STEP_COST = 0.1                    # tiny per-step cost, discourages stalling
    OVERSPRAY_PENALTY_PER_HIT = 0.001     # per spray-ray hit landed on a no-water region
    SPRAY_HIT_REWARD_PER_HIT = 0.01     # per spray-ray hit landed on a needs-water region (up to its cap)
    PROGRESS_REWARD_SCALE = 0.0           # multiplier on step-to-step completeness delta
    REGION_COMPLETE_BONUS = 12.0          # one-time bonus, paid the step the ACTIVE (pink-line-targeted) region finishes
    OFF_TARGET_REGION_COMPLETE_BONUS = 6.0  # one-time bonus/penalty, paid the step a NON-active region finishes -- paid immediately, same step it crosses threshold, NOT deferred to whenever it later becomes active. Set >0 for partial credit, <0 to penalize wandering off-task, or leave at 0 (default) to only reward finishing the region it's pointed at.
    ALL_REGIONS_COMPLETE_BONUS = 15     # one-time bonus, paid the step EVERY region finishes
    IN_ACTIVE_REGION_REWARD = 0.00        # per-step reward while physically inside the active region's footprint
    APPROACH_REWARD_SCALE = 13.0           # multiplier on step-to-step closing distance to the active region

    # --- "stuck" detection: no real displacement over a window of moves ---
    STUCK_WINDOW = 15                     # look back this many moves
    STUCK_DISPLACEMENT_THRESHOLD = 3    # if bounding-box diagonal over the window is under this many blocks, it's "stuck"

    # --- REGION_DONE_THRESHOLD curriculum (staircase, not a continuous ramp) ---
    # Fraction of a region's cells that must be watered before it counts as
    # "done" and the drone moves on to the next one. Rather than sliding
    # smoothly from START to END as training progresses, the total training
    # run is chopped into REGION_DONE_THRESHOLD_INTERVALS equal-length
    # stages, and the threshold jumps by one fixed increment at the start
    # of each stage -- flat within a stage, step up at the boundary.
    # All three are decided/changeable here in one place:
    REGION_DONE_THRESHOLD_START = 0.01      # threshold used during stage 0 (start of training)
    REGION_DONE_THRESHOLD_END = 0.90        # threshold used during the final stage
    REGION_DONE_THRESHOLD_INTERVALS = 20     # number of stepped stages spanning the whole run

    @staticmethod
    def region_done_threshold_for_progress(progress, start=None, end=None, n_intervals=None):
        """Stepped/staircase REGION_DONE_THRESHOLD for a given training
        progress fraction in [0, 1] (e.g. frames_collected / total_steps).

        The [0, 1] progress range is split into `n_intervals` equal-length
        stages. The increment between consecutive stages is
        (end - start) / (n_intervals - 1), chosen so stage 0 lands exactly
        on `start` and the final stage lands exactly on `end`. The value
        is constant within a stage and only steps up at stage boundaries
        -- unlike a linear ramp, which changes every single step.

        Any of start/end/n_intervals can be overridden per-call; otherwise
        they default to the class constants above.
        """
        start = DroneEnv.REGION_DONE_THRESHOLD_START if start is None else start
        end = DroneEnv.REGION_DONE_THRESHOLD_END if end is None else end
        n_intervals = DroneEnv.REGION_DONE_THRESHOLD_INTERVALS if n_intervals is None else n_intervals

        if n_intervals <= 1:
            return float(end)

        progress = float(np.clip(progress, 0.0, 1.0))
        # which stage (0 .. n_intervals-1) this progress falls into
        stage = min(n_intervals - 1, int(progress * n_intervals))
        increment = (end - start) / (n_intervals - 1)
        return float(start + stage * increment)

    def _get_reward(self, prev_pos, current_step=0, max_steps=400):
        # Stuck (no real displacement over STUCK_WINDOW moves), a real
        # crash, AND running out of time (truncated) are all treated as the
        # same failure outcome: same flat penalty, and it skips the rest of
        # the reward calc. Timing out without finishing every needs-water
        # region is exactly as bad as flying into a tree -- the drone
        # failed the episode either way, so it shouldn't just collect its
        # ordinary per-step reward for the final step and walk away clean.
        if self._hit_something() or self._last_stuck or self._is_truncated():
            return self.CRASH_PENALTY

        st = self.spray_tracker

        # --- overspray penalty: discourage spraying no-water regions. This
        # stays global (not scoped to the active region) -- spraying the
        # wrong material is always wasteful, regardless of which
        # needs-water region the drone is currently assigned to.
        overspray_hits_total = sum(st.overspray_counts().values())
        overspray_penalty = self.OVERSPRAY_PENALTY_PER_HIT * overspray_hits_total

        # No active region left means every needs-water region has already
        # been finished and passed in sequence -- the all-done bonus already
        # paid out the step that happened, so there's nothing left to
        # reward except the ambient step cost / overspray penalty.
        if self.active_region_id is None:
            return float(-self.STEP_COST - overspray_penalty)

        # --- 1. progress reward: did the ACTIVE region's completeness improve? ---
        # (unlike before, this is scoped to a single region rather than an
        # area-weighted total across every needs-water region at once, since
        # the drone should only be working on one at a time.)
        completeness_all = st.region_completeness()
        active_completeness_now = completeness_all.get(self.active_region_id, 0.0)
        progress_reward = self.PROGRESS_REWARD_SCALE * (active_completeness_now - self._prev_active_completeness)

        # --- 2. spray-hit reward: direct, immediate credit for choosing to
        # spray a needs-water cell IN THE ACTIVE REGION, rather than waiting
        # for progress_reward or the one-time completion bonus. Only counts
        # hits up to each cell's required amount, so it can't be farmed by
        # oversaturating an already-done cell.
        useful_hits_now = st.useful_hits_for_region(self.active_region_id)
        new_useful_hits = max(useful_hits_now - self._prev_active_useful_hits, 0.0)
        spray_hit_reward = self.SPRAY_HIT_REWARD_PER_HIT * new_useful_hits

        # --- 3. in-active-region presence reward: flat per-step reward for
        # being physically inside the active region's footprint right now
        # (not just close to its centroid -- an oddly-shaped or large
        # region's centroid can sit outside the region itself). This is on
        # top of the completeness-driven rewards above so simply loitering
        # inside the region without spraying still isn't the best strategy,
        # but it does give a clean, unambiguous signal for "in bounds" vs
        # "wandered off" that doesn't depend on hitting anything.
        in_region_reward = self.IN_ACTIVE_REGION_REWARD if self._is_in_region(self.active_region_id) else 0.0

        # --- 4. approach reward: did the drone get closer to the active
        # region THIS step? Uses the same normalised centroid distance as
        # the "active region" observation vector, so it's on the same scale
        # the network already sees. This is what gives a navigation signal
        # right after a hand-off to a new region (see block 5 below) --
        # before the drone has even entered the new region's footprint, it
        # still gets rewarded for closing the distance to it, rather than
        # waiting on progress/spray rewards that only fire once it arrives.
        active_distance_now = self._region_vector(self.active_region_id)[1]
        approach_reward = self.APPROACH_REWARD_SCALE * (self._prev_active_distance - active_distance_now)

        # --- 5. region-complete bonus + hand-off to the next region ---
        # Check EVERY needs-water region that isn't already marked done --
        # not just the active one -- since a spray cone can incidentally
        # finish a region other than the one the drone is currently pointed
        # at (the pink target line). Finishing the ACTIVE region pays the
        # full REGION_COMPLETE_BONUS and hands off to the next target, same
        # as before. Finishing any OTHER region instead pays
        # OFF_TARGET_REGION_COMPLETE_BONUS (0 by default), paid THIS SAME
        # STEP -- not deferred -- and gets marked done immediately so it can
        # never later be assigned as active and silently cash in the full
        # bonus for work it already did off-task.
        #
        # IMPORTANT invariant (unchanged): active_region_id is ONLY ever
        # reassigned when the ACTIVE region finishes. Off-target completions
        # never trigger a hand-off by themselves.
        region_complete_bonus = 0.0
        all_done_bonus = 0.0
        newly_completed = [
            rid for rid in st.needs_water_regions
            if rid not in self._completed_regions_order
            and completeness_all.get(rid, 0.0) >= self.REGION_DONE_THRESHOLD
        ]
        for rid in newly_completed:
            self._completed_regions_order.append(rid)
            if rid == self.active_region_id:
                region_complete_bonus += self.REGION_COMPLETE_BONUS
            else:
                region_complete_bonus += self.OFF_TARGET_REGION_COMPLETE_BONUS

        if self.active_region_id in newly_completed:
            next_id, _ = st.closest_region(self._remaining_regions(), self.drone.pos[:2])
            self.active_region_id = next_id
            if next_id is None:
                all_done_bonus = self.ALL_REGIONS_COMPLETE_BONUS

        # --- save state for next step's deltas, scoped to whichever region
        # is active NOW (freshly switched-to regions start their deltas
        # from their own current numbers, not the region that just finished).
        if self.active_region_id is not None:
            self._prev_active_completeness = completeness_all.get(self.active_region_id, 0.0)
            self._prev_active_useful_hits = st.useful_hits_for_region(self.active_region_id)
            self._prev_active_distance = self._region_vector(self.active_region_id)[1]
        else:
            self._prev_active_completeness = 0.0
            self._prev_active_useful_hits = 0.0
            self._prev_active_distance = 0.0

        reward = (
            progress_reward
            + region_complete_bonus
            + all_done_bonus
            + spray_hit_reward
            + in_region_reward
            + approach_reward
            - overspray_penalty
            - self.STEP_COST
        )
        # return float(np.clip(reward, -1.0, 1.0))
        return float(reward)

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
        real displacement over STUCK_WINDOW moves), or worked through every
        needs-water region in sequence (active_region_id goes to None once
        the last one is finished and there's nothing left to hand off to).
        Nothing to do with step count."""
        return (
            self._hit_something()
            or self._last_stuck
            or self.active_region_id is None
        )

    def _is_truncated(self):
        """Ran out of time: hit max_steps without crashing or finishing."""
        return self.current_step >= self.max_steps

    def _set_seed(self, seed):
        np.random.seed(seed)