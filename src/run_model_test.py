import os
os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"
os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"

import random
import numpy as np
import torch
from torch import nn
import open3d as o3d
import matplotlib.pyplot as plt

from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torchrl.modules import ProbabilisticActor, TanhNormal
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.data import Bounded

import open3d_sim_env as sim
from drone_env import DroneEnv
from drone import Drone
import pick_block

MAX_STEPS = 400
num_cells = 256          # must match rl_model.py
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "..", "pax_v2_5.pt")
FRAMES_PER_STEP = 10      # adjust this to slow down: higher = slower (1=normal speed)
TEST_REGION_DONE_THRESHOLD = 0.90


# ── rebuild model architecture (must match rl_model.py exactly) ──────────────
# Copied verbatim from rl_model.py's MapActorNet -- this HAS to match the
# trained architecture exactly, or load_state_dict will fail / silently
# load into the wrong shapes.
#
# CHANGED: no more `map_feat` embedding of coverage_map fed into the trunk.
# Instead, coverage_map's channel 0 (downscaled spray-need density) is used
# to pick a t in [0, 1] via density_direction, and pick_block.pick_block()
# resolves that t into a concrete block. That block's (direction, distance)
# -- `block_loc` -- is what actually feeds the trunk, concatenated with
# flat_feat.
#
# NEW: pick_block resolves its block against density_map_full/drone_xy --
# DroneEnv's REAL full-resolution grid (e.g. 30x30) and the drone's real
# (x, y) -- not the coarse downscaled coverage_map channels. Both travel
# through the tensordict as their own observation keys (see
# DroneEnv.observation_spec), same as rl_model.py -- so block_rc coming out
# of pick_block is already a world (x, y) cell, no separate rescaling
# needed for the pink pointer line.
class MapActorNet(nn.Module):
    def __init__(self, num_cells, action_dim, device=None):
        super().__init__()

        self.density_direction = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, device=device),
            nn.ReLU(),

            nn.Conv2d(16, 32, 3, padding=1, device=device),
            nn.ReLU(),

            nn.Flatten(),

            nn.LazyLinear(64, device=device),
            nn.ReLU(),

            nn.Linear(64, 1, device=device),
            nn.Sigmoid(),
        )

        self.flat_branch = nn.Sequential(
            nn.LazyLinear(num_cells, device=device),
            nn.ReLU(),
        )

        self.trunk = nn.Sequential(
            nn.LazyLinear(num_cells, device=device),
            nn.Tanh(),
            nn.LazyLinear(num_cells, device=device),
            nn.Tanh(),
            nn.LazyLinear(num_cells, device=device),
            nn.Tanh(),
            nn.LazyLinear(num_cells, device=device),
            nn.Tanh(),
            nn.LazyLinear(num_cells, device=device),
            nn.Tanh(),
            nn.LazyLinear(2 * action_dim, device=device),
        )

        self.extractor = NormalParamExtractor()

        # NEW: world (x, y) of the block pick_block() picked on the most
        # recent forward() call, or None if nothing valid was picked. This
        # is what the pink line now points to, instead of the old
        # active-region centroid.
        self.last_target_xy = None
        self._last_picked_blocks = None

    def _target_xy_from_block(self, block_rc):
        """Convert a (row, col) picked by pick_block.pick_block() into a
        world (x, y) point on the terrain. Since pick_block now operates
        directly on the full-resolution density grid -- the same
        coordinate system as the drone's world position -- block_rc
        already IS the world cell; this just recenters it (+0.5, matching
        the +0.5 cell-centre offset used elsewhere, e.g. make_drone_mesh /
        path_points), with no rescaling needed.

        Returns None if no block was picked.
        """
        if block_rc is None:
            return None
        row, col = block_rc
        return np.array([float(row) + 0.5, float(col) + 0.5], dtype=np.float32)

    def _block_loc_batch(self, density_full_batch, drone_xy_batch, t_batch):
        """density_full_batch: (N, H, W) tensor -- DroneEnv's REAL
        full-resolution active-region need map (density_map_full),
        flattened over any leading batch dims. drone_xy_batch: (N, 2)
        tensor -- the drone's real (x, y) for each of those N steps.
        t_batch: (N, 1) tensor in [0, 1] from density_direction.

        Returns an (N, 3) tensor of [direction_x, direction_y, distance]
        -- pick_block.pick_block() works on raw numpy grids/positions, not
        batched torch tensors, so this loops per-sample.
        """
        density_np = density_full_batch.detach().cpu().numpy()
        pos_np = drone_xy_batch.detach().cpu().numpy()
        t_np = t_batch.detach().cpu().numpy().reshape(-1)

        rows = []
        # NEW: (row, col) each sample's pick_block() call actually picked,
        # or None -- lets forward() below convert sample 0's pick into a
        # world-space target for the pink pointer line. run_model_test.py
        # only ever runs the policy on a single env, so index 0 is "the"
        # pick for whatever step is currently being visualised.
        picked_blocks = []
        for i in range(density_np.shape[0]):
            result = pick_block.pick_block(density_np[i], pos_np[i], float(t_np[i]))
            if result is None:
                rows.append(np.array([0.0, 0.0, 1.0], dtype=np.float32))
                picked_blocks.append(None)
            else:
                direction, distance = result["direction"], result["distance"]
                rows.append(np.array([direction[0], direction[1], distance], dtype=np.float32))
                picked_blocks.append(result["block"])

        self._last_picked_blocks = picked_blocks
        return torch.tensor(np.stack(rows), dtype=torch.float32, device=density_full_batch.device)

    def forward(self, coverage_map, flat, density_map_full, drone_xy):
        # coverage_map: (*batch, 2, H, W) -- channel 0 = spray-need
        # density (downscaled), channel 1 = drone position (downscaled,
        # unused here -- drone_xy below is the real position instead).
        density_channel = coverage_map[..., 0, :, :]

        batch_shape = density_channel.shape[:-2]
        density_flat = density_channel.reshape(-1, *density_channel.shape[-2:])

        # density_map_full/drone_xy carry the same leading batch dims as
        # coverage_map/flat -- flatten them the same way so rows line up
        # 1:1 with density_flat/t below.
        density_full_flat = density_map_full.reshape(-1, *density_map_full.shape[-2:])
        drone_xy_flat = drone_xy.reshape(-1, drone_xy.shape[-1])

        chosen_t = self.density_direction(density_flat.unsqueeze(1))  # (N, 1)
        block_loc = self._block_loc_batch(density_full_flat, drone_xy_flat, chosen_t)  # (N, 3)
        block_loc = block_loc.reshape(*batch_shape, 3)

        # NEW: remember sample 0's pick as a world (x, y) point, for the
        # pink pointer line in run_model_test.py's visualisation.
        picked = self._last_picked_blocks[0] if self._last_picked_blocks else None
        self.last_target_xy = self._target_xy_from_block(picked)

        flat_feat = self.flat_branch(flat)
        fused = torch.cat([flat_feat, block_loc], dim=-1)
        out = self.trunk(fused)
        return self.extractor(out)


def build_policy(action_dim, device):
    actor_net = MapActorNet(num_cells=num_cells, action_dim=action_dim, device=device)

    policy_module = TensorDictModule(
        actor_net, in_keys=["coverage_map", "flat", "density_map_full", "drone_xy"], out_keys=["loc", "scale"]
    )

    action_spec = Bounded(low=-1, high=1, shape=(action_dim,))

    policy_module = ProbabilisticActor(
        module=policy_module,
        spec=action_spec,
        in_keys=["loc", "scale"],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": action_spec.space.low,
            "high": action_spec.space.high,
        },
        return_log_prob=True,
    )

    return policy_module, actor_net


def build_world():
    """Same as rl_model.py's build_world() -- fresh random terrain via
    open3d_sim_env, with an explicit seed to avoid generate_voxels()'s
    default-argument seeding bug (seed=random.randint(...) is evaluated
    once at import time, not per call)."""
    seed = random.randint(0, 10000)
    voxels = sim.generate_voxels(seed=seed)
    drone = sim.spawn_drone(voxels)
    start = drone.get_position()
    end = start.copy()  # no navigation goal in the watering task
    clean = voxels.copy()
    return start, end, clean


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    # ── generate world ────────────────────────────────────────────────────
    start, end, clean_grid = build_world()
    print(f"Start: {start}")

    env = DroneEnv(clean_grid.copy(), start, end, region_done_threshold=TEST_REGION_DONE_THRESHOLD)

    # ── build & load policy ───────────────────────────────────────────────
    action_dim = env.action_spec.shape[-1]
    policy, actor_net = build_policy(action_dim, device)

    # NOTE: no extra wiring needed here anymore -- density_map_full/drone_xy
    # (DroneEnv's real full-resolution grid + drone position) travel through
    # the tensordict as their own observation keys, same as coverage_map/flat.

    # warm up lazy layers so state_dict loads cleanly
    dummy = env.reset().to(device)
    with torch.no_grad():
        policy(dummy)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    policy.load_state_dict(checkpoint["policy"])
    policy.eval()
    print("Model loaded.")


    # ── run rollout, collect path + spray history ───────────────────────────
    td = env.reset().to(device)
    path_history = [env.drone.pos.copy()]
    # NEW: coverage_map history -- one (1, h_out, w_out) array per step,
    # straight from the tensordict the env already returns. Channel 0 is
    # the active-region "spray-need" map (0 = outside the active region or
    # already fully sprayed, fraction = how much spraying is still needed),
    # already downscaled/averaged by DroneEnv -- nothing left to capture
    # from inside the network since there's no conv stack transforming it
    # anymore. Used to drive a live 2D viewer kept in lockstep with the 3D
    # playback below, no recomputation needed.
    coverage_history = [td["coverage_map"].cpu().numpy()]
    # NEW: full-resolution active-region need map (e.g. 30x30) history --
    # the REAL grid pick_block.pick_block() now chooses a target block
    # from (see MapActorNet._block_loc_batch), as opposed to coverage_map's
    # channel 0 above, which is the coarser downscaled version the network
    # itself reads. Tracked separately so the live viewer can show both.
    full_res_history = [env.full_res_active_need_map()]
    # NEW: which region is active, and its TRUE region-wide completeness
    # fraction (region_completeness()), at each step. coverage_history[i][0]
    # is scoped to the active region already (0 elsewhere), but it's still
    # useful to track the exact scalar fraction directly rather than
    # eyeballing the heatmap.
    active_id_history = [env.active_region_id]
    active_completeness_history = [
        env.spray_tracker.region_completeness().get(env.active_region_id, 0.0)
        if env.active_region_id is not None else None
    ]
    # region labels are only rebuilt on _reset(), so this is constant for the
    # whole episode -- safe to snapshot once here.
    region_labels = env.labels.copy()
    spray_history = []   # hit_xyz arrays, one per step, for the lineset animation
    spray_steps = 0      # how many steps the policy actually chose to spray on
    total_reward = 0.0   # NEW: cumulative reward across the rollout

    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        # NEW: the pink pointer line now points at whichever block
        # pick_block.pick_block() actually picked from the model's own
        # density/position input (MapActorNet.last_target_xy), not the
        # active region's centroid. Run the policy once against the reset
        # state purely to seed target_history[0] with that pick -- the loop
        # below already calls policy(td) once per step for the real action,
        # which is reused for target_history[step + 1].
        policy(td.to(device))
        target_history = [actor_net.last_target_xy]

        for step in range(MAX_STEPS):
            td = policy(td.to(device))
            action = td["action"].cpu().numpy()
            spray_flag = action[..., 3] > 0
            if spray_flag:
                spray_steps += 1

            td = env._step(td)
            reward = td["reward"].item()
            total_reward += reward

            coverage_history.append(td["coverage_map"].cpu().numpy())
            full_res_history.append(env.full_res_active_need_map())
            active_id_history.append(env.active_region_id)
            active_completeness_history.append(
                env.spray_tracker.region_completeness().get(env.active_region_id, 0.0)
                if env.active_region_id is not None else None
            )
            path_history.append(env.drone.pos.copy())
            # actor_net.last_target_xy was set by the policy(td) call above,
            # which is what produced this step's action -- so it's still
            # the right "this is what the model was aiming at" pick to pair
            # with the position we just moved to.
            target_history.append(actor_net.last_target_xy)

            # Re-fire spray purely for the visual lineset -- DroneEnv already
            # registered the real hits internally during _step(); this call
            # doesn't touch spray_tracker, it's just to get ray endpoints to draw.
            if spray_flag and not env._hit_something():
                spray_history.append(env.drone.spray())
            else:
                spray_history.append(np.zeros((0, 3)))

            done = td["done"].item()
            hit = env._hit_something()
            comp = env.spray_tracker.needs_water_completeness()
            mean_comp = float(np.mean(list(comp.values()))) if comp else 1.0

            if step % 10 == 0:
                print(f"  step {step:4d} | coverage: {mean_comp:.2%} | hit: {hit} | done: {done} "
                      f"| reward: {reward:+.4f} | total: {total_reward:+.4f}")

            if done:
                if hit:
                    print(f"Crashed at step {step}.")
                elif env.spray_tracker.all_needs_water_done():
                    print(f"All crops watered at step {step}!")
                else:
                    print(f"Episode ended at step {step}.")

                # --- diagnostics: how big was the needs-water footprint,
                # and what fraction of it actually got sprayed (area-weighted,
                # not averaged per-region), plus how often the policy chose
                # to spray at all during the rollout. ---
                st = env.spray_tracker
                cell_counts = st.region_cell_counts()  # {region_id: (total, watered)}
                needs_total = sum(t for rid, (t, w) in cell_counts.items()
                                   if rid in st.needs_water_regions)
                needs_watered = sum(w for rid, (t, w) in cell_counts.items()
                                     if rid in st.needs_water_regions)
                map_total = st.shape[0] * st.shape[1]
                block_frac = needs_watered / needs_total if needs_total > 0 else 1.0

                print(f"  needs-water cells watered: {needs_watered}/{needs_total} "
                      f"({block_frac:.2%}, area-weighted)")
                print(f"  needs-water footprint: {needs_total}/{map_total} cells "
                      f"({needs_total / map_total:.2%} of the map)")
                print(f"  sprayed on {spray_steps}/{step + 1} steps "
                      f"({spray_steps / (step + 1):.2%} of the episode)")
                print(f"  total reward for this rollout: {total_reward:+.4f} "
                      f"({total_reward / (step + 1):+.4f} avg/step)")
                break

            td = td.select("coverage_map", "flat", "density_map_full", "drone_xy").to(device)
        else:
            # loop exhausted MAX_STEPS without the env ever reporting done
            # (e.g. MAX_STEPS set lower than env.max_steps) -- still report
            # what was accumulated so this isn't silently skipped.
            print(f"Reached MAX_STEPS ({MAX_STEPS}) without the env reporting done.")
            print(f"  total reward for this rollout: {total_reward:+.4f} "
                  f"({total_reward / MAX_STEPS:+.4f} avg/step)")

    print(f"Rollout finished — {len(path_history)} positions recorded.")

    # ── live animated visualisation (Open3D, matching open3d_sim_env.py's
    # run_simulation() pattern -- no pyvista here, this codebase uses Open3D) ──
    terrain_geoms = [sim.voxelgrid_from_voxels(env.values), sim.voxel_edges_lineset(env.values)]

    vis_drone = Drone(0, 10, env.values, start, end)
    drone_mesh = sim.make_drone_mesh(vis_drone)
    spray_lineset = sim.make_spray_lineset(vis_drone.get_position(), np.zeros((0, 3)))

    # --- NEW: pink pointer line from the drone to whichever block
    # pick_block.pick_block() picked from the model's own density/position
    # input this step (MapActorNet.last_target_xy), converted to world (x, y).
    target_lineset = sim.make_target_lineset(vis_drone.get_position(), target_history[0])

    # --- NEW: trail LineSet tracing the drone's flight path so far.
    # Starts with just the initial point (no segments yet); advance()
    # appends a new segment each frame as the drone moves. +0.5 matches
    # the same voxel-center offset make_drone_mesh() already uses.
    path_points = [np.array(start, dtype=np.float32) + 0.5]
    path_lineset = o3d.geometry.LineSet()
    path_lineset.points = o3d.utility.Vector3dVector(np.array(path_points))
    path_lineset.lines = o3d.utility.Vector2iVector(np.zeros((0, 2), dtype=np.int32))
    path_lineset.colors = o3d.utility.Vector3dVector(np.zeros((0, 3)))
    PATH_COLOR = (0.1, 0.4, 1.0)  # blue, distinct from spray's green

    # ── live coverage-map viewer (matplotlib, separate window) ──────────────
    # Three panels now: channel 0, the downscaled active-region "spray-need"
    # map the network's density_direction head actually reads (0 outside
    # the active region or already fully sprayed there, fraction = how
    # much spraying is still needed); channel 1, the downscaled
    # drone-position spike; and the REAL full-resolution density grid
    # (e.g. 30x30) that pick_block.pick_block() now resolves its target
    # block from (see MapActorNet._block_loc_batch) -- shown at its true
    # resolution so it's clear the block choice is made on the real grid,
    # not the coarse one the network sees. Updated in lockstep with the 3D
    # playback's advance() callback below.
    plt.ion()
    cov_fig, (cov_ax_need, cov_ax_pos, cov_ax_full) = plt.subplots(1, 3, figsize=(13.5, 4.5))
    cov_fig.canvas.manager.set_window_title("Coverage Map (live)")

    cov0 = coverage_history[0]
    im_need = cov_ax_need.imshow(cov0[0].T, origin="lower", cmap="viridis", vmin=0.0, vmax=1.0)
    cov_ax_need.set_title("Active-region spray need (downscaled)")
    cov_fig.colorbar(im_need, ax=cov_ax_need, fraction=0.046)

    im_pos = cov_ax_pos.imshow(cov0[1].T, origin="lower", cmap="viridis", vmin=0.0, vmax=cov0[1].max() or 1.0)
    cov_ax_pos.set_title("Drone position (downscaled)")
    cov_fig.colorbar(im_pos, ax=cov_ax_pos, fraction=0.046)

    full0 = full_res_history[0]
    im_full = cov_ax_full.imshow(full0.T, origin="lower", cmap="viridis", vmin=0.0, vmax=1.0)
    cov_ax_full.set_title(f"Full-res spray need ({full0.shape[0]}x{full0.shape[1]})")
    cov_fig.colorbar(im_full, ax=cov_ax_full, fraction=0.046)

    cov_suptitle = cov_fig.suptitle("step 0")
    cov_fig.tight_layout()
    cov_fig.canvas.draw()
    plt.pause(0.001)

    def _update_coverage_view(step_i):
        """Redraw the live matplotlib panels for coverage_history[step_i].
        No-ops quietly if the person closed the coverage window.

        Since the spray-need channel is already scoped to just the active
        region (0 outside it, or once a cell there is fully sprayed), it
        doesn't need the region-boundary contour overlay the old global
        completion channel needed -- what's on screen already only lights
        up inside the active region's real footprint. Still prints the
        true region_completeness() fraction in the title since the map's
        resolution is coarse (downscaled/averaged) and the exact scalar is
        more precise than eyeballing it.

        The position panel's color scale is re-normalised to that step's
        own max each redraw (imshow's vmax below, set once at init to the
        first frame's peak, is left as a starting point) since the spike's
        peak value shrinks/shifts slightly depending on exactly which
        output bin the drone's cell pools into.
        """
        if not plt.fignum_exists(cov_fig.number):
            return
        if step_i >= len(coverage_history):
            return
        cov = coverage_history[step_i]
        comp_frac = active_completeness_history[step_i] if step_i < len(active_completeness_history) else None

        cov_ax_need.clear()
        cov_ax_need.imshow(cov[0].T, origin="lower", cmap="viridis", vmin=0.0, vmax=1.0)
        comp_str = f"{comp_frac:.1%}" if comp_frac is not None else "no active region -- all done"
        cov_ax_need.set_title(f"Active-region spray need (downscaled, completeness: {comp_str})")

        cov_ax_pos.clear()
        cov_ax_pos.imshow(cov[1].T, origin="lower", cmap="viridis", vmin=0.0, vmax=float(cov[1].max()) or 1.0)
        cov_ax_pos.set_title("Drone position (downscaled)")

        if step_i < len(full_res_history):
            full_map = full_res_history[step_i]
            cov_ax_full.clear()
            cov_ax_full.imshow(full_map.T, origin="lower", cmap="viridis", vmin=0.0, vmax=1.0)
            cov_ax_full.set_title(f"Full-res spray need ({full_map.shape[0]}x{full_map.shape[1]})")

        cov_suptitle.set_text(f"step {step_i}")
        cov_fig.canvas.draw_idle()
        cov_fig.canvas.flush_events()

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Drone Rollout Playback", width=1024, height=768)
    for geom in terrain_geoms:
        vis.add_geometry(geom)
    vis.add_geometry(drone_mesh)
    vis.add_geometry(spray_lineset)
    vis.add_geometry(path_lineset)
    vis.add_geometry(target_lineset)
    vis.get_render_option().line_width = 4.0

    state = {"paused": False, "step_index": 0, "last_pos": np.array(start, dtype=np.float32), "frame_count": 0}

    def toggle_pause(vis_):
        state["paused"] = not state["paused"]
        print("-- paused --" if state["paused"] else "-- resumed --")
        return False

    vis.register_key_callback(ord(" "), toggle_pause)

    def advance(vis_):
        if state["paused"]:
            return False
        state["frame_count"] += 1
        if state["frame_count"] < FRAMES_PER_STEP:
            return False
        state["frame_count"] = 0
        i = state["step_index"]
        if i >= len(path_history):
            return False

        new_pos = path_history[i].astype(np.float32)
        delta = new_pos - state["last_pos"]
        drone_mesh.translate(delta)
        vis_.update_geometry(drone_mesh)
        state["last_pos"] = new_pos.copy()

        hits = spray_history[i - 1] if 0 < i <= len(spray_history) else np.zeros((0, 3))
        fresh_lineset = sim.make_spray_lineset(new_pos, hits)
        spray_lineset.points = fresh_lineset.points
        spray_lineset.lines = fresh_lineset.lines
        spray_lineset.colors = fresh_lineset.colors
        vis_.update_geometry(spray_lineset)

        # --- NEW: re-point the pink target line at this step's picked
        # block (target_history[i] lines up with path_history[i], both
        # recorded once per step -- see the rollout loop above).
        target_xy = target_history[i] if i < len(target_history) else None
        fresh_target = sim.make_target_lineset(new_pos, target_xy)
        target_lineset.points = fresh_target.points
        target_lineset.lines = fresh_target.lines
        target_lineset.colors = fresh_target.colors
        vis_.update_geometry(target_lineset)

        # --- NEW: extend the path trail with a segment from the previous
        # point to this one. Rebuilding points/lines/colors each frame is
        # cheap at this scale (at most a few hundred points for MAX_STEPS=600).
        path_points.append(new_pos + 0.5)
        path_lineset.points = o3d.utility.Vector3dVector(np.array(path_points))
        segments = [[j, j + 1] for j in range(len(path_points) - 1)]
        path_lineset.lines = o3d.utility.Vector2iVector(np.array(segments, dtype=np.int32))
        path_lineset.colors = o3d.utility.Vector3dVector(
            np.tile(PATH_COLOR, (len(segments), 1))
        )
        vis_.update_geometry(path_lineset)

        # NEW: keep the live coverage-map window in lockstep with this step.
        _update_coverage_view(i)

        state["step_index"] += 1
        return True

    vis.register_animation_callback(advance)
    print("Press SPACE to pause/resume playback. Rotate/zoom/pan anytime — close window to exit.")
    vis.run()
    vis.destroy_window()
    if plt.fignum_exists(cov_fig.number):
        plt.close(cov_fig)


if __name__ == "__main__":
    main()