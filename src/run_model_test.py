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

MAX_STEPS = 400
num_cells = 512          # must match rl_model.py
MAP_CHANNELS = 3         # must match drone_env.py's coverage_map channel count (material, completion, drone position)
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "..", "pax_v2_3.pt")
FRAMES_PER_STEP = 20      # adjust this to slow down: higher = slower (1=normal speed)
TEST_REGION_DONE_THRESHOLD = 0.50


# ── rebuild model architecture (must match rl_model.py exactly) ──────────────
# Copied verbatim from rl_model.py's CNNActorNet -- this HAS to match the
# trained architecture exactly, or load_state_dict will fail / silently
# load into the wrong shapes.
class CNNActorNet(nn.Module):
    def __init__(self, num_cells, action_dim, map_channels=3, device=None):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.LazyConv2d(16, kernel_size=3, stride=1, padding=1, device=device),
            nn.ReLU(),
            nn.LazyConv2d(32, kernel_size=3, stride=2, padding=1, device=device),
            nn.ReLU(),
            nn.LazyConv2d(32, kernel_size=3, stride=2, padding=1, device=device),
            nn.ReLU(),
            nn.Flatten(),
            nn.LazyLinear(num_cells, device=device),
            nn.ReLU(),
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

    def forward(self, coverage_map, flat):
        batch_shape = coverage_map.shape[:-3]
        coverage_map_flat = coverage_map.reshape(-1, *coverage_map.shape[-3:])
        map_feat = self.cnn(coverage_map_flat)
        map_feat = map_feat.reshape(*batch_shape, -1)

        flat_feat = self.flat_branch(flat)
        fused = torch.cat([map_feat, flat_feat], dim=-1)
        out = self.trunk(fused)
        return self.extractor(out)


def build_policy(action_dim, device):
    actor_net = CNNActorNet(num_cells=num_cells, action_dim=action_dim,
                             map_channels=MAP_CHANNELS, device=device)

    policy_module = TensorDictModule(
        actor_net, in_keys=["coverage_map", "flat"], out_keys=["loc", "scale"]
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

    # warm up lazy layers so state_dict loads cleanly
    dummy = env.reset().to(device)
    with torch.no_grad():
        policy(dummy)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    policy.load_state_dict(checkpoint["policy"])
    policy.eval()
    print("Model loaded.")

    # ── NEW: hook the CNN's last conv/ReLU block (index 5 of actor_net.cnn --
    # the ReLU right after the third LazyConv2d, BEFORE Flatten/LazyLinear)
    # so we can grab the spatial feature maps the network actually produces
    # from coverage_map, not just its final flattened num_cells vector.
    # Shape out of this layer is (batch, 32, H/4, W/4) given the two
    # stride-2 convs -- we keep it as-is (no upsampling) and just display it
    # at its native resolution.
    _last_cnn_feat = {}

    def _capture_cnn_feat(module, inputs, output):
        _last_cnn_feat["value"] = output.detach()

    actor_net.cnn[5].register_forward_hook(_capture_cnn_feat)

    # ── run rollout, collect path + spray history ───────────────────────────
    td = env.reset().to(device)
    path_history = [env.drone.pos.copy()]
    # NEW: coverage_map history -- one (3, H, W) array per step, straight from
    # the tensordict the env already returns (channel 0 = material/active-region,
    # channel 1 = spray-completion fraction, channel 2 = drone position). Used
    # to drive a live 2D viewer kept in lockstep with the 3D playback below,
    # no recomputation needed.
    coverage_history = [td["coverage_map"].cpu().numpy()]
    # NEW: one (32, H/4, W/4) feature-map array per step, captured from the
    # CNN's last conv block via the hook above. cnn_feat_history[i] is what
    # the network internally produced FROM coverage_history[i] -- populated
    # here for the reset state (index 0), then appended in-loop right after
    # each policy(td) call below (which is exactly when the hook fires for
    # that step's coverage_map).
    with torch.no_grad():
        actor_net(td["coverage_map"], td["flat"])
    cnn_feat_history = [_last_cnn_feat["value"].cpu().numpy()[0]]
    # NEW: which region is active, and its TRUE region-wide completeness
    # fraction (region_completeness()), at each step. The completion channel
    # (coverage_history[i][1]) is a GLOBAL hit-count map -- a small, densely
    # -sprayed patch can look "100% done" there while the rest of a large or
    # oddly-shaped active region (see coverage_history[i][0]) still sits at
    # 0 hits, indistinguishable from background. This is the number that
    # actually gates the region-complete hand-off, so it's worth tracking
    # separately rather than eyeballing the heatmap.
    active_id_history = [env.active_region_id]
    active_completeness_history = [
        env.spray_tracker.region_completeness().get(env.active_region_id, 0.0)
        if env.active_region_id is not None else None
    ]
    # region labels are only rebuilt on _reset(), so this is constant for the
    # whole episode -- safe to snapshot once here.
    region_labels = env.labels.copy()
    spray_history = []   # hit_xyz arrays, one per step, for the lineset animation
    # NEW: active-region centroid at each step, for the pink pointer line
    # showing where the drone is currently trying to go. None once every
    # needs-water region has been finished (active_region_id goes to None).
    def _active_target():
        rid = env.active_region_id
        return env.spray_tracker.region_centroid(rid) if rid is not None else None
    target_history = [_active_target()]
    spray_steps = 0      # how many steps the policy actually chose to spray on
    total_reward = 0.0   # NEW: cumulative reward across the rollout

    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        for step in range(MAX_STEPS):
            td = policy(td.to(device))
            # hook fired during the policy(td) call just above, capturing
            # the CNN's response to THIS step's coverage_map (same one that
            # was appended to coverage_history at the end of the previous
            # iteration / at reset) -- so this stays index-aligned with
            # coverage_history despite being appended one line early.
            cnn_feat_history.append(_last_cnn_feat["value"].cpu().numpy()[0])
            action = td["action"].cpu().numpy()
            spray_flag = action[..., 3] > 0
            if spray_flag:
                spray_steps += 1

            td = env._step(td)
            reward = td["reward"].item()
            total_reward += reward

            coverage_history.append(td["coverage_map"].cpu().numpy())
            active_id_history.append(env.active_region_id)
            active_completeness_history.append(
                env.spray_tracker.region_completeness().get(env.active_region_id, 0.0)
                if env.active_region_id is not None else None
            )
            path_history.append(env.drone.pos.copy())
            target_history.append(_active_target())

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

            td = td.select("coverage_map", "flat").to(device)
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

    # --- NEW: pink pointer line from the drone to whichever region it's
    # currently trying to reach (DroneEnv.active_region_id's centroid).
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
    # Four panels: channel 0 (material -- 0.3 no-water, 0.6 needs-water
    # queued, 1.0 active region), channel 1 (spray-completion fraction,
    # 0-1), channel 2 (drone position -- a single 1.0 cell), and NEW: the
    # CNN's own internal feature map for this step -- the mean across its
    # 32 output channels from the last conv block (actor_net.cnn[5]),
    # before Flatten/LazyLinear. This is at the CNN's native (downsampled)
    # resolution, not upsampled back to the map's H x W, since that's
    # literally the spatial resolution the policy is reasoning at. Updated
    # in lockstep with the 3D playback's advance() callback below, so all
    # four windows step forward together.
    plt.ion()
    cov_fig, (cov_ax_mat, cov_ax_comp, cov_ax_pos, cov_ax_cnn) = plt.subplots(1, 4, figsize=(17, 4.5))
    cov_fig.canvas.manager.set_window_title("Coverage Map (live)")

    cov0 = coverage_history[0]
    im_material = cov_ax_mat.imshow(cov0[0].T, origin="lower", cmap="viridis", vmin=0.0, vmax=1.0)
    cov_ax_mat.set_title("Material / active region")
    cov_fig.colorbar(im_material, ax=cov_ax_mat, fraction=0.046)

    im_completion = cov_ax_comp.imshow(cov0[1].T, origin="lower", cmap="viridis", vmin=0.0, vmax=1.0)
    cov_ax_comp.set_title("Spray completion")
    cov_fig.colorbar(im_completion, ax=cov_ax_comp, fraction=0.046)

    im_position = cov_ax_pos.imshow(cov0[2].T, origin="lower", cmap="viridis", vmin=0.0, vmax=1.0)
    cov_ax_pos.set_title("Drone position")
    cov_fig.colorbar(im_position, ax=cov_ax_pos, fraction=0.046)

    cnn0 = cnn_feat_history[0].mean(axis=0)  # (32, h', w') -> (h', w')
    im_cnn = cov_ax_cnn.imshow(cnn0.T, origin="lower", cmap="magma")
    cov_ax_cnn.set_title("CNN feature map (mean of 32 ch.)")
    cov_fig.colorbar(im_cnn, ax=cov_ax_cnn, fraction=0.046)

    cov_suptitle = cov_fig.suptitle("step 0")
    cov_fig.tight_layout()
    cov_fig.canvas.draw()
    plt.pause(0.001)

    def _update_coverage_view(step_i):
        """Redraw the live matplotlib panels for coverage_history[step_i].
        No-ops quietly if the person closed the coverage window.

        The completion panel (channel 1) is a GLOBAL hit-count map, so a
        region's actual boundary isn't visible in it -- a densely-sprayed
        patch can look fully yellow while most of a larger/odd-shaped
        region sits untouched just outside that patch, still reading as
        the same dark "0" background. To make that visible instead of
        misleading, this draws a red contour of the ACTIVE region's real
        footprint (from the constant region_labels grid) on top of the
        completion heatmap, and prints its true region_completeness()
        fraction in the title -- that fraction, not how full the yellow
        blob looks, is what actually gates the region-complete hand-off.

        CHANGED: the drone's position used to be drawn as a manually-added
        dot overlay on the material/completion panels, since coverage_map
        itself had no notion of "where the drone is". Now that the env
        bakes drone position into channel 2 directly, this panel shows
        exactly that raw channel instead -- what's on screen is literally
        the same array values the policy is looking at, same as the other
        two panels, rather than a debug annotation layered on top.
        """
        if not plt.fignum_exists(cov_fig.number):
            return
        if step_i >= len(coverage_history):
            return
        cov = coverage_history[step_i]
        active_id = active_id_history[step_i] if step_i < len(active_id_history) else None
        comp_frac = active_completeness_history[step_i] if step_i < len(active_completeness_history) else None

        cov_ax_mat.clear()
        cov_ax_mat.imshow(cov[0].T, origin="lower", cmap="viridis", vmin=0.0, vmax=1.0)
        cov_ax_mat.set_title("Material / active region")

        cov_ax_comp.clear()
        cov_ax_comp.imshow(cov[1].T, origin="lower", cmap="viridis", vmin=0.0, vmax=1.0)
        if active_id is not None:
            mask = (region_labels == active_id).astype(float)
            cov_ax_comp.contour(mask.T, levels=[0.5], colors="red", linewidths=1.5)
            comp_str = f"{comp_frac:.1%}" if comp_frac is not None else "?"
            cov_ax_comp.set_title(f"Spray completion (active region: {comp_str})")
        else:
            cov_ax_comp.set_title("Spray completion (no active region -- all done)")

        cov_ax_pos.clear()
        cov_ax_pos.imshow(cov[2].T, origin="lower", cmap="viridis", vmin=0.0, vmax=1.0)
        cov_ax_pos.set_title("Drone position")

        cov_ax_cnn.clear()
        if step_i < len(cnn_feat_history):
            feat = cnn_feat_history[step_i].mean(axis=0)  # (32, h', w') -> (h', w')
            cov_ax_cnn.imshow(feat.T, origin="lower", cmap="magma")
        cov_ax_cnn.set_title("CNN feature map (mean of 32 ch.)")

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

        # --- NEW: re-point the pink target line at this step's active
        # region centroid (target_history[i] lines up with path_history[i],
        # both recorded once per step -- see the rollout loop above).
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