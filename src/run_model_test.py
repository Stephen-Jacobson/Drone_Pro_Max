import os
os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"
os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"

import random
import numpy as np
import torch
from torch import nn
import open3d as o3d

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
MAP_CHANNELS = 2         # must match rl_model.py
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "..", "pax_v2_1.pt")
FRAMES_PER_STEP = 10      # adjust this to slow down: higher = slower (1=normal speed)


# ── rebuild model architecture (must match rl_model.py exactly) ──────────────
# Copied verbatim from rl_model.py's CNNActorNet -- this HAS to match the
# trained architecture exactly, or load_state_dict will fail / silently
# load into the wrong shapes.
class CNNActorNet(nn.Module):
    def __init__(self, num_cells, action_dim, map_channels=2, device=None):
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

    return policy_module


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

    env = DroneEnv(clean_grid.copy(), start, end)

    # ── build & load policy ───────────────────────────────────────────────
    action_dim = env.action_spec.shape[-1]
    policy = build_policy(action_dim, device)

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
    spray_history = []   # hit_xyz arrays, one per step, for the lineset animation
    spray_steps = 0      # how many steps the policy actually chose to spray on

    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        for step in range(MAX_STEPS):
            td = policy(td.to(device))
            action = td["action"].cpu().numpy()
            spray_flag = action[..., 3] > 0
            if spray_flag:
                spray_steps += 1

            td = env._step(td)

            path_history.append(env.drone.pos.copy())

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
                print(f"  step {step:4d} | coverage: {mean_comp:.2%} | hit: {hit} | done: {done}")

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
                break

            td = td.select("coverage_map", "flat").to(device)

    print(f"Rollout finished — {len(path_history)} positions recorded.")

    # ── live animated visualisation (Open3D, matching open3d_sim_env.py's
    # run_simulation() pattern -- no pyvista here, this codebase uses Open3D) ──
    terrain_geoms = [sim.voxelgrid_from_voxels(env.values), sim.voxel_edges_lineset(env.values)]

    vis_drone = Drone(0, 10, env.values, start, end)
    drone_mesh = sim.make_drone_mesh(vis_drone)
    spray_lineset = sim.make_spray_lineset(vis_drone.get_position(), np.zeros((0, 3)))

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

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Drone Rollout Playback", width=1024, height=768)
    for geom in terrain_geoms:
        vis.add_geometry(geom)
    vis.add_geometry(drone_mesh)
    vis.add_geometry(spray_lineset)
    vis.add_geometry(path_lineset)
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

        state["step_index"] += 1
        return True

    vis.register_animation_callback(advance)
    print("Press SPACE to pause/resume playback. Rotate/zoom/pan anytime — close window to exit.")
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()