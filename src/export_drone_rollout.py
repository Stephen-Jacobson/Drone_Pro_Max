"""
Exports one or more trained-policy rollouts to JSON so they can be replayed
in the browser (assets/js/drone-replay.js), the same way record_game.py /
tetris_run_N.json feed tetris-replay.js.

This is run_model_test.py's model-loading + rollout loop, with the
Open3D/matplotlib visualization stripped out and JSON writing put in its
place. Run it locally (same env you train/test in) after you have a
checkpoint at CHECKPOINT_PATH:

    python export_drone_rollout.py

It writes assets/data/drone_run_1.json, drone_run_2.json, drone_run_3.json
(one per run of NUM_RUNS) into the folder this script lives in. Copy that
assets/data folder into your portfolio site's assets/data alongside the
tetris_run_N.json files.

--- Data format written per run --------------------------------------------
{
  "shape": [X, Y, Z],                 # voxel grid dimensions
  "start": [x, y, z],
  "voxels": [[x, y, z, material], ...],   # SURFACE voxels only (see note)
  "region_labels": [[...]],           # X x Y ints, 0 = path/no region
  "region_materials": {"1": 4, "2": 5, ...},  # region_id -> 4 (needs water) / 5 (no water)
  "map_shape": [X, Y],
  "steps": [
    {
      "pos": [x, y, z],
      "spray_hits": [[x, y, z], ...],  # empty list if it didn't spray this step
      "target_xy": [x, y] | null,      # where the model's density head is aiming
      "active_region_id": int | null,
      "active_completeness": float | null,  # 0-1, that region's true watered fraction
      "crashed": bool,
      "reward": float,
      "need_diff": [[x, y, value], ...]  # sparse update to the full-res "still needs
                                          # spraying" heatmap -- see viewer notes below
    },
    ...
  ]
}

need_diff is a delta encoding, not a full frame: the viewer keeps one
persistent (X, Y) float array per run, seeded from step 0's need_diff (which
is a full sparse dump of every nonzero cell), and each later step's
need_diff only lists the cells whose value actually changed since last
step. This keeps the JSON small even over hundreds of steps, since most of
the map is untouched most of the time -- only cells near a spray hit, or an
entire region's worth of cells on a hand-off to a new active region, ever
show up in a diff.

Only SURFACE voxels (at least one exposed face) are written for the
terrain, same optimisation open3d_sim_env.voxel_edges_lineset already makes
for the same reason: a fully buried voxel is never visible no matter what,
so it's a wasted row in the file.
"""
import os
os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"
os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"

import json
import random

import numpy as np
import torch
from torch import nn

from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torchrl.modules import ProbabilisticActor, TanhNormal
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.data import Bounded

import open3d_sim_env as sim
from drone_env import DroneEnv
import pick_block

MAX_STEPS = 400
num_cells = 256  # must match rl_model.py
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "..", "trained_models/pax_v2_5.pt")
TEST_REGION_DONE_THRESHOLD = 0.90
NUM_RUNS = 3  # how many drone_run_N.json files to produce
OUT_DIR = os.path.join(os.path.dirname(__file__), "assets", "data")


# ── model architecture: copied verbatim from run_model_test.py ─────────────
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
            nn.LazyLinear(num_cells, device=device), nn.Tanh(),
            nn.LazyLinear(num_cells, device=device), nn.Tanh(),
            nn.LazyLinear(num_cells, device=device), nn.Tanh(),
            nn.LazyLinear(num_cells, device=device), nn.Tanh(),
            nn.LazyLinear(num_cells, device=device), nn.Tanh(),
            nn.LazyLinear(2 * action_dim, device=device),
        )

        self.extractor = NormalParamExtractor()
        self.last_target_xy = None
        self._last_picked_blocks = None

    def _target_xy_from_block(self, block_rc):
        if block_rc is None:
            return None
        row, col = block_rc
        return np.array([float(row) + 0.5, float(col) + 0.5], dtype=np.float32)

    def _block_loc_batch(self, density_full_batch, drone_xy_batch, t_batch):
        density_np = density_full_batch.detach().cpu().numpy()
        pos_np = drone_xy_batch.detach().cpu().numpy()
        t_np = t_batch.detach().cpu().numpy().reshape(-1)

        rows = []
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
        density_channel = coverage_map[..., 0, :, :]
        batch_shape = density_channel.shape[:-2]
        density_flat = density_channel.reshape(-1, *density_channel.shape[-2:])

        density_full_flat = density_map_full.reshape(-1, *density_map_full.shape[-2:])
        drone_xy_flat = drone_xy.reshape(-1, drone_xy.shape[-1])

        chosen_t = self.density_direction(density_flat.unsqueeze(1))
        block_loc = self._block_loc_batch(density_full_flat, drone_xy_flat, chosen_t)
        block_loc = block_loc.reshape(*batch_shape, 3)

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
        distribution_kwargs={"low": action_spec.space.low, "high": action_spec.space.high},
        return_log_prob=True,
    )
    return policy_module, actor_net


def build_world():
    seed = random.randint(0, 10000)
    voxels = sim.generate_voxels(seed=seed)
    drone = sim.spawn_drone(voxels)
    start = drone.get_position()
    end = start.copy()
    clean = voxels.copy()
    return start, end, clean


# ── voxel/material export helpers ───────────────────────────────────────────
def surface_voxels(values):
    """(N, 4) int array of [x, y, z, material] for every voxel with at
    least one exposed face -- same 'skip fully-buried cells' optimisation
    as open3d_sim_env.voxel_edges_lineset, just kept as raw rows instead of
    a LineSet."""
    occ = values != 0
    pad = np.pad(occ, 1, mode="constant", constant_values=False)
    fully_buried = (
        pad[2:, 1:-1, 1:-1] & pad[:-2, 1:-1, 1:-1] &
        pad[1:-1, 2:, 1:-1] & pad[1:-1, :-2, 1:-1] &
        pad[1:-1, 1:-1, 2:] & pad[1:-1, 1:-1, :-2]
    )
    surface = occ & ~fully_buried
    idx = np.argwhere(surface)
    mats = values[idx[:, 0], idx[:, 1], idx[:, 2]].astype(int)
    return np.column_stack([idx, mats])


def sparse_diff(prev, cur, decimals=3):
    """List of [x, y, round(value, decimals)] for cells that changed
    (or all nonzero cells, if prev is None -- used for the very first
    frame, which has to be a full dump since there's nothing to diff
    against)."""
    cur_r = np.round(cur, decimals)
    if prev is None:
        idx = np.argwhere(cur_r != 0)
    else:
        idx = np.argwhere(np.round(prev, decimals) != cur_r)
    return [[int(x), int(y), float(cur_r[x, y])] for x, y in idx]


def run_one_rollout(policy, actor_net, device):
    start, end, clean_grid = build_world()
    env = DroneEnv(clean_grid.copy(), start, end, region_done_threshold=TEST_REGION_DONE_THRESHOLD)

    dummy = env.reset().to(device)
    with torch.no_grad():
        policy(dummy)  # warm up lazy layers (state_dict already loaded by caller before this)

    td = env.reset().to(device)

    steps = []
    prev_need_map = None

    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        policy(td.to(device))  # seed actor_net.last_target_xy for step 0

        for step in range(MAX_STEPS):
            td = policy(td.to(device))
            action = td["action"].cpu().numpy()
            spray_flag = bool(action[..., 3] > 0)

            td = env._step(td)
            reward = float(td["reward"].item())

            hit_xyz = np.zeros((0, 3))
            if spray_flag and not env._hit_something():
                hit_xyz = env.drone.spray()  # re-fire purely for the visual, matches run_model_test.py

            need_map = env.full_res_active_need_map()
            need_diff = sparse_diff(prev_need_map, need_map)
            prev_need_map = need_map

            comp = env.spray_tracker.region_completeness()
            active_completeness = (
                comp.get(env.active_region_id) if env.active_region_id is not None else None
            )

            target_xy = actor_net.last_target_xy
            target_xy_list = (
                [round(float(target_xy[0]), 2), round(float(target_xy[1]), 2)]
                if target_xy is not None else None
            )

            steps.append({
                "pos": [round(float(v), 2) for v in env.drone.pos],
                "spray_hits": hit_xyz.astype(int).tolist(),
                "target_xy": target_xy_list,
                "active_region_id": int(env.active_region_id) if env.active_region_id is not None else None,
                "active_completeness": round(float(active_completeness), 3) if active_completeness is not None else None,
                "crashed": bool(env._hit_something()),
                "reward": round(reward, 4),
                "need_diff": need_diff,
            })

            done = bool(td["done"].item())
            if done:
                break

            td = td.select("coverage_map", "flat", "density_map_full", "drone_xy").to(device)

    region_materials = {str(k): int(v) for k, v in env.region_materials.items()}

    return {
        "shape": list(env.values.shape),
        "start": [round(float(v), 2) for v in start],
        "voxels": surface_voxels(env.values).tolist(),
        "region_labels": env.labels.tolist(),
        "region_materials": region_materials,
        "map_shape": list(env.spray_tracker.shape),
        "steps": steps,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    dummy_start, dummy_end, dummy_grid = build_world()
    dummy_env = DroneEnv(dummy_grid.copy(), dummy_start, dummy_end)
    action_dim = dummy_env.action_spec.shape[-1]
    policy, actor_net = build_policy(action_dim, device)

    warm_td = dummy_env.reset().to(device)
    with torch.no_grad():
        policy(warm_td)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    policy.load_state_dict(checkpoint["policy"])
    policy.eval()
    print("Model loaded.")

    os.makedirs(OUT_DIR, exist_ok=True)

    for run_idx in range(1, NUM_RUNS + 1):
        print(f"Recording run {run_idx}/{NUM_RUNS}...")
        data = run_one_rollout(policy, actor_net, device)
        out_path = os.path.join(OUT_DIR, f"drone_run_{run_idx}.json")
        with open(out_path, "w") as f:
            json.dump(data, f)
        size_kb = os.path.getsize(out_path) / 1024
        print(f"  wrote {out_path} ({len(data['steps'])} steps, {size_kb:.0f} KB)")


if __name__ == "__main__":
    main()