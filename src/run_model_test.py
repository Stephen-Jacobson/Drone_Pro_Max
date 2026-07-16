import os

os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"
os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"

import time
import torch
from torch import nn
import numpy as np

from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torchrl.modules import ProbabilisticActor, TanhNormal
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.data import Bounded

import sim_env as sim
from drone_env import DroneEnv
from drone import Drone

FOLLOW = True
MAX_STEPS = 600
num_cells = 256

# ── rebuild model architecture (must match rl_model.py exactly) ──────────────


def build_policy(obs_size, action_shape, device):
    actor_net = nn.Sequential(
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(2 * action_shape, device=device),
        NormalParamExtractor(),
    )

    policy_module = TensorDictModule(
        actor_net, in_keys=["observation"], out_keys=["loc", "scale"]
    )

    action_spec = Bounded(low=-1, high=1, shape=(action_shape,))

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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    # ── generate world ────────────────────────────────────────────────────────
    sim.generate_terrain()
    sim.generate_trees()
    start, end = sim.generate_start_and_end()
    sim.replace_within_clearance(start[0], start[1], start[2], 2, 0)
    sim.replace_within_clearance(end[0], end[1], end[2], 2, 0)

    print(f"Start: {start}, End: {end}")

    env = DroneEnv(sim.values, start, end)

    # ── build & load policy ───────────────────────────────────────────────────
    obs_size = env.observation_spec["observation"].shape[0]
    policy = build_policy(obs_size, env.action_spec.shape[-1], device)

    # warm up lazy layers so state_dict loads cleanly
    dummy = env.reset().to(device)
    with torch.no_grad():
        policy(dummy)

    model_path = os.path.join(os.path.dirname(__file__), "..", "pax_v1.1.pt")
    checkpoint = torch.load(model_path, map_location=device)
    policy.load_state_dict(checkpoint["policy"])
    policy.eval()
    print("Model loaded.")

    # ── run rollout, collect path ─────────────────────────────────────────────
    td = env.reset().to(device)
    path_history = [env.drone.pos.copy()]

    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        for step in range(MAX_STEPS):
            td = policy(td.to(device))
            td = env._step(td)

            path_history.append(env.drone.pos.copy())

            done = td["done"].item()
            hit = env._hit_something()
            dist = float(np.linalg.norm(env.drone.pos - np.array(end)))

            if step % 50 == 0:
                print(
                    f"  step {step:4d} | dist to goal: {dist:.2f} | hit: {hit} | done: {done}"
                )

            if done:
                if hit:
                    print(f"Crashed at step {step}.")
                else:
                    print(f"Reached goal at step {step}!")
                break

            td = td.select("observation").to(device)

    print(f"Rollout finished — {len(path_history)} positions recorded.")

    # ── live animated visualisation ───────────────────────────────────────────
    sim.values[int(env.drone.pos[0]), int(env.drone.pos[1]), int(env.drone.pos[2])] = 0
    sim.values[int(start[0]), int(start[1]), int(start[2])] = 5

    vis_drone = Drone(0, 10, sim.values, start, end)
    pl, drone_mesh = sim.show_grid(vis_drone)
    assert drone_mesh is not None

    step_index = [0]
    prev_pos = np.array(start, dtype=np.float32)

    pl.show(interactive_update=True)
    while step_index[0] < len(path_history):
        new_pos = path_history[step_index[0]].astype(np.float32)
        delta = new_pos - prev_pos
        drone_mesh.translate(delta, inplace=True)
        prev_pos = new_pos
        step_index[0] += 1

        if FOLLOW:
            lookat = prev_pos
            pl.camera_position = [
                lookat + [50, -30, 40],
                lookat,
                [0, 0, 1],
            ]

        pl.update()
        time.sleep(0.03)

    pl.close()


if __name__ == "__main__":
    main()
