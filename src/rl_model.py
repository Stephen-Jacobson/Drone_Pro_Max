# TODO: create model in here to be used by drone
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from torch import nn
from torch import multiprocessing

from collections import defaultdict
import threading
from torchrl.envs import ParallelEnv

from torchrl.envs import TransformedEnv
from torchrl.envs.transforms import StepCounter
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.collectors import Collector
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.data.replay_buffers import ReplayBuffer

from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type

from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor

from tqdm import tqdm

import matplotlib.pyplot as plt

import random
import math

from drone_env import DroneEnv
import open3d_sim_env as sim
 
MAX_STEPS = 400
VISUALISE = False
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "..", "pax_v2_3.pt")

if __name__ == "__main__":
    is_fork = multiprocessing.get_start_method() == "fork"
    device = (
        torch.device(0)        # GPU
        if torch.cuda.is_available() and not is_fork
        else torch.device("cpu")
    )                               # decides if torch will go on gpu or cpu, gpu better
    num_cells = 512                  # cells per hidden layer
    lr = 3e-4
    max_grad_norm = 1.0             # prevents too violent actions early in training
    
    print(f"training on {device}")
    
    steps_per_batch = 1600          # how many moves will make before model learns from it, so model isnt updating until steps_per_batch moves have been done, 1 step = 1 move
    total_steps = 1_000_000            # how many total steps until done training
    #therefore if 1000 steps in batch and 50000 steps total, will learn 50000/1000 = 50 times

    # --- REGION_DONE_THRESHOLD staircase curriculum, spread evenly across total_steps ---
    # Instead of a continuous ramp, training is chopped into this many equal
    # stages; REGION_DONE_THRESHOLD jumps by one fixed increment at each
    # stage boundary (flat within a stage). All three are decided/changeable:
    REGION_DONE_THRESHOLD_START = 0.01      # threshold for stage 0 (start of training)
    REGION_DONE_THRESHOLD_END = 0.50        # threshold for the final stage
    REGION_DONE_THRESHOLD_INTERVALS = 8     # number of stepped stages spanning total_steps
    
    # PPO Parameters (Proximal Policy Optimization)
        # At each steps_per_batch we will run to optimise model, this is done by taking a sub batch size, eg 64,
        # then taking 64 random steps(each step stored as action and reward as well current and next lidar scans, 
        # and if done or hit object) from the 1000 which just occured, and will do that until steps_per_batch=1000 random values have been taken. 
        # Will do that num_epoch number of times, eg 10, therefore will have (1000/64)*10 = ~156 gradient updates. At each epoch gradient/model 
        # updates ~15 times, so model updates every 64 taken from 1000, ie ~156
    sub_batch_size = 128
    num_epochs = 10
    clip_epsilon = 0.2              # stops policy from updating too much in one steps, 0.2 will stop from updating when change is more than 20%
    gamma = 0.99                    # between 0-1, closer to 1, worries more about future rewards, closer to 0, worries more about immediate rewards
    lmbda = 0.95                    # used to compute advantage of move, ie was it better or worse than the expaected reward which our critic calculates
    entropy_eps = 0.01              # rewards exploration at beginning of training so model doesnt commit to badd moves, less important later in training
    frames_collected = 0 
    
    def build_world(progress=0.0):
        """Generate a fresh terrain using open3d_sim_env and return a drone
        start position, a dummy goal, a clean grid snapshot, the prob4 for
        this generation (probability a crop region gets painted "needs
        water" vs "doesn't"), and the current stepped REGION_DONE_THRESHOLD.

        `progress` is training progress in [0, 1] (frames_collected / total_steps).
        It drives a curriculum on water-region density via sim.water_region_prob():
        early worlds have few water regions, ramping up to a harder steady-state
        fraction as training progresses. prob4 itself is applied inside DroneEnv
        (both __init__ and _reset call fill_regions_top_materials with it) --
        NOT here, since DroneEnv re-labels/re-fills the grid from og_values on
        every single episode reset regardless of what materials this function's
        voxels already have baked in. Baking regions in here would just get
        silently overwritten.

        `progress` also drives a SEPARATE, stepped (staircase) curriculum on
        REGION_DONE_THRESHOLD via DroneEnv.region_done_threshold_for_progress():
        total_steps is chopped into REGION_DONE_THRESHOLD_INTERVALS equal
        stages, and the threshold jumps by a fixed increment at each stage
        boundary instead of ramping continuously. Unlike prob4, this value
        IS passed straight into DroneEnv's constructor (region_done_threshold=),
        since it's just an instance attribute read each _get_reward() call,
        not something baked into the voxel grid that _reset() would overwrite.

        NOTE: no terrain-size curriculum here -- open3d_sim_env.py hardcodes
        max_x/max_y=300 as module globals, unlike the old sim_env's set_dim().
        If you want progressive difficulty back, it would need monkey-patching
        sim.max_x/sim.max_y before generate_voxels(), which is fragile.

        NOTE: `end` is just set equal to `start` -- there's no real navigation
        goal anymore since the task is coverage/watering, not reaching a point.
        DroneEnv's goal-direction/distance features will always read ~zero
        distance as a result. Fine to leave as dead weight for now; worth
        dropping from `flat` in DroneEnv._get_obs() later if it doesn't help.
        """
        prob4 = sim.water_region_prob(progress)
        region_done_threshold = DroneEnv.region_done_threshold_for_progress(
            progress,
            start=REGION_DONE_THRESHOLD_START,
            end=REGION_DONE_THRESHOLD_END,
            n_intervals=REGION_DONE_THRESHOLD_INTERVALS,
        )
        voxels = sim.generate_voxels()      # terrain + crop-plot paths, one call (no regions yet)
        drone = sim.spawn_drone(voxels)     # safe, non-colliding spawn point
        start = drone.get_position()
        end = start.copy()

        clean = voxels.copy()  # snapshot BEFORE DroneEnv labels/paints it
        return start, end, clean, prob4, region_done_threshold

    start, end, clean_grid, prob4, region_done_threshold = build_world(progress=0.0)

    def make_env():
        v = clean_grid.copy()  # each env gets its own copy of the guaranteed-clean grid
        return DroneEnv(v, start, end, prob4=prob4, region_done_threshold=region_done_threshold)

    base_env = ParallelEnv(16, make_env)
    env = TransformedEnv(base_env, StepCounter(max_steps=MAX_STEPS))
    env = env.to(device)
    
    # actor_net = nn.Sequential(
    #     nn.LazyLinear(num_cells, device=device),
    #     nn.Tanh(),
    #     nn.LazyLinear(num_cells, device=device),
    #     nn.Tanh(),
    #     nn.LazyLinear(num_cells, device=device),
    #     nn.Tanh(),
    #     nn.LazyLinear(2 * env.action_spec.shape[-1], device=device),
    #     NormalParamExtractor(),
    # )

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

    actor_net = CNNActorNet(num_cells=num_cells, action_dim=env.action_spec.shape[-1], device=device)

    
    policy_module = TensorDictModule(
        actor_net, in_keys=["coverage_map", "flat"], out_keys=["loc", "scale"]
    ) 
    
    policy_module = ProbabilisticActor(
        module=policy_module,
        spec=env.action_spec,
        in_keys=["loc", "scale"],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": env.action_spec.space.low,
            "high": env.action_spec.space.high,
        },
        return_log_prob=True,
        # we'll need the log-prob for the numerator of the importance weights
    )
    
    # value_net = nn.Sequential(
    #     nn.LazyLinear(num_cells, device=device),
    #     nn.Tanh(),
    #     nn.LazyLinear(num_cells, device=device),
    #     nn.Tanh(),
    #     nn.LazyLinear(num_cells, device=device),
    #     nn.Tanh(),
    #     nn.LazyLinear(1, device=device),
    # )

    class CNNValueNet(nn.Module):
        def __init__(self, num_cells, map_channels=3, device=None):
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
                nn.LazyLinear(1, device=device),
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
            return out
        
    value_net = CNNValueNet(num_cells=num_cells, device=device)
    
    value_module = ValueOperator(
        module=value_net,
        in_keys=["coverage_map", "flat"],
    )
    
    print("Running policy:", policy_module(env.reset()))
    print("Running value:", value_module(env.reset()))
    
    collector = Collector(
        env,
        policy_module,
        frames_per_batch=steps_per_batch,
        total_frames=total_steps,
        split_trajs=False,
        device=device,
    )
    
    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=steps_per_batch),
        sampler=SamplerWithoutReplacement(),
    )
    
    advantage_module = GAE(
        gamma=gamma, lmbda=lmbda, value_network=value_module, average_gae=True, device=device,
    )
    
    loss_module = ClipPPOLoss(
        actor_network=policy_module,
        critic_network=value_module,
        clip_epsilon=clip_epsilon,
        entropy_bonus=True,
        entropy_coeff=entropy_eps,
        # these keys match by default but we set this for completeness
        critic_coeff=1.0,
        loss_critic_type="smooth_l1",
    )
    
    optim = torch.optim.Adam(loss_module.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, total_steps // steps_per_batch, 0.0
    )
    
    logs = defaultdict(list)
    pbar = tqdm(total=total_steps, position=0, leave=True)
    # Second stacked bar used purely to display a second line of stats text
    # (no progress of its own) -- keeps the training/eval numbers from being
    # crammed onto one line and truncated on narrower terminals.
    stats_bar = tqdm(total=0, position=1, bar_format="{desc}", leave=True)
    eval_str = ""
    
    REGEN_EVERY = 5  # new world every N batches (~104 batches total)

    # `for i, x in enumerate(collector)` would only call iter(collector) ONCE, up front.
    # Reassigning the `collector` variable inside the loop body (on regen) does NOT change
    # what that already-bound iterator pulls from — it keeps calling next() on the OLD,
    # now-shutdown() collector, whose internal `_final_rollout` has been torn down. That's
    # the source of `AttributeError: 'Collector' object has no attribute '_final_rollout'`.
    # Driving the iterator manually lets us actually swap it out after a regen.
    i = 0
    collector_iter = iter(collector)

    while frames_collected < total_steps:
        try:
            tensordict_data = next(collector_iter)
        except StopIteration:
            break

        # regenerate terrain periodically so the model trains on varied worlds
        if i > 0 and i % REGEN_EVERY == 0:

            start, end, clean_grid, prob4, region_done_threshold = build_world(progress=frames_collected / total_steps)
            # collector.shutdown() already closed base_env — don't call base_env.close()
            collector.shutdown()
            del collector, base_env, env
            import gc
            gc.collect()
            torch.cuda.empty_cache()

            base_env = ParallelEnv(16, make_env)
            env = TransformedEnv(base_env, StepCounter(max_steps=MAX_STEPS))
            env = env.to(device)
            collector = Collector(
                env,
                policy_module,
                frames_per_batch=steps_per_batch,
                total_frames=total_steps - frames_collected,
                split_trajs=False,
                device=device,
            )
            collector_iter = iter(collector)

        # visualise one rollout every 10th batch via a standalone env
        # (base_env is a ParallelEnv proxy — path_history/record don't reach workers)
        if i % 50 == 0 and VISUALISE:
            vis_env = DroneEnv(clean_grid.copy(), start, end, prob4=prob4, region_done_threshold=region_done_threshold)
            vis_env.record = True
            td_vis = vis_env.reset()
            with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
                for _ in range(MAX_STEPS):
                    td_vis = td_vis.to(device)
                    td_vis = policy_module(td_vis)
                    td_vis = vis_env._step(td_vis)
                    if td_vis["done"].item():
                        break
                    td_vis = td_vis.select("coverage_map", "flat")
            path_copy = vis_env.path_history.copy()
            threading.Thread(target=sim.show_path, args=(path_copy, clean_grid.copy()), daemon=True).start()
    
        for _ in range(num_epochs):
            advantage_module(tensordict_data)
            data_view = tensordict_data.reshape(-1)
            replay_buffer.extend(data_view.cpu())
            for _ in range(steps_per_batch // sub_batch_size):
                subdata = replay_buffer.sample(sub_batch_size)
                loss_vals = loss_module(subdata.to(device))
                loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )
                loss_value.backward()
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
                optim.step()
                optim.zero_grad()
    
        logs["reward"].append(tensordict_data["next", "reward"].mean().item())
        pbar.update(tensordict_data.numel())
        cum_reward_str = (
            f"average reward={logs['reward'][-1]: 4.4f} (init={logs['reward'][0]: 4.4f})"
        )
        logs["step_count"].append(tensordict_data["step_count"].max().item())
        stepcount_str = f"step count (max): {logs['step_count'][-1]}"

        # --- NEW: track the policy's action std (scale) during training.
        # If this stays flat/large instead of shrinking over training, it
        # means the policy is still leaning on exploration noise to survive
        # rather than converging its mean (loc) toward genuinely good
        # actions -- exactly the gap you'd see between a decent training
        # step-count and a much worse deterministic eval step-count.
        logs["policy_std"].append(tensordict_data["scale"].mean().item())
        std_str = f"policy std: {logs['policy_std'][-1]: 4.4f}"

        logs["lr"].append(optim.param_groups[0]["lr"])
        lr_str = f"lr policy: {logs['lr'][-1]: 4.4f}"
        region_done_str = f"region-done thresh: {region_done_threshold:.3f}"
        if i % 10 == 0:
            with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
                eval_rollout = env.rollout(1000, policy_module)
                logs["eval reward"].append(eval_rollout["next", "reward"].mean().item())
                logs["eval reward (sum)"].append(eval_rollout["next", "reward"].sum().item())
                logs["eval step_count"].append(eval_rollout["step_count"].max().item())
                logs["eval_policy_std"].append(eval_rollout["scale"].mean().item())
                eval_str = (
                    f"eval cumulative reward: {logs['eval reward (sum)'][-1]: 4.4f} "
                    f"(init: {logs['eval reward (sum)'][0]: 4.4f}), "
                    f"eval step-count: {logs['eval step_count'][-1]}, "
                    f"eval std: {logs['eval_policy_std'][-1]: 4.4f}"
                )
                del eval_rollout
        pbar.set_description(", ".join([cum_reward_str, stepcount_str, std_str, lr_str, region_done_str]))
        stats_bar.set_description(eval_str)
        scheduler.step()

        frames_collected += tensordict_data.numel()
        i += 1
    
    torch.save({
        "policy": policy_module.state_dict(),
        "value":  value_module.state_dict(),
    }, CHECKPOINT_PATH)
    print(f"model saved to {CHECKPOINT_PATH}")

    plt.figure(figsize=(10, 15))
    plt.subplot(3, 2, 1)
    plt.plot(logs["reward"])
    plt.title("training rewards (average)")
    plt.subplot(3, 2, 2)
    plt.plot(logs["step_count"])
    plt.title("Max step count (training)")
    plt.subplot(3, 2, 3)
    plt.plot(logs["eval reward (sum)"])
    plt.title("Return (test)")
    plt.subplot(3, 2, 4)
    plt.plot(logs["eval step_count"])
    plt.title("Max step count (test)")
    plt.subplot(3, 2, 5)
    plt.plot(logs["policy_std"])
    plt.title("Policy action std (training, every iter)")
    plt.subplot(3, 2, 6)
    plt.plot(logs["eval_policy_std"])
    plt.title("Policy action std (eval, every 10th iter)")
    plt.show()