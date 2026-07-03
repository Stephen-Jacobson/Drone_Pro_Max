# TODO: create model in here to be used by drone
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
import sim_env as sim
# keep imports and constants at top as normal
import torch
from torch import nn
from torch import multiprocessing
# ... all imports ...
 
MAX_STEPS = 600
VISUALISE = False

if __name__ == "__main__":
    is_fork = multiprocessing.get_start_method() == "fork"
    device = (
        torch.device(0)        # GPU
        if torch.cuda.is_available() and not is_fork
        else torch.device("cpu")
    )                               # decides if torch will go on gpu or cpu, gpu better
    num_cells = 256                  # cells per hidden layer
    lr = 3e-4
    max_grad_norm = 1.0             # prevents too violent actions early in training
    
    print(f"training on {device}")
    
    steps_per_batch = 8000          # how many moves will make before model learns from it, so model isnt updating until steps_per_batch moves have been done, 1 step = 1 move
    total_steps = 1_000_000            # how many total steps until done training
    #therefore if 1000 steps in batch and 50000 steps total, will learn 50000/1000 = 50 times
    
    # PPO Parameters (Proximal Policy Optimization)
        # At each steps_per_batch we will run to optimise model, this is done by taking a sub batch size, eg 64,
        # then taking 64 random steps(each step stored as action and reward as well current and next lidar scans, 
        # and if done or hit object) from the 1000 which just occured, and will do that until steps_per_batch=1000 random values have been taken. 
        # Will do that num_epoch number of times, eg 10, therefore will have (1000/64)*10 = ~156 gradient updates. At each epoch gradient/model 
        # updates ~15 times, so model updates every 64 taken from 1000, ie ~156
    sub_batch_size = 512
    num_epochs = 10
    clip_epsilon = 0.2              # stops policy from updating too much in one steps, 0.2 will stop from updating when change is more than 20%
    gamma = 0.95                    # between 0-1, closer to 1, worries more about future rewards, closer to 0, worries more about immediate rewards
    lmbda = 0.95                    # used to compute advantage of move, ie was it better or worse than the expaected reward which our critic calculates
    entropy_eps = 0.01              # rewards exploration at beginning of training so model doesnt commit to badd moves, less important later in training
    frames_collected = 0 
    
    def build_world():
        """Generate a fresh terrain and return start, end, and a clean grid snapshot."""
        sim.reseed()
        progress = frames_collected / total_steps
        rx = math.floor(30 + progress * 500)
        ry = math.floor(30 + progress * 150)
        sim.set_dim(rx, ry)
        sim.values[:] = 0
        sim.ground_level[:] = 0
        sim.generate_terrain()
        sim.generate_trees()
        start, end = sim.generate_start_and_end()
        sim.replace_within_clearance(start[0], start[1], start[2], 2, 0)
        sim.replace_within_clearance(end[0], end[1], end[2], 2, 0)
        clean = sim.values.copy()  # snapshot BEFORE any worker touches it
        return start, end, clean

    start, end, clean_grid = build_world()

    def make_env():
        v = clean_grid.copy()  # each env gets its own copy of the guaranteed-clean grid
        return DroneEnv(v, start, end)

    # this will run 20 environments: based on the number of cores that works 
    # the more the better 
    base_env = ParallelEnv(20, make_env)
    env = TransformedEnv(base_env, StepCounter(max_steps=MAX_STEPS))
    env = env.to(device)
    
    actor_net = nn.Sequential(
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(2 * env.action_spec.shape[-1], device=device),
        NormalParamExtractor(),
    )
    
    policy_module = TensorDictModule(
        actor_net, in_keys=["observation"], out_keys=["loc", "scale"]
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
    
    value_net = nn.Sequential(
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(1, device=device),
    )
    
    value_module = ValueOperator(
        module=value_net,
        in_keys=["observation"],
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
    pbar = tqdm(total=total_steps)
    eval_str = ""
    
    REGEN_EVERY = 3  # new world every N batches (~104 batches total)

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
            
            start, end, clean_grid = build_world()
            # collector.shutdown() already closed base_env — don't call base_env.close()
            collector.shutdown()
            base_env = ParallelEnv(24, make_env)
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
            vis_env = DroneEnv(clean_grid.copy(), start, end)
            vis_env.record = True
            td_vis = vis_env.reset()
            with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
                for _ in range(MAX_STEPS):
                    td_vis = td_vis.to(device)
                    td_vis = policy_module(td_vis)
                    td_vis = vis_env._step(td_vis)
                    if td_vis["done"].item():
                        break
                    td_vis = td_vis.select("observation")
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
        logs["lr"].append(optim.param_groups[0]["lr"])
        lr_str = f"lr policy: {logs['lr'][-1]: 4.4f}"
        if i % 10 == 0:
            with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
                eval_rollout = env.rollout(1000, policy_module)
                logs["eval reward"].append(eval_rollout["next", "reward"].mean().item())
                logs["eval reward (sum)"].append(eval_rollout["next", "reward"].sum().item())
                logs["eval step_count"].append(eval_rollout["step_count"].max().item())
                eval_str = (
                    f"eval cumulative reward: {logs['eval reward (sum)'][-1]: 4.4f} "
                    f"(init: {logs['eval reward (sum)'][0]: 4.4f}), "
                    f"eval step-count: {logs['eval step_count'][-1]}"
                )
                del eval_rollout
        pbar.set_description(", ".join([eval_str, cum_reward_str, stepcount_str, lr_str]))
        scheduler.step()

        frames_collected += tensordict_data.numel()
        i += 1
    
    torch.save({
        "policy": policy_module.state_dict(),
        "value":  value_module.state_dict(),
    }, "pax_v1.1.pt")
    print("model saved")

    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.plot(logs["reward"])
    plt.title("training rewards (average)")
    plt.subplot(2, 2, 2)
    plt.plot(logs["step_count"])
    plt.title("Max step count (training)")
    plt.subplot(2, 2, 3)
    plt.plot(logs["eval reward (sum)"])
    plt.title("Return (test)")
    plt.subplot(2, 2, 4)
    plt.plot(logs["eval step_count"])
    plt.title("Max step count (test)")
    plt.show()
