import numpy as np
import torch
from tensordict import TensorDict
from torchrl.data import Bounded, Composite, Unbounded
from torchrl.envs import EnvBase

import lidar
from drone import Drone

GOAL_RADIUS = 5.0

class DroneEnv(EnvBase):
    def __init__(self, values, start, end, n_rays=100, sight_range=10):
        super().__init__()
        self.values = values
        self.og_values = values.copy()
        self.current_step = 0
        self.start = np.array(start, dtype=np.float32)
        self.end = np.array(end, dtype=np.float32)
        self.n_rays = n_rays
        self.sight_range = sight_range
        self.drone = Drone(0, self.sight_range, self.values, self.start, self.end)
        self.path_history = []  # in order to visually show first run of batch
        self.record = False

        self._last_scan = np.ones(self.n_rays)
        self._last_hit = False  # authoritative collision flag, set in _step()

        obs_size = n_rays + 4  # number of lidar rays + direction(3) + is_done

        # do these calls so TorchRL knows what to expect and what shape and size shit will be when it works with this env
        self.observation_spec = Composite(observation=Unbounded(shape=(obs_size,)))
        self.action_spec = Bounded(low=-1, high=1, shape=(3,))

        self.reward_spec = Unbounded(shape=(1,))

    def _step(self, tensordict):
        self.current_step += 1
        action = tensordict["action"].cpu().numpy()
        prev_pos = self.drone.pos.copy()
        hit = self.drone.move(action, amount=1)
        self._last_hit = hit
        if self.record and not hit:   # don't record the position if the drone just crashed
            self.path_history.append(self.drone.pos.copy())

        obs = self._get_obs()
        reward = self._get_reward(prev_pos, current_step=self.current_step)
        done = self._is_done()

        return TensorDict(
            {
                "observation": torch.tensor(obs, device=self.device),
                "reward": torch.tensor(
                    [reward], dtype=torch.float32, device=self.device
                ),
                "done": torch.tensor([done], device=self.device),
            },
            batch_size=[],
        )

    def _reset(self, tensordict=None, **kwargs):
        self.values = self.og_values.copy()
        self.current_step = 0
        self.drone = Drone(0, self.sight_range, self.values, self.start, self.end)
        self._last_hit = False
        obs = self._get_obs()
        return TensorDict(
            {"observation": torch.tensor(obs, device=self.device)}, batch_size=[]
        )

    def _get_obs(self):
        self._last_scan = lidar.get_lidar_surroundings(  # store for get reward
            self.values, self.drone.pos, self.sight_range
        )
        direction, distance = lidar.get_goal_vector(self.drone.pos, self.end, self.sight_range)
        return np.concatenate([self._last_scan, direction, [distance]]).astype(np.float32)

    def _get_reward(self, prev_pos, current_step=0, max_steps=600):
        curr_dist = np.linalg.norm(self.drone.pos - self.end)
        prev_dist = np.linalg.norm(prev_pos - self.end)
    
        if self._hit_something():
            return -1.0
        if curr_dist < GOAL_RADIUS:
            # time_bonus = 10.0 * (1 - current_step / max_steps)  # 10.0 early, decays to 0.0 late
            # return float(np.clip(time_bonus, 5.0, 10.0))         # always at least 5.0 for reaching goal
            return 1
        return -0.01
    
        # max_step = float(np.sqrt(3))
        # progress = (prev_dist - curr_dist) / max_step
    
        # if progress <= 0:
        #     reward = 1.5 * float(np.clip(progress, -1.0, 1.0))
        # else:
        #     reward = 0.5 * float(np.clip(progress, -1.0, 1.0))
    
        # reward -= 0.01
        # hazard-only shaping: penalize being dangerously close, never reward open air
        # SAFE_CLEARANCE = 0.3
        # clearance = float(np.min(self._last_scan))
        # if clearance < SAFE_CLEARANCE:
        #     reward -= 0.02 * (SAFE_CLEARANCE - clearance)
    
        # return float(np.clip(reward, -1.0, 1.0))

    def _hit_something(self):
        # authoritative — set directly from Drone.move()'s return value in _step().
        # (previously this re-derived crash status from self.values at self.drone.pos,
        # which is wrong in two ways: an out-of-bounds attempt never updates pos, so
        # this always read "safe"; and an in-bounds collision used to still move into
        # the solid voxel and overwrite it with the drone marker before this check ran,
        # which hid real collisions from the reward/done logic entirely.)
        return self._last_hit

    def _is_done(self):
        return self._hit_something() or np.linalg.norm(self.drone.pos - self.end) < GOAL_RADIUS

    def _set_seed(self, seed):
        np.random.seed(seed)