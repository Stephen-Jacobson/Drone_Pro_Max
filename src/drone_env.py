import numpy as np

from torchrl.envs import EnvBase
from torchrl.data import BoundedTensorSpec, CompositeSpec, UnboundedContinuousTensorSpec
import torch

from tensordict import TensorDict

from drone import Drone
import lidar

class DroneEnv(EnvBase):
    def __init__(self, values, start, end, n_rays=300, sight_range=10):
        self.values = values
        self.start = np.array(start, dtype=np.float32)
        self.end   = np.array(end,   dtype=np.float32)
        self.n_rays = n_rays
        self.sight_range = sight_range
        self.drone = Drone(0, self.sight_range, self.values, self.start, self.end)

        obs_size = n_rays + 4           # number of lidar rays + direction(3) + is_done
        
        # do these calls so TorchRL knows what to expect and what shape and size shit will be when it works with this env
        self.observation_space = CompositeSpec(
            UnboundedContinuousTensorSpec(shape=(obs_size,))
        )
        self.action_spec = BoundedTensorSpec(
            low=-1, high=1, shape=(3,)
        )

        self.reward_spec = UnboundedContinuousTensorSpec(shape=(1,))
        
    def _step(self, tensordict):
        action = tensordict["action"].numpy()
        prev_pos = self.drone.pos.copy()

        self.drone.move(action, amount=1)
        obs = self._get_obs()
        reward = self._get_reward(prev_pos)
        done = self._is_done()

        return TensorDict({
            "observation": torch.tensor(obs),
            "reward": torch.tensor([reward], dtype=torch.float32),
            "done": torch.tensor([done]),
        }, batch_size=[])

    def _reset(self, tensordict=None, **kwargs):
        self.drone = Drone(0, self.sight_range, self.values, self.start, self.end)
        obs = self._get_obs()
        return TensorDict({"observation": torch.tensor(obs)}, batch_size=[])

    def _get_obs(self):
        scan = lidar.get_lidar_surroundings(self.values, self.drone.pos, self.sight_range)
        direction, distance = lidar.get_goal_vector(self.drone.pos, self.end, self.sight_range)
        return np.concatenate([scan, direction, [distance]]).astype(np.float32)

    def _get_reward(self, prev_pos):
        prev_dist = np.linalg.norm(prev_pos - self.end)
        curr_dist = np.linalg.norm(self.drone.pos - self.end)
        if self._hit_something():
            return -10.0
        if curr_dist < 2.0:
            return +100.0
        return float(prev_dist - curr_dist)  # positive if closer

    def _hit_something(self):
        x, y, z = self.drone.pos.astype(int)
        return self.values[x, y, z] in [1, 2]

    def _is_done(self):
        return self._hit_something() or np.linalg.norm(self.drone.pos - self.end) < 2.0

    def _set_seed(self, seed):
        np.random.seed(seed)