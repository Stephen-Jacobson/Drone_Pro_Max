"""RL environment glue: same role as drone_env.py, backed by real PyBullet
physics instead of the voxel-grid teleport mechanic.
"""

import numpy as np
import pybullet as p
import torch
from tensordict import TensorDict
from torchrl.data import Bounded, Composite, Unbounded
from torchrl.envs import EnvBase

from quadrotor import Quadrotor
from motor_mixer import MotorMixer
from flight_controller import FlightController

GOAL_RADIUS = 5.0
ARENA_SIZE = None  # bounded box arena half-extent
RL_HZ = 50
PHYSICS_HZ = 240
SUBSTEPS_PER_STEP = PHYSICS_HZ // RL_HZ


class PyBulletDroneEnv(EnvBase):
    def __init__(self, gui=False):
        super().__init__()
        self.client_id = None
        self.quadrotor = None
        self.motor_mixer = None
        self.flight_controller = None
        self.target = None
        self.current_step = 0

        obs_size = 3 + 1 + 3 + 3  # direction-to-target + distance + velocity + rpy

        self.observation_spec = Composite(observation=Unbounded(shape=(obs_size,)))
        self.action_spec = Bounded(low=-1, high=1, shape=(3,))
        self.reward_spec = Unbounded(shape=(1,))

    def _build_arena(self):
        """Floor + 4 walls as static PyBullet box colliders."""
        raise NotImplementedError

    def _spawn_target(self):
        """Random point within the arena interior, margin away from walls."""
        raise NotImplementedError

    def _step(self, tensordict):
        self.current_step += 1
        action = tensordict["action"].cpu().numpy()

        self.flight_controller.step(action, SUBSTEPS_PER_STEP)

        obs = self._get_obs()
        reward = self._get_reward()
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
        raise NotImplementedError

    def _get_obs(self):
        """direction-to-target(3) + normalized distance(1) + linear velocity(3)
        + roll/pitch/yaw(3) — velocity and attitude are needed now that the
        drone has real momentum and tilt, unlike the grid world.
        """
        raise NotImplementedError

    def _get_reward(self):
        """Small per-step penalty, large positive on reaching target
        (GOAL_RADIUS), large negative on wall collision or excessive tilt
        (treated as a crash). Spawns a new target on reach, same episode.
        """
        raise NotImplementedError

    def _hit_something(self):
        raise NotImplementedError

    def _is_done(self):
        raise NotImplementedError

    def _set_seed(self, seed):
        np.random.seed(seed)
