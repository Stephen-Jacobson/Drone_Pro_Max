"""The physical drone body: a single rigid-body quadcopter in PyBullet."""

import numpy as np
import pybullet as p


class Quadrotor:
    # Physical constants
    MASS = 0.5  # kg, ~0.5-1kg
    ARM_LENGTH = 0.2 # meters 
    MAX_MOTOR_THRUST = 3 # N/motor
    DRAG_COEFFICIENT = 0.2 # increase up to 1 to simulate wind

    # Motor positions in body frame: front-left, front-right, back-left,
    # back-right (X-configuration), alternating CW/CCW spin direction.
    
    #if arm length is 0.2m and each motor sits at 45 deg from the body
    # then 0.2 * cos(45) = 0.141
    MOTOR_OFFSETS = np.array([
        [0.141, 0.141, 0.0], # fl
        [0.141, -0.141, 0.0], # fr
        [-0.141, 0.141, 0.0], # bl
        [-0.141, -0.141, 0.0], # br
    ])
    MOTOR_SPIN_DIRECTIONS = np.array([1, -1, -1, 1]) # +1 is CCW, -1 is CW. Order = fl,fr,bl,br

    def __init__(self, client_id, start_pos, start_orientation):
        self.client_id = client_id
        self.body_id = None
        raise NotImplementedError

    def _load_body(self, start_pos, start_orientation):
        """Create the single rigid-body box link (mass + inertia) in PyBullet."""
        raise NotImplementedError

    def apply_motor_thrusts(self, motor_thrusts: np.ndarray):
        """Apply per-motor upward force + summed yaw reaction torque.

        motor_thrusts: np.ndarray[4], thrust per motor in Newtons.
        """
        raise NotImplementedError

    def get_state(self):
        """Return (position, linear_velocity, orientation_rpy, angular_velocity)."""
        raise NotImplementedError

    def reset(self, start_pos, start_orientation):
        raise NotImplementedError
