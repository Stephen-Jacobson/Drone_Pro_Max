"""Math for mixing the NN movement directions to motor thrusts"""

import numpy as np


class MotorMixer:
    motor_thrusts = np.array([0, 0, 0, 0])

    def __init__(self, arm_length, yaw_torque_coefficient, max_motor_thrust):
        self.arm_length = arm_length
        self.yaw_torque_coefficient = yaw_torque_coefficient
        self.max_motor_thrust = max_motor_thrust
        self.mixing_matrix = None
        self.mixing_matrix_inv = None
        self._build_mixing_matrix()

    def _build_mixing_matrix(self):
        """Build the 4x4 matrix mapping [T, tau_roll, tau_pitch, tau_yaw] ->
        per-motor thrusts for an X-config quad, and precompute its inverse.
        """
        raise NotImplementedError

    def mix(self, total_thrust, roll_torque, pitch_torque, yaw_torque):
        """[T, tau_roll, tau_pitch, tau_yaw] -> motor_thrusts[4], clipped to
        [0, max_motor_thrust] (motors can't spin backwards or exceed max RPM).
        """
        raise NotImplementedError
