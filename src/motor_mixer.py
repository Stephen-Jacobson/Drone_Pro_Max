"""Math for mixing the NN movement directions to motor thrusts"""

import numpy as np

class MotorMixer:
    def __init__(self, motor_offsets, motor_spin_directions, yaw_torque_coefficient, max_motor_thrust):
        self.motor_offsets = motor_offsets 
        self.motor_spin_directions = motor_spin_directions
        self.yaw_torque_coefficient = yaw_torque_coefficient
        self.max_motor_thrust = max_motor_thrust
        self.mixing_matrix = None
        self.mixing_matrix_inv = None
        self._build_mixing_matrix()

    def _build_mixing_matrix(self):
        """Build the 4x4 matrix mapping [T, tau_roll, tau_pitch, tau_yaw] ->
        per-motor thrusts for an X-config quad, and precompute its inverse.
        """
        thrust = np.array([1, 1, 1, 1]) # each motor's contribution to total thrust
        roll = np.array([ # each motor's contribution to roll (proportional to y-offset)
            self.motor_offsets[0][1],
            self.motor_offsets[1][1],
            self.motor_offsets[2][1],
            self.motor_offsets[3][1],
            ]) 
        pitch = np.array([ # each motor's contribution to pitch (proportional to x-offset)
            self.motor_offsets[0][0],
            self.motor_offsets[1][0],
            self.motor_offsets[2][0],
            self.motor_offsets[3][0],
        ])
        yaw = np.array([ # each motor's contribution to yaw (signed by spin direction, proportional to yaw_torque_coef)
            self.motor_spin_directions[0] * self.yaw_torque_coefficient,
            self.motor_spin_directions[1] * self.yaw_torque_coefficient,
            self.motor_spin_directions[2] * self.yaw_torque_coefficient,
            self.motor_spin_directions[3] * self.yaw_torque_coefficient,
        ])
        self.mixing_matrix = np.array([thrust, roll, pitch, yaw])
        self.mixing_matrix_inv = np.linalg.inv(self.mixing_matrix);

    def mix(self, total_thrust, roll_torque, pitch_torque, yaw_torque):
        """[T, tau_roll, tau_pitch, tau_yaw] -> motor_thrusts[4], clipped to
        [0, max_motor_thrust] (motors can't spin backwards or exceed max RPM).
        """
        r = np.array([total_thrust, roll_torque, pitch_torque, yaw_torque])
        t = self.mixing_matrix_inv @ r
        return np.clip(t, 0, self.max_motor_thrust) # return thrust array capped at max and min thrusts

if __name__ == "__main__":
    MOTOR_OFFSETS = np.array([
        [0.141, 0.141, 0.0], 
        [0.141, -0.141, 0.0],
        [-0.141, 0.141, 0.0],
        [-0.141, -0.141, 0.0],
    ])
    MOTOR_SPIN_DIRECTIONS = np.array([1, -1, -1, 1])     
    mm = MotorMixer(MOTOR_OFFSETS, MOTOR_SPIN_DIRECTIONS, 0.2, 3)
    thrusts = mm.mix(4.905, 0.1, -0.5, 0.02)
    print(thrusts);
