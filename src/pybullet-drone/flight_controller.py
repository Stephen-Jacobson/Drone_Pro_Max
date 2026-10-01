"""Interprets the network's raw direction vector into motor thrusts.

Cascaded control: an outer loop (once per RL step) turns the direction
vector into a desired thrust + attitude; an inner loop (every physics
substep) runs attitude PID and hands off to the motor mixer.
"""

import numpy as np

GRAVITY = 9.81


class FlightController:
    def __init__(self, quadrotor, motor_mixer, max_speed, outer_gains, inner_gains):
        self.quadrotor = quadrotor
        self.motor_mixer = motor_mixer
        self.max_speed = max_speed
        self.outer_gains = outer_gains  # P/PD gains, velocity -> acceleration
        self.inner_gains = inner_gains  # PID gains, attitude -> torque
        self.desired_attitude = np.zeros(3)  # roll, pitch, yaw (yaw held at 0)
        self.desired_thrust = 0.0

    def outer_step(self, direction_vector: np.ndarray, state):
        """Run once per RL step.

        direction_vector: network's raw 3D action, Bounded(-1, 1).
        state: drone state from quadrotor.get_state().

        direction_vector -> desired_velocity (scaled by max_speed) -> P/PD
        against current velocity -> desired_acceleration -> gravity
        compensation -> thrust vector -> (thrust magnitude, desired roll/pitch).
        Updates self.desired_thrust and self.desired_attitude.
        """
        raise NotImplementedError

    def inner_step(self, state):
        """Run once per physics substep (~240Hz).

        PID on desired vs. actual roll/pitch/yaw -> (tau_roll, tau_pitch, tau_yaw).
        Mixes with self.desired_thrust via motor_mixer.mix() and applies the
        result via quadrotor.apply_motor_thrusts().
        """
        raise NotImplementedError

    def step(self, direction_vector: np.ndarray, n_substeps: int):
        """Run one outer_step, then n_substeps of inner_step + physics stepping."""
        raise NotImplementedError
