"""Vectorized Gym Pendulum Environment."""

import numpy as np
from gym.envs.classic_control.pendulum import PendulumEnv, angle_normalize

from rllib.environment.vectorized.util import VectorizedEnv
from rllib.util.utilities import get_backend


class VectorizedPendulumEnv(PendulumEnv, VectorizedEnv):
    """Vectorized implementation of Pendulum."""

    def step(self, action):
        """See `PendulumEnv.step()'."""
        g = self.g
        m = self.m
        length = self.l
        inertia = m * length ** 2
        bk = self.bk
        dt = self.dt
        theta, theta_dot = self.state[..., 0], self.state[..., 1]

        u = self.clip(action, -self.max_torque, self.max_torque)[..., 0]

        if not u.shape:
            self.last_u = u  # for rendering
        costs = angle_normalize(theta) ** 2 + 0.1 * theta_dot ** 2 + 0.001 * (u ** 2)

        theta_d_dot = -3 * g / (2 * length) * bk.sin(theta + np.pi) + 3.0 / inertia * u
        new_theta_dot = theta_dot + theta_d_dot * dt
        new_theta = theta + new_theta_dot * dt
        new_theta_dot = self.clip(new_theta_dot, -self.max_speed, self.max_speed)

        self.state = self.bk.stack((new_theta, new_theta_dot), -1)

        done = bk.zeros_like(costs, dtype=bk.bool)
        return self._get_obs(), -costs, done, {}

    def set_state(self, observation):
        """Set state from a given observation."""
        bk = get_backend(observation)
        self.state = bk.zeros_like(observation[..., :2])
        self.state[..., 0] = self.atan2(observation[..., 1], observation[..., 0])
        self.state[..., 1] = observation[..., 2]

    def _get_obs(self):
        theta, theta_dot = self.state[..., 0], self.state[..., 1]
        return self.bk.stack((self.bk.cos(theta), self.bk.sin(theta), theta_dot), -1)
