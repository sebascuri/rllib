"""Pendulum Swing-up Environment with full observation."""
import numpy as np
import torch
from gym.envs.classic_control.pendulum import PendulumEnv, angle_normalize

from rllib.reward.state_action_reward import StateActionReward


class PendulumReward(StateActionReward):
    """Get Pendulum Reward."""

    dim_action = (1,)

    def __init__(self, action_cost_ratio=0.001, *args, **kwargs):
        super().__init__(action_cost_ratio=action_cost_ratio)

    def scale(self, state, action):
        """Get scale."""
        return torch.zeros(1)

    def state_reward(self, state, next_state=None):
        """Compute reward associated with state dynamics."""
        th, thdot = torch.atan2(state[..., 1], state[..., 0]), state[..., 2]
        return -(th ** 2 + 0.1 * thdot ** 2)


class PendulumSwingUpEnv(PendulumEnv):
    """Pendulum Swing-up Environment."""

    def __init__(self, reset_noise_scale=0.01, action_cost_ratio=0.001):
        super().__init__()
        self.reset_noise_scale = reset_noise_scale
        self._ctrl_cost_weight = action_cost_ratio
        self.state = np.zeros(2)
        self.last_u = None

    def reset(self):
        """Reset to fix initial conditions."""
        x0 = np.array([np.pi, 0])
        self.state = self.np_random.uniform(
            low=x0 - self.reset_noise_scale, high=x0 + self.reset_noise_scale
        )

        self.last_u = None
        return self._get_obs()

    def step(self, u):
        """Override step method of pendulum env."""
        theta, omega = self.state

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u  # for rendering
        ctrl_cost = self._ctrl_cost_weight * (u ** 2)
        state_cost = angle_normalize(theta) ** 2 + 0.1 * omega ** 2

        inertia = self.m * self.l ** 2 / 3.0
        omega_dot = -3 * self.g / (2 * self.l) * np.sin(theta + np.pi) + u / inertia

        new_omega = omega + omega_dot * self.dt
        new_theta = theta + new_omega * self.dt  # Simplectic integration new_omega.

        new_omega = np.clip(new_omega, -self.max_speed, self.max_speed)

        self.state = np.array([new_theta, new_omega])
        return self._get_obs(), -(ctrl_cost + state_cost), False, {}

    def reward_model(self):
        """Get reward model."""
        return PendulumReward(action_cost_ratio=self._ctrl_cost_weight)
