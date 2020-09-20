"""Pendulum Swing-up Environment with full observation."""
import numpy as np
import torch
from gym.envs.classic_control.pendulum import PendulumEnv, angle_normalize

from rllib.reward.state_action_reward import StateActionReward
from rllib.reward.utilities import tolerance


class PendulumReward(StateActionReward):
    """Get Pendulum Reward."""

    dim_action = (1,)

    def __init__(self, action_cost_ratio=0.001, sparse=False, *args, **kwargs):
        super().__init__(action_cost_ratio=action_cost_ratio, sparse=sparse)

    @staticmethod
    def state_sparse_reward(theta, omega):
        """Get sparse reward."""
        if not isinstance(theta, torch.Tensor):
            theta = torch.tensor(theta, dtype=torch.get_default_dtype())
        if not isinstance(omega, torch.Tensor):
            omega = torch.tensor(omega, dtype=torch.get_default_dtype())
        angle_tolerance = tolerance(torch.cos(theta), lower=0.95, upper=1.0, margin=0.3)
        velocity_tolerance = tolerance(omega, lower=-0.5, upper=0.5, margin=0.5)
        return angle_tolerance * velocity_tolerance

    @staticmethod
    def state_non_sparse_reward(theta, omega):
        """Get sparse reward."""
        theta = angle_normalize(theta)
        return -(theta ** 2 + 0.1 * omega ** 2)

    def state_reward(self, state, next_state=None):
        """Compute reward associated with state dynamics."""
        theta, omega = torch.atan2(state[..., 1], state[..., 0]), state[..., 2]
        if self.sparse:
            return self.state_sparse_reward(theta, omega)
        else:
            return self.state_non_sparse_reward(theta, omega)


class PendulumSwingUpEnv(PendulumEnv):
    """Pendulum Swing-up Environment."""

    def __init__(self, reset_noise_scale=0.01, action_cost_ratio=0.001, sparse=False):
        super().__init__()
        self.reset_noise_scale = reset_noise_scale
        self._ctrl_cost_weight = action_cost_ratio
        self.state = np.zeros(2)
        self.last_u = None
        self.sparse = sparse

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
        action = torch.tensor(u, dtype=torch.get_default_dtype())
        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u  # for rendering
        if self.sparse:
            ctrl_reward = PendulumReward.action_sparse_reward(action)
            state_reward = PendulumReward.state_sparse_reward(theta, omega)
        else:
            ctrl_reward = PendulumReward.action_non_sparse_reward(action)
            state_reward = PendulumReward.state_non_sparse_reward(theta, omega)

        ctrl_cost = -self._ctrl_cost_weight * ctrl_reward.item()
        state_cost = -state_reward.item()

        inertia = self.m * self.l ** 2 / 3.0
        omega_dot = -3 * self.g / (2 * self.l) * np.sin(theta + np.pi) + u / inertia

        new_omega = omega + omega_dot * self.dt
        new_theta = theta + new_omega * self.dt  # Simplectic integration new_omega.

        new_omega = np.clip(new_omega, -self.max_speed, self.max_speed)

        self.state = np.array([new_theta, new_omega])
        return self._get_obs(), -(ctrl_cost + state_cost), False, {}

    def reward_model(self):
        """Get reward model."""
        return PendulumReward(
            action_cost_ratio=self._ctrl_cost_weight, sparse=self.sparse
        )
