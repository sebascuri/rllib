"""Pendulum Swing-up Environment with full observation."""
import numpy as np
import torch
from gym.envs.classic_control.pendulum import PendulumEnv, angle_normalize

from rllib.reward.state_action_reward import StateActionReward
from rllib.reward.utilities import tolerance


class PendulumReward(StateActionReward):
    """Get Pendulum Reward."""

    dim_action = (1,)

    def __init__(self, ctrl_cost_weight=0.001, sparse=False, *args, **kwargs):
        super().__init__(ctrl_cost_weight=ctrl_cost_weight, sparse=sparse)

    def copy(self):
        """Get copy of reward model."""
        return PendulumReward(
            ctrl_cost_weight=self.ctrl_cost_weight, sparse=self.sparse
        )

    @staticmethod
    def state_sparse_reward(theta, omega):
        """Get sparse reward."""
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

    def __init__(self, reset_noise_scale=0.01, ctrl_cost_weight=0.001, sparse=False):
        self.base_mujoco_name = "Pendulum-v0"

        super().__init__()
        self.reset_noise_scale = reset_noise_scale
        self.state = np.zeros(2)
        self.last_u = None
        self._reward_model = PendulumReward(
            ctrl_cost_weight=ctrl_cost_weight, sparse=sparse
        )

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
        reward = self._reward_model(self._get_obs(), u)[0].item()

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u  # for rendering
        theta, omega = self.state

        inertia = self.m * self.l ** 2 / 3.0
        omega_dot = -3 * self.g / (2 * self.l) * np.sin(theta + np.pi) + u / inertia

        new_omega = omega + omega_dot * self.dt
        new_theta = theta + new_omega * self.dt  # Simplectic integration new_omega.

        new_omega = np.clip(new_omega, -self.max_speed, self.max_speed)

        self.state = np.array([new_theta, new_omega])
        next_obs = self._get_obs()
        return next_obs, reward, False, self._reward_model.info

    def reward_model(self):
        """Get reward model."""
        return self._reward_model.copy()
