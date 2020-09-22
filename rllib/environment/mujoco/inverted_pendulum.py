"""Inverted Pendulum Environment with full observation."""
import gym.error
import numpy as np
import torch

from rllib.reward.state_action_reward import StateActionReward

from .locomotion import LargeStateTermination

try:
    from gym.envs.mujoco.inverted_pendulum import InvertedPendulumEnv
except (ModuleNotFoundError, gym.error.DependencyNotInstalled):
    InvertedPendulumEnv = object


class InvertedPendulumReward(StateActionReward):
    """Inverted Pendulum Reward model."""

    dim_action = (1,)

    def __init__(self, ctrl_cost_weight=0.0):
        super().__init__(ctrl_cost_weight=ctrl_cost_weight)

    def copy(self):
        """Copy reward model."""
        return InvertedPendulumReward(ctrl_cost_weight=self.ctrl_cost_weight)

    def state_reward(self, state, next_state=None):
        """Compute State reward."""
        return torch.ones_like(state[..., 0])


class MBInvertedPendulumEnv(InvertedPendulumEnv):
    """Inverted Pendulum Environment."""

    def __init__(self, ctrl_cost_weight=0.0):
        self.base_mujoco_name = "InvertedPendulum-v2"
        self._reward_model = InvertedPendulumReward(ctrl_cost_weight=ctrl_cost_weight)
        self._termination_model = LargeStateTermination(
            z_dim=1, healthy_state_range=(-np.inf, np.inf), healthy_z_range=(-0.2, 0.2)
        )
        super().__init__()

    def reward_model(self):
        """Get reward model."""
        return self._reward_model.copy()

    def termination_model(self):
        """Get reward model."""
        return self._termination_model.copy()

    def step(self, a):
        """See `AbstractEnvironment.step()'."""
        obs = self._get_obs()
        reward = self._reward_model(obs, a)[0].item()
        done = self._termination_model(obs, a)
        self.do_simulation(a, self.frame_skip)
        next_obs = self._get_obs()

        return next_obs, reward, done, self._reward_model.info
