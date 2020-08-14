"""Reward implemented by querying an environment."""

import torch

from .abstract_reward import AbstractReward


class EnvironmentReward(AbstractReward):
    """Quadratic Reward Function."""

    def __init__(self, environment):
        super().__init__(goal=environment.goal)
        self.environment = environment

    def forward(self, state, action, next_state):
        """Get Reward distribution."""
        self.environment.state = state
        _, reward, _, _ = self.environment.step(action)
        return reward, torch.zeros(1)
