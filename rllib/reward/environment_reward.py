"""Reward implemented by querying an environment."""

import torch

from .abstract_reward import AbstractReward
from rllib.environment.abstract_environment import AbstractEnvironment


class EnvironmentReward(AbstractReward):
    """Quadratic Reward Function."""

    def __init__(self, environment: AbstractEnvironment):
        super().__init__()

        self.environment = environment

    def forward(self, state, action, next_state):
        """Get Reward distribution."""
        self.environment.state = state
        next_state, reward, done, _ = self.environment.step(action)
        return reward, torch.zeros(1)
