"""Interface for reward models."""

from abc import ABCMeta, abstractmethod

import torch.jit
import torch.nn as nn


class AbstractReward(nn.Module, metaclass=ABCMeta):
    """Interface for Rewards of an Environment.

    A Reward is a model of the reward of the environment.

    Methods
    -------
    forward(state, action): Tensor, Union[Tensor, Tensor]
        return the next state distribution given a state and an action.

    """

    def __init__(self, goal=None):
        super().__init__()
        self.goal = goal

    @abstractmethod
    def forward(self, state, action, next_state):
        """Get reward distribution at current state and action."""
        raise NotImplementedError

    @torch.jit.export
    def set_goal(self, goal):
        """Set reward model goal."""
        if goal is None:
            return
        self.goal = goal
