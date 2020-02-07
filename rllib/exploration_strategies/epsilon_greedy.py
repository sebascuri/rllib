"""Epsilon Greedy Exploration Strategy."""


from .abstract_exploration_strategy import AbstractExplorationStrategy
import numpy as np
from .utilities import argmax

__all__ = ['EpsGreedy']


class EpsGreedy(AbstractExplorationStrategy):
    """Implementation of Epsilon Greedy Exploration Strategy.

    An epsilon greedy exploration strategy chooses the greedy strategy with probability
    1-epsilon, and a random action with probability epsilon.

    If eps_end and eps_decay are not set, then epsilon will be always eps_start.
    If not, epsilon will decay exponentially at rate eps_decay from eps_start to
    eps_end.

    """

    def __call__(self, action_distribution, steps=None):
        """See `AbstractExplorationStrategy.__call__'."""
        epsilon = self.param(steps)
        if np.random.random() > epsilon:
            return argmax(action_distribution)
        else:
            return action_distribution.sample().numpy()
