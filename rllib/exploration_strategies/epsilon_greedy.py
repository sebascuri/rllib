"""Epsilon Greedy Exploration Strategy."""


from .abstract_exploration_strategy import AbstractExplorationStrategy
import numpy as np
from torch.distributions import Categorical, MultivariateNormal


__all__ = ['EpsGreedy']


def _mode(action_distribution):
    """Return the mode of a distribution.

    Parameters
    ----------
    action_distribution: torch.distributions.Distribution

    Returns
    -------
    ndarray or int

    """
    if type(action_distribution) is Categorical:
        return action_distribution.logits.argmax().numpy()
    elif type(action_distribution) is MultivariateNormal:
        return action_distribution.loc.numpy()
    else:
        raise NotImplementedError("""
        Action Distribution should be of type Categorical or MultivariateNormal but {}
        type was passed.
        """.format(type(action_distribution)))


class EpsGreedy(AbstractExplorationStrategy):
    """Implementation of Epsilon Greedy Exploration Strategy.

    An epsilon greedy exploration strategy chooses the greedy strategy with probability
    1-epsilon, and a random action with probability epsilon.

    If eps_end and eps_decay are not set, then epsilon will be always eps_start.
    If not, epsilon will decay exponentially at rate eps_decay from eps_start to
    eps_end.

    Parameters
    ----------
    eps_start: float
        initial value of epsilon.
    eps_end: float, optional
        final value of epsilon.
    eps_decay: int, optional
        epsilon rate of decay.

    """

    def __init__(self, eps_start, eps_end=None, eps_decay=None):
        self._eps_start = eps_start

        if eps_end is None:
            eps_end = eps_start
        self._eps_end = eps_end

        if eps_decay is None:
            eps_decay = 1
        self._eps_decay = eps_decay

    def __call__(self, action_distribution, steps=None):
        if np.random.random() > self.epsilon(steps):
            return _mode(action_distribution)
        else:
            return action_distribution.sample().numpy()

    def epsilon(self, steps=None):
        """Get current value for epsilon.

        Parameters
        ----------
        steps: int, optional

        Returns
        -------
        epsilon: float

        """
        if steps is None:
            return self._eps_start
        else:
            decay = (self._eps_start - self._eps_end) * np.exp(-steps / self._eps_decay)
            return self._eps_end + decay
