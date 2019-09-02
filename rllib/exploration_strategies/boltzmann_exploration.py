"""Epsilon Greedy Exploration Strategy."""


from .abstract_exploration_strategy import AbstractExplorationStrategy
from torch.distributions import Categorical, MultivariateNormal
import numpy as np


__all__ = ['BoltzmannExploration']


class BoltzmannExploration(AbstractExplorationStrategy):
    """Implementation of Boltzmann Exploration Strategy.

    An boltzmann exploration strategy samples an action with the probability of the
    original policy, but scaled with a temperature parameter.

    If eps_end and eps_decay are not set, then epsilon will be always eps_start.
    If not, epsilon will decay exponentially at rate eps_decay from eps_start to
    eps_end.

    Parameters
    ----------
    t_start: float
        initial value of temperature.
    t_end: float, optional
        final value of temperature.
    t_decay: int, optional
        temperature rate of decay.

    """

    def __init__(self, t_start, t_end=None, t_decay=None):
        self._t_start = t_start

        if t_end is None:
            t_end = t_start
        self._t_end = t_end

        if t_decay is None:
            t_decay = 1
        self._t_decay = t_decay

    def __call__(self, action_distribution, steps=None):
        """See `AbstractExplorationStrategy.__call__'."""
        t = self.temperature(steps)
        if type(action_distribution) is Categorical:
            d = Categorical(logits=action_distribution.logits / (t + 1e-12))
        else:
            d = MultivariateNormal(
                loc=action_distribution.loc,
                covariance_matrix=action_distribution.covariance_matrix * (t + 1e-12))

        return d.sample().numpy()

    def temperature(self, steps=None):
        """Get current value for epsilon.

        If steps is None return initial temperature, else the temperature decayed
        according to the exponential decay parameters.

        Parameters
        ----------
        steps: int, optional

        Returns
        -------
        temperature: float

        """
        if steps is None:
            return self._t_start
        else:
            decay = (self._t_start - self._t_end) * np.exp(-steps / self._t_decay)
            return self._t_end + decay
