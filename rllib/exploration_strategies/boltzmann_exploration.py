"""Boltzmann Sampling Exploration Strategy."""

from .abstract_exploration_strategy import AbstractExplorationStrategy
from torch.distributions import Categorical, MultivariateNormal


__all__ = ['BoltzmannExploration']


class BoltzmannExploration(AbstractExplorationStrategy):
    """Implementation of Boltzmann Exploration Strategy.

    An boltzmann exploration strategy samples an action with the probability of the
    original policy, but scaled with a temperature parameter.

    If eps_end and eps_decay are not set, then epsilon will be always eps_start.
    If not, epsilon will decay exponentially at rate eps_decay from eps_start to
    eps_end.


    """

    def __call__(self, action_distribution, steps=None):
        """See `AbstractExplorationStrategy.__call__'."""
        temperature = self.param(steps) + + 1e-12
        if type(action_distribution) is Categorical:
            d = Categorical(logits=action_distribution.logits / temperature)
        else:
            d = MultivariateNormal(
                loc=action_distribution.loc,
                covariance_matrix=action_distribution.covariance_matrix * temperature)

        return d.sample().numpy()
