"""Implementation of an EXP3 Experience Replay Buffer."""

import torch

from .prioritized_experience_replay import PrioritizedExperienceReplay


class EXP3ExperienceReplay(PrioritizedExperienceReplay):
    r"""Sampler for EXP3-Sampler.

    The exp-3 algorithm maintains a sampling distribution over the index set of size K.
    The sampling probability at time t of index k is proportional to:

    ..math :: p_{k, t} = (1-\beta) w_{k, t} / \sum_k w_{k, t} + \beta / K,

    where \beta is a mixing parameter and w_{:,t} is a vector of weights computed as.

    ..math :: w_{k, t} = w_{k, 0} \exp^{\alpha \sum_{t'=0}^t r_{k, t'}},
    where r_{k, t} is an unbiased estimate of the reward obtained at time t at index k,
    and \alpha is the learning rate of the algorithm.

    Usually, the parameter r_{k, t} is not observed but only a (batch of) index(es)
    is (are) sampled.
    Let I_{t} be a sample from p_{k, t} at time t, and only r_t is observed.

    Then the following estimator is used as an unbiased estimate of r_{k, t}.

    ..math :: r_{t} / p_{:, t} 1[I_{t} = k]

    Parameters
    ----------
    max_len: int.
        Maximum length of Buffer.
    alpha: float, ParameterDecay, optional.
        Learning rate of Exp-3 algorithm.
    beta: float, ParameterDecay, optional.
        Mixing parameter with uniform distribution.
    epsilon: float, optional.
        Minimum sampling probability.
    max_priority: float, optional.
        Maximum value for the priorities.
        New observations are initialized with this value.
    transformations: list of transforms.AbstractTransform, optional.
        A sequence of transformations to apply to the dataset, each of which is a
        callable that takes an observation as input and returns a modified observation.
        If they have an `update` method it will be called whenever a new trajectory
        is added to the dataset.
    num_steps: int, optional.
        Number of steps in return vector.

    References
    ----------
    Auer, P., Cesa-Bianchi, N., Freund, Y., & Schapire, R. E. (2002).
    The nonstochastic multiarmed bandit problem. SIAM journal on computing.

    Lattimore, T., & Szepesvári, C. (2018).
    Bandit algorithms. Preprint.

    Bubeck, S., & Cesa-Bianchi, N. (2012).
    Regret analysis of stochastic and nonstochastic multi-armed bandit problems.
    Foundations and Trends® in Machine Learning.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def probabilities(self):
        """Get list of probabilities."""
        probs = self.weights[: len(self)].reciprocal()
        return probs / torch.sum(probs)

    def update(self, indexes, td):
        """Update experience replay sampling distribution with set of weights."""
        idx, inverse_idx, counts = torch.unique(
            indexes, return_counts=True, return_inverse=True
        )

        inv_prob = self.probabilities[indexes].reciprocal()
        self._priorities[indexes] += self.alpha() * td * inv_prob * counts[inverse_idx]

        self.max_priority = max(
            self.max_priority, torch.max(self._priorities[idx]).item()
        )
        self.alpha.update()
        self.beta.update()
        self._update_weights()

    def _update_weights(self):
        """Update priorities and weights."""
        num = len(self)
        probs = torch.exp(self._priorities[:num] - self.max_priority)
        probs = (1 - self.beta()) * probs / torch.sum(probs) + self.beta() / num

        weights = 1.0 / (probs * num)
        self.weights[:num] = weights
