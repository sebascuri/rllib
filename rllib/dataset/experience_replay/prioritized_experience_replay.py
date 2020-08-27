"""Implementation of a Prioritized Experience Replay Buffer."""

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate

from rllib.dataset.datatypes import Observation
from rllib.util.parameter_decay import Constant, ParameterDecay

from .experience_replay import ExperienceReplay


class PrioritizedExperienceReplay(ExperienceReplay):
    r"""Implementation of Prioritized Experience Replay Algorithm.

    The priority is:
    ..math :: p_i = |\delta_i| + \epsilon_i,
    where \delta_i is the last TD-error.

    The observations are sampled according to the probability:
    ..math :: P(i) \propto p_i ^ \alpha,
    where \alpha is a parameter.

    The IS weights are given by:
    ..math :: w_i = (N P(i)) ^ \beta,
    where \beta is a parameter.

    Parameters
    ----------
    max_len: int.
        Maximum length of Buffer.
    alpha: float, ParameterDecay, optional.
        Parameter alpha for probabilities.
    beta: float, ParameterDecay, optional.
        Parameter beta for importance weights.
    epsilon: float, optional.
        Epsilon parameter for minimum priority.
    max_priority: float, optional.
        Maximum value for the priorities.
        New observations are initialized with this value.
    transformations: list of transforms.AbstractTransform, optional.
        A sequence of transformations to apply to the dataset, each of which is a
        callable that takes an observation as input and returns a modified observation.
        If they have an `update` method it will be called whenever a new trajectory
        is added to the dataset.

    References
    ----------
    Schaul, T., Quan, J., Antonoglou, I., & Silver, D. (2015).
    Prioritized experience replay. ICLR.
    """

    def __init__(
        self, alpha=0.6, beta=0.4, epsilon=0.01, max_priority=10.0, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        if not isinstance(alpha, ParameterDecay):
            alpha = Constant(alpha)
        self.alpha = alpha

        if not isinstance(beta, ParameterDecay):
            beta = Constant(beta)
        self.beta = beta

        if not isinstance(epsilon, torch.Tensor):
            epsilon = torch.tensor(epsilon)
        self.epsilon = epsilon

        self.max_priority = max_priority
        self._priorities = torch.zeros(self.max_len)
        self.weights = torch.zeros(self.max_len)

    @classmethod
    def from_other(cls, other, num_steps=None):
        """Initialize EXP3Experience Replay from another one."""
        new = cls(
            max_len=other.max_len,
            alpha=other.alpha,
            beta=other.beta,
            epsilon=other.epsilon,
            max_priority=other.max_priority,
            transformations=other.transformations,
            num_steps=num_steps if num_steps else other.num_steps,
        )

        for observation in other.memory:
            if isinstance(observation, Observation):
                new.append(observation)
        return new

    @property
    def priorities(self):
        """Get list of priorities."""
        return self._priorities

    @priorities.setter
    def priorities(self, value):
        """Set list of priorities."""
        self._priorities = value
        self._update_weights()

    @property
    def probabilities(self):
        """Get list of probabilities."""
        num = len(self)
        return self._priorities[:num] / torch.sum(self._priorities[:num])

    def sample_batch(self, batch_size):
        """Get a batch of data."""
        probs = self.probabilities.numpy()
        indices = np.random.choice(len(self), batch_size, p=probs / np.sum(probs))
        return default_collate([self[i] for i in indices])

    def append(self, observation):
        """Append new observation to the dataset.

        Parameters
        ----------
        observation: Observation

        Raises
        ------
        TypeError
            If the new observation is not of type Observation.
        """
        self._priorities[self.ptr] = self.max_priority
        super().append(observation)
        self._update_weights()

    def update(self, indexes, td_error):
        """Update experience replay sampling distribution with set of weights."""
        self._priorities[indexes] = (td_error + self.epsilon) ** self.alpha()
        self.alpha.update()
        self.beta.update()
        self._update_weights()

    def _update_weights(self):
        """Update priorities and weights."""
        num = len(self)
        weights = torch.pow(self.probabilities * num, -self.beta())
        self.weights[:num] = weights
