"""Implementation of a Transformation that normalizes attributes."""

import torch
import torch.jit
import torch.nn as nn

from rllib.dataset.transforms.utilities import rescale, update_mean, update_var

from .abstract_transform import AbstractTransform


class Normalizer(nn.Module):
    """Normalizer Class."""

    preserve_orign: bool

    def __init__(self, preserve_origin=False):
        super().__init__()
        self.mean = torch.zeros(1)
        self.variance = torch.ones(1)
        self.count = torch.tensor(0.0)
        self.preserve_origin = preserve_origin

    def forward(self, array):
        """See `AbstractTransform.__call__'."""
        if self.preserve_origin:
            scale = torch.sqrt(self.variance + self.mean ** 2)
            return array / scale
        else:
            return (array - self.mean) / torch.sqrt(self.variance)

    @torch.jit.export
    def inverse(self, array):
        """See `AbstractTransform.inverse'."""
        if self.preserve_origin:
            scale = torch.sqrt(self.variance + self.mean ** 2)
            return array * scale
        else:
            return self.mean + array * torch.sqrt(self.variance)

    @torch.jit.export
    def update(self, array):
        """See `AbstractTransform.update'."""
        while array.ndim <= 1:
            array = array.unsqueeze(0)
        new_mean = torch.mean(array, 0)
        new_var = torch.var(array, 0)
        if torch.any(torch.isnan(new_var)):
            new_var = torch.zeros_like(new_var)
        new_count = torch.tensor(1.0) * torch.tensor(array.shape[0])
        self.variance = update_var(
            self.mean, self.variance, self.count, new_mean, new_var, new_count
        )
        # If variance is too small, clamp the variance to 1.0
        self.variance[self.variance < 1e-6] = 1.0
        self.mean = update_mean(self.mean, self.count, new_mean, new_count)

        self.count += new_count


class StateNormalizer(AbstractTransform):
    r"""Implementation of a transformer that normalizes the states and next states.

    The state and next state of an observation are shifted by the mean and then are
    re-scaled with the standard deviation as:
        .. math:: state = (raw_state - mean) / std_dev

    The mean and standard deviation are computed with running statistics of the state.

    Parameters
    ----------
    preserve_origin: bool, optional (default=False)
        preserve the origin when rescaling.

    """

    def __init__(self, preserve_origin=False):
        super().__init__()
        self._normalizer = Normalizer(preserve_origin)

    def forward(self, observation):
        """See `AbstractTransform.__call__'."""
        scale = torch.diag_embed(1 / torch.sqrt(self._normalizer.variance))
        observation.state = self._normalizer(observation.state)
        observation.state_scale_tril = rescale(observation.state_scale_tril, scale)

        return observation

    @torch.jit.export
    def inverse(self, observation):
        """See `AbstractTransform.inverse'."""
        inv_scale = torch.diag_embed(torch.sqrt(self._normalizer.variance))
        observation.state = self._normalizer.inverse(observation.state)
        observation.state_scale_tril = rescale(observation.state_scale_tril, inv_scale)
        return observation

    @torch.jit.export
    def update(self, observation):
        """See `AbstractTransform.update'."""
        self._normalizer.update(observation.state)


class NextStateNormalizer(AbstractTransform):
    r"""Implementation of a transformer that normalizes the next states.

    The next state of an observation is shifted by the mean and then re-scaled with the
    standard deviation as:
        .. math:: state = (raw_state - mean) / std_dev

    The mean and standard deviation are computed with running statistics of the state.

    Parameters
    ----------
    preserve_origin: bool, optional (default=False)
        preserve the origin when rescaling.

    """

    def __init__(self, preserve_origin=False):
        super().__init__()
        self._normalizer = Normalizer(preserve_origin)

    def forward(self, observation):
        """See `AbstractTransform.__call__'."""
        scale = torch.diag_embed(1 / torch.sqrt(self._normalizer.variance))
        observation.next_state = self._normalizer(observation.next_state)
        observation.next_state_scale_tril = rescale(
            observation.next_state_scale_tril, scale
        )
        return observation

    @torch.jit.export
    def inverse(self, observation):
        """See `AbstractTransform.inverse'."""
        inv_scale = torch.diag_embed(torch.sqrt(self._normalizer.variance))
        observation.next_state = self._normalizer.inverse(observation.next_state)
        observation.next_state_scale_tril = rescale(
            observation.next_state_scale_tril, inv_scale
        )
        return observation

    @torch.jit.export
    def update(self, observation):
        """See `AbstractTransform.update'."""
        self._normalizer.update(observation.next_state)


class RewardNormalizer(AbstractTransform):
    """Implementation of a transformer that normalizes the rewards."""

    def __init__(self, preserve_origin=False):
        super().__init__()
        self._normalizer = Normalizer(preserve_origin)

    def forward(self, observation):
        """See `AbstractTransform.__call__'."""
        scale = torch.diag_embed(1 / torch.sqrt(self._normalizer.variance))
        observation.reward = self._normalizer(observation.reward)
        observation.reward_scale_tril = rescale(observation.reward_scale_tril, scale)
        return observation

    @torch.jit.export
    def inverse(self, observation):
        """See `AbstractTransform.inverse'."""
        inv_scale = torch.diag_embed(torch.sqrt(self._normalizer.variance))
        observation.reward = self._normalizer.inverse(observation.reward)
        observation.reward_scale_tril = rescale(
            observation.reward_scale_tril, inv_scale
        )
        return observation

    @torch.jit.export
    def update(self, observation):
        """See `AbstractTransform.update'."""
        self._normalizer.update(observation.reward)


class ActionNormalizer(AbstractTransform):
    """Implementation of a transformer that normalizes the action.

    The action of an observation is shifted by the mean and then re-scaled with the
    standard deviation as:
        .. math:: action = (raw_action - mean) / std_dev

    The mean and standard deviation are computed with running statistics of the action.

    Parameters
    ----------
    preserve_origin: bool, optional (default=False)
        preserve the origin when rescaling.

    """

    def __init__(self, preserve_origin=False):
        super().__init__()
        self._normalizer = Normalizer(preserve_origin)

    def forward(self, observation):
        """See `AbstractTransform.__call__'."""
        observation.action = self._normalizer(observation.action)
        return observation

    @torch.jit.export
    def inverse(self, observation):
        """See `AbstractTransform.inverse'."""
        observation.action = self._normalizer.inverse(observation.action)
        return observation

    @torch.jit.export
    def update(self, observation):
        """See `AbstractTransform.update'."""
        self._normalizer.update(observation.action)
