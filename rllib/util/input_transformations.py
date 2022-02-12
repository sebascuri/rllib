"""Input transformations for spot."""
from abc import ABC, abstractmethod

import torch


class AbstractTransform(ABC):
    """Abstract Transformation definition."""

    extra_dim: int

    @abstractmethod
    def __call__(self, state):
        """Apply transformation."""
        raise NotImplementedError


class ComposeTransforms(AbstractTransform):
    """Compose transformations."""

    def __init__(self, transforms):
        super().__init__()
        self.extra_dim = 0
        for transform in transforms:
            self.extra_dim += transform.extra_dim
        self.transforms = transforms

    def __call__(self, x):
        """Apply sequence of transformations."""
        for transform in self.transforms:
            x = transform(x)
        return x


class AngleToCosSin(AbstractTransform):
    """Transform angles to CosSin."""

    extra_dim = 1

    @staticmethod
    def __call__(x):
        """Transform state before applying function approximation."""
        angle, angular_velocity = torch.split(x, [1, x.shape[-1] - 1], dim=-1)
        return torch.cat((torch.cos(angle), torch.sin(angle), angular_velocity), dim=-1)


class RemoveAngle(AbstractTransform):
    """Remove angle from state vector."""

    extra_dim = -1

    @staticmethod
    def __call__(x):
        """Transform state before applying function approximation."""
        _, velocities = torch.split(x, [1, x.shape[-1] - 1], dim=-1)
        return velocities


class RemovePosition(AbstractTransform):
    """Remove position from state vector."""

    extra_dim = -2

    @staticmethod
    def __call__(x):
        """Transform state before applying function approximation."""
        _, other = torch.split(x, [2, x.shape[-1] - 2], -1)
        return other
