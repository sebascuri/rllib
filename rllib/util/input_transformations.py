"""Input transformations for spot."""
from abc import ABC, abstractmethod


class AbstractTransform(ABC):
    """Abstract Transformation definition."""

    extra_dim: int

    @abstractmethod
    def __call__(self, state):
        """Apply transformation."""
        raise NotImplementedError


class ComposeTransforms(AbstractTransform):
    """Compose a list of transformations."""

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
