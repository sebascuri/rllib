"""Python Script Template."""
from abc import ABCMeta

from .abstract_algorithm import AbstractAlgorithm


class DerivedAlgorithm(AbstractAlgorithm, metaclass=ABCMeta):
    """Derived Algorithm.

    A derived algorithm has a base algorithm and "inherits" its properties but overrides
    some methods.
    """

    def __init__(self, base_algorithm, *args, **kwargs):
        self.base_algorithm_name = base_algorithm.__class__.__name__
        super().__init__(
            **{**base_algorithm.__dict__, **dict(base_algorithm.named_modules())}
        )
        self.base_algorithm = base_algorithm

    def forward(self, observation):
        """Rollout model and call base algorithm with transitions."""
        raise NotImplementedError

    def update(self):
        """Update base algorithm."""
        self.base_algorithm.update()

    def reset(self):
        """Reset base algorithm."""
        super().reset()
        self.base_algorithm.reset()

    def info(self):
        """Get info from base algorithm."""
        return {**self.base_algorithm.info(), **self._info}

    def reset_info(self):
        """Reset info from base algorithm."""
        super().reset_info()
        self.base_algorithm.reset_info()

    def set_policy(self, new_policy):
        """Set policy in base algorithm."""
        super().set_policy(new_policy)
        self.base_algorithm.set_policy(new_policy)
