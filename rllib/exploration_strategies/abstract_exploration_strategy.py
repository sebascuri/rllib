"""Interface for exploration strategies."""


from abc import ABC, abstractmethod
from rllib.util import ExponentialDecay

__all__ = ['AbstractExplorationStrategy']


class AbstractExplorationStrategy(ABC):
    """Interface for policies to control an environment.

    Parameters
    ----------
    start: float
        initial value of exploration parameter.
    end: float, optional
        final value of exploration parameter.
    decay: float, optional
        rate of decay of exploration parameter.

    Attributes
    ----------
    __call__(action_distribution, steps): ndarray or int
    return the action that the action_distribution

    """

    def __init__(self, start, end=None, decay=None, max_value=1, dimension=1):
        self.param = ExponentialDecay(start, end, decay)
        self.max_value = max_value
        self.dimension = dimension

    @abstractmethod
    def __call__(self, state=None):
        """Get noise from exploration strategy."""
        raise NotImplementedError
