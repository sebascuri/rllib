"""Interface for exploration strategies."""


from abc import ABC, abstractmethod


class AbstractExplorationStrategy(ABC):
    """Interface for policies to control an environment.

    Attributes
    ----------
    __call__(action_distribution, steps): ndarray or int
    return the action that the action_distribution

    """

    @abstractmethod
    def __call__(self, action_distribution, steps=None):
        raise NotImplementedError
