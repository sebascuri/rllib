from abc import ABC, abstractmethod


class AbstractExplorationStrategy(ABC):
    @abstractmethod
    def __str__(self):
        raise NotImplementedError

    @abstractmethod
    def __call__(self, action_distribution, steps=None):
        raise NotImplementedError
