from abc import ABC, abstractmethod


class AbstractExplorationStrategy(ABC):
    @abstractmethod
    def __call__(self, action_distribution, steps=None):
        raise NotImplementedError
