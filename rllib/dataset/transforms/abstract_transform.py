from abc import ABC, abstractmethod


class AbstractTransform(ABC):
    @abstractmethod
    def update(self, trajectory):
        raise NotImplementedError

    @abstractmethod
    def __call__(self, observation):
        raise NotImplementedError

    @abstractmethod
    def reverse(self, observation):
        raise NotImplementedError
