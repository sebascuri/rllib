from abc import ABCMeta, abstractmethod
from rllib.dataset.datatypes import Observation


class AbstractTransform(object, metaclass=ABCMeta):

    @abstractmethod
    def __call__(self, observation: Observation) -> Observation: ...

    @abstractmethod
    def inverse(self, observation: Observation) -> Observation: ...

    def update(self, observation: Observation) -> None: ...
