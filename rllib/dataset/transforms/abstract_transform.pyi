from abc import ABC, abstractmethod
from .. import Observation


class AbstractTransform(ABC):

    @abstractmethod
    def __call__(self, observation: Observation) -> Observation: ...

    @abstractmethod
    def inverse(self, observation: Observation) -> Observation: ...

    def update(self, observation: Observation) -> None: ...
