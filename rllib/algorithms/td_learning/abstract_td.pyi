from abc import ABCMeta, abstractmethod
from typing import List, Tuple

from torch import Tensor

from rllib.dataset import ExperienceReplay
from rllib.dataset.datatypes import Observation
from rllib.environment import AbstractEnvironment
from rllib.policy import AbstractPolicy
from rllib.value_function import NNValueFunction


class AbstractTDLearning(object, metaclass=ABCMeta):
    double_sample: bool = False
    dimension: int
    omega: Tensor
    theta: Tensor
    value_function: NNValueFunction
    environment: AbstractEnvironment
    policy: AbstractPolicy
    sampler: ExperienceReplay
    gamma: float
    lr_theta: float
    lr_omega: float
    exact_value_function: NNValueFunction

    def __init__(self, environment: AbstractEnvironment,
                 policy: AbstractPolicy,
                 sampler: ExperienceReplay,
                 value_function: NNValueFunction,
                 gamma: float,
                 lr_theta: float = 0.1, lr_omega: float = 0.1,
                 exact_value_function: NNValueFunction = None) -> None: ...


    def _step(self, state: Tensor) -> Tuple[Tensor, Tensor, bool]: ...

    def simulate(self, observation: Observation
                 ) -> Tuple[Tensor, Tensor, Tensor, Tensor]: ...

    def train(self, epochs: int) -> List[float]: ...

    @abstractmethod
    def _update(self, td_error: Tensor, phi: Tensor, next_phi: Tensor, weight: Tensor
                ) -> None: ...
