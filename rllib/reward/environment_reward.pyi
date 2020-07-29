from typing import Any

from torch import Tensor

from rllib.dataset.datatypes import TupleDistribution
from rllib.environment.abstract_environment import AbstractEnvironment
from rllib.reward import AbstractReward

class EnvironmentReward(AbstractReward):
    environment: AbstractEnvironment
    def __init__(self, environment: AbstractEnvironment) -> None: ...
    def forward(self, *args: Tensor, **kwargs: Any) -> TupleDistribution: ...
