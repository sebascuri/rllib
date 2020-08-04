from typing import Any, List

from torch.nn.modules.loss import _Loss

from rllib.dataset.datatypes import Observation
from rllib.policy import AbstractPolicy
from rllib.value_function import AbstractValueFunction

from .abstract_algorithm import AbstractAlgorithm, PGLoss

class REINFORCE(AbstractAlgorithm):
    eps: float = ...
    baseline: AbstractValueFunction
    criterion: _Loss
    def __init__(
        self,
        baseline: AbstractValueFunction,
        criterion: _Loss,
        *args: Any,
        **kwawrgs: Any,
    ) -> None: ...
    def forward_slow(self, trajectories: List[Observation]) -> PGLoss: ...
    def forward(self, trajectories: List[Observation], **kwargs: Any) -> PGLoss: ...
