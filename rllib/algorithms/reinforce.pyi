from typing import Any, List

from torch.nn.modules.loss import _Loss

from rllib.algorithms.policy_evaluation.gae import GAE
from rllib.dataset.datatypes import Observation
from rllib.value_function import AbstractValueFunction

from .abstract_algorithm import AbstractAlgorithm, PGLoss

class REINFORCE(AbstractAlgorithm):
    baseline: AbstractValueFunction
    criterion: _Loss
    gae: GAE
    def __init__(
        self,
        baseline: AbstractValueFunction,
        criterion: _Loss,
        *args: Any,
        **kwawrgs: Any,
    ) -> None: ...
    def forward_slow(self, trajectories: List[Observation]) -> PGLoss: ...
    def forward(self, trajectories: List[Observation], **kwargs: Any) -> PGLoss: ...
