"""Fitted value evaluation agent."""

from typing import Any, Type

from torch.nn.modules.loss import _Loss

from rllib.algorithms.fitted_value_evaluation import FittedValueEvaluationAlgorithm
from rllib.policy import AbstractPolicy
from rllib.value_function import AbstractQFunction

from .off_policy_agent import OffPolicyAgent

class FittedValueEvaluationAgent(OffPolicyAgent):
    algorithm: FittedValueEvaluationAlgorithm
    policy: AbstractPolicy
    def __init__(
        self,
        critic: AbstractQFunction,
        policy: AbstractPolicy,
        criterion: Type[_Loss] = ...,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
