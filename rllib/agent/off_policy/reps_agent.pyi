from typing import Any, Union

from rllib.algorithms.reps import REPS
from rllib.policy import AbstractPolicy
from rllib.util.parameter_decay import ParameterDecay
from rllib.value_function import AbstractValueFunction

from .off_policy_agent import OffPolicyAgent

class REPSAgent(OffPolicyAgent):
    algorithm: REPS
    def __init__(
        self,
        policy: AbstractPolicy,
        critic: AbstractValueFunction,
        eta: Union[float, ParameterDecay] = ...,
        entropy_regularization: bool = ...,
        learn_policy: bool = ...,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
    def _optimize_dual(self) -> None: ...
    def _fit_policy(self) -> None: ...
    def _optimize_loss(self, loss_name: str = ...) -> None: ...
