from typing import Any, Optional, Type

from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer

from rllib.algorithms.reinforce import REINFORCE
from rllib.policy import AbstractPolicy
from rllib.value_function import AbstractQFunction, AbstractValueFunction

from .on_policy_agent import OnPolicyAgent

class REINFORCEAgent(OnPolicyAgent):
    algorithm: REINFORCE
    policy_optimizer: Optimizer
    baseline_optimizer: Optimizer
    target_update_frequency: int
    num_iter: int
    def __init__(
        self,
        policy: AbstractPolicy,
        optimizer: Optimizer,
        baseline: Optional[AbstractValueFunction] = ...,
        criterion: Optional[Type[_Loss]] = ...,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
