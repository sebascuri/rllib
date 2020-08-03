from typing import Any, List, Type, Union

from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer

from rllib.algorithms.sarsa import SARSA
from rllib.dataset.datatypes import Observation
from rllib.policy import AbstractQFunctionPolicy
from rllib.value_function import AbstractQFunction

from .on_policy_agent import OnPolicyAgent

class SARSAAgent(OnPolicyAgent):
    algorithm: SARSA
    policy: AbstractQFunctionPolicy
    optimizer: Optimizer
    target_update_frequency: int
    last_observation: Union[None, Observation]
    num_iter: int
    batch_size: int
    trajectory = List[Observation]
    def __init__(
        self,
        q_function: AbstractQFunction,
        policy: AbstractQFunctionPolicy,
        criterion: Type[_Loss],
        optimizer: Optimizer,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
