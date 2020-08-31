from typing import Any, Optional, Type, Union

from torch.nn.modules.loss import _Loss

from rllib.agent.off_policy.mpo_agent import MPOAgent
from rllib.algorithms.vmpo import VMPO
from rllib.policy import AbstractPolicy
from rllib.util.parameter_decay import ParameterDecay
from rllib.value_function import AbstractValueFunction

class VMPOAgent(MPOAgent):

    algorithm: VMPO
    def __init__(
        self,
        policy: AbstractPolicy,
        critic: AbstractValueFunction,
        criterion: Type[_Loss],
        num_action_samples: int = ...,
        epsilon: Union[ParameterDecay, float] = ...,
        epsilon_mean: Union[ParameterDecay, float] = ...,
        epsilon_var: Optional[Union[ParameterDecay, float]] = ...,
        regularization: bool = ...,
        top_k_fraction: float = ...,
        train_frequency: int = ...,
        num_rollouts: int = ...,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
