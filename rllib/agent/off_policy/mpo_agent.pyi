from typing import Any, Optional, Type, Union

from torch.nn.modules.loss import _Loss

from rllib.agent.off_policy.off_policy_agent import OffPolicyAgent
from rllib.algorithms.mpo import MPO
from rllib.policy import AbstractPolicy
from rllib.util.parameter_decay import ParameterDecay
from rllib.value_function import AbstractQFunction

class MPOAgent(OffPolicyAgent):

    algorithm: MPO
    def __init__(
        self,
        policy: AbstractPolicy,
        critic: AbstractQFunction,
        criterion: Type[_Loss] = ...,
        num_action_samples: int = ...,
        epsilon: Union[ParameterDecay, float] = ...,
        epsilon_mean: Union[ParameterDecay, float] = ...,
        epsilon_var: Optional[Union[ParameterDecay, float]] = ...,
        kl_regularization: bool = ...,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
