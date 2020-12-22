from typing import Any

from torch import Tensor

from rllib.policy import AbstractPolicy
from rllib.value_function import AbstractQFunction, AbstractValueFunction

class IntegrateQValueFunction(AbstractValueFunction):
    """Value function that arises from integrating a q function with a policy.

    Parameters
    ----------
    q_function: AbstractQFunction
        q _function.
    policy: AbstractPolicy
        q _function.
    num_samples: int, optional (default=15).
        Number of states in discrete environments.
    """

    q_function: AbstractQFunction
    policy: AbstractPolicy
    num_samples: int
    def __init__(
        self,
        q_function: AbstractQFunction,
        policy: AbstractPolicy,
        num_samples: int = ...,
    ) -> None: ...
    def forward(self, *args: Tensor, **kwargs: Any) -> Tensor: ...
    def set_policy(self, new_policy: AbstractPolicy) -> None: ...
