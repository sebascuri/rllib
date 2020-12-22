from typing import Any, Optional

from rllib.algorithms.abstract_mb_algorithm import AbstractMBAlgorithm
from rllib.model import AbstractModel
from rllib.policy import AbstractPolicy
from rllib.util.utilities import RewardTransformer
from rllib.value_function import AbstractQFunction, AbstractValueFunction

class ModelBasedQFunction(AbstractQFunction):
    sim: AbstractMBAlgorithm
    policy: Optional[AbstractPolicy]
    value_function: Optional[AbstractValueFunction]
    gamma: float
    lambda_: float
    reward_transformer: RewardTransformer
    entropy_regularization: float
    def __init__(
        self,
        dynamical_model: AbstractModel,
        reward_model: AbstractModel,
        num_steps: int = ...,
        num_samples: int = ...,
        termination_model: Optional[AbstractModel] = ...,
        policy: Optional[AbstractPolicy] = ...,
        value_function: Optional[AbstractValueFunction] = ...,
        gamma: float = ...,
        lambda_: float = ...,
        reward_transformer: RewardTransformer = ...,
        entropy_regularization: float = ...,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
    def set_policy(self, new_policy: AbstractPolicy) -> None: ...
