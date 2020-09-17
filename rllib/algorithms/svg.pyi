from typing import Any, Optional

from rllib.model import AbstractModel
from rllib.value_function import AbstractValueFunction

from .abstract_algorithm import AbstractAlgorithm

class SVG(AbstractAlgorithm):
    critic: AbstractValueFunction
    critic_target: AbstractValueFunction
    dynamical_model: AbstractModel
    reward_model: AbstractModel
    termination_model: Optional[AbstractModel]
    def __init__(
        self,
        dynamical_model: AbstractModel,
        reward_model: AbstractModel,
        termination_model: Optional[AbstractModel] = ...,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
