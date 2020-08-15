from typing import Any

from rllib.algorithms.policy_evaluation.gae import GAE
from rllib.value_function import AbstractValueFunction

from .abstract_algorithm import AbstractAlgorithm

class REINFORCE(AbstractAlgorithm):
    gae: GAE
    def __init__(
        self, baseline: AbstractValueFunction, *args: Any, **kwargs: Any,
    ) -> None: ...
