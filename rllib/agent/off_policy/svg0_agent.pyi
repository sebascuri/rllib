"""Implementation of SVG-0 Algorithm."""

from typing import Any, Type

import torch.nn.modules.loss as loss
from torch.nn.modules.loss import _Loss

from rllib.algorithms.svg0 import SVG0
from rllib.policy import AbstractPolicy
from rllib.value_function import AbstractQFunction

from .off_policy_agent import OffPolicyAgent

class SVG0Agent(OffPolicyAgent):
    algorithm: SVG0
    def __init__(
        self,
        critic: AbstractQFunction,
        policy: AbstractPolicy,
        criterion: Type[_Loss] = ...,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
