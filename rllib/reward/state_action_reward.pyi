"""State Action Reward Functions."""
from abc import ABCMeta
from typing import Optional, Tuple

from torch import Tensor

from rllib.dataset.datatypes import Action, Reward, State
from rllib.model import AbstractModel

class StateActionReward(AbstractModel, metaclass=ABCMeta):
    dim_action: Tuple
    action_scale: Action
    sparse: bool
    ctrl_cost_weight: float
    def __init__(
        self,
        ctrl_cost_weight: float = ...,
        sparse: bool = ...,
        goal: Optional[State] = ...,
        action_scale: Action = ...,
    ) -> None: ...
    @staticmethod
    def action_sparse_reward(action: Tensor) -> Tensor: ...
    @staticmethod
    def action_non_sparse_reward(action: Tensor) -> Tensor: ...
    def action_reward(self, action: Action) -> Reward: ...
    def state_reward(
        self, state: State, next_state: Optional[State] = ...
    ) -> Reward: ...
