"""State Action Reward Functions."""
from abc import ABCMeta
from typing import Optional, Tuple

from rllib.dataset.datatypes import Action, State, Reward
from rllib.model import AbstractModel

class StateActionReward(AbstractModel, metaclass=ABCMeta):
    dim_action: Tuple
    action_scale: Action
    sparse: bool
    action_cost_ratio: float
    def __init__(
        self,
        action_cost_ratio: float = ...,
        sparse: bool = ...,
        goal: Optional[State] = ...,
        action_scale: Action = ...,
    ) -> None: ...
    def action_reward(self, action: Action) -> Reward: ...
    def state_reward(
        self, state: State, next_state: Optional[State] = ...
    ) -> Reward: ...
