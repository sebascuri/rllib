from typing import Tuple

from .state_action_reward import StateActionReward

class LocomotionReward(StateActionReward):
    forward_reward_weight: float
    healthy_reward: float
    def __init__(
        self,
        dim_action: Tuple[int],
        action_cost_ratio: float,
        forward_reward_weight: float = ...,
        healthy_reward: float = ...,
    ) -> None: ...
