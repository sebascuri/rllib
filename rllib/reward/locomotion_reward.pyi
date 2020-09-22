from typing import Tuple

from .state_action_reward import StateActionReward

class LocomotionReward(StateActionReward):
    forward_reward_weight: float
    healthy_reward: float
    def __init__(
        self,
        dim_action: Tuple[int],
        ctrl_cost_weight: float,
        forward_reward_weight: float = ...,
        healthy_reward: float = ...,
    ) -> None: ...
