"""Locomotion Reward."""
from .state_action_reward import StateActionReward


class LocomotionReward(StateActionReward):
    r"""A locomotion reward model is used for locomotion robots.

    The reward function is computed as:
    r(s, a) = velocity + healthy + action_reward.

    The action reward is computed from the state-action reward.
    The velocity is the first component of the state.
    """

    def __init__(
        self,
        dim_action,
        ctrl_cost_weight,
        forward_reward_weight=1.0,
        healthy_reward=0.0,
    ):
        self.dim_action = dim_action
        super().__init__(ctrl_cost_weight=ctrl_cost_weight)
        self.forward_reward_weight = forward_reward_weight
        self.healthy_reward = healthy_reward

    def copy(self):
        """Get copy of locomotion reward."""
        return LocomotionReward(
            dim_action=self.dim_action,
            ctrl_cost_weight=self.ctrl_cost_weight,
            forward_reward_weight=self.forward_reward_weight,
            healthy_reward=self.healthy_reward,
        )

    def state_reward(self, state, next_state=None):
        """Get reward that corresponds to the states."""
        return self.forward_reward_weight * state[..., 0] + self.healthy_reward
