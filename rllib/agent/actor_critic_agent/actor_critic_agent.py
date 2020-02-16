"""Implementation of Episodic Q Actor-Critic Algorithm."""

from .abstract_actor_critic_agent import AbstractPolicyGradient
from rllib.value_function import AbstractQFunction
import torch


class ACAgent(AbstractPolicyGradient):
    """Implementation of Q Actor-(expected)Critic algorithm.

    This Actor critic algorithm updates at the end of each episode only.

    References
    ----------
    Sutton, Richard S., et al. "Policy gradient methods for reinforcement learning with
    function approximation." Advances in neural information processing systems. 2000.

    """

    critic: AbstractQFunction
    critic_target: AbstractQFunction

    def _return(self, state, action=None, reward=None, next_state=None, done=None):
        if self.baseline is not None:
            baseline = self.baseline(state).detach()
        else:
            baseline = torch.zeros_like(reward)

        return self.critic(state, action) - baseline

    def _td_critic(self, state, action=None, reward=None, next_state=None, done=None):
        pred_q = self.critic(state, action)

        next_policy_action = self.policy_target(next_state).sample()
        next_v = self.critic_target(next_state, next_policy_action) * (1 - done)

        target_q = reward + self.gamma * next_v

        return pred_q, target_q.detach()
