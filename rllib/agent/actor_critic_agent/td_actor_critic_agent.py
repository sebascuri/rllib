"""Implementation of TD Actor-Critic Algorithm Algorithm."""

from .abstract_actor_critic_agent import AbstractPolicyGradient
import torch


class TDACAgent(AbstractPolicyGradient):
    """Implementation of TD Actor-Critic Algorithm.

    References
    ----------
    Schulman, John, et al. (2015)
    "High-dimensional continuous control using generalized advantage estimation." ICLR.
    """

    def _return(self, state, action=None, reward=None, next_state=None, done=None):
        if self.baseline is not None:
            baseline = self.baseline(state).detach()
        else:
            baseline = torch.zeros_like(reward)

        td = reward + self.gamma * self.critic(next_state) - self.critic(state)
        return td - baseline

    def _td_critic(self, state, action=None, reward=None, next_state=None, done=None):
        pred_v = self.critic(state)
        next_v = self.critic_target(next_state) * (1 - done)

        target_v = reward + self.gamma * next_v

        return pred_v, target_v.detach()
