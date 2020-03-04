"""Implementation of Episodic Q Actor-Critic Algorithm."""

from .abstract_expected_actor_critic_agent import AbstractExpectedActorCritic
import torch


class EACAgent(AbstractExpectedActorCritic):
    """Implementation of Expected Q Actor-Critic algorithm.

    This Actor critic algorithm updates at the end of each episode only.

    References
    ----------
    Sutton, Richard S., et al. "Policy gradient methods for reinforcement learning with
    function approximation." Advances in neural information processing systems. 2000.

    """

    def _return(self, state, action=None, reward=None, next_state=None, done=None,
                *args, **kwargs):
        if self.baseline is not None:
            baseline = self.baseline(state).detach()
        else:
            baseline = torch.zeros_like(reward)

        return self.critic(state, action) - baseline

    def _td_critic(self, state, action=None, reward=None, next_state=None, done=None,
                   *args, **kwargs):
        pred_q = self.critic(state, action)

        next_v = self.critic_target.value(next_state, self.policy) * (1 - done)

        target_q = reward + self.gamma * next_v

        return pred_q, target_q.detach()
