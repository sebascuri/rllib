"""Implementation of REINFORCE Algorithm."""

from .abstract_episodic_policy_gradient_agent import AbstractEpisodicPolicyGradient
from rllib.value_function import AbstractQFunction


class EpisodicA2CAgent(AbstractEpisodicPolicyGradient):
    """Implementation of Advantage Actor-Critic Algorithm.

    References
    ----------
    Schulman, John, et al. (2015)
    "High-dimensional continuous control using generalized advantage estimation." ICLR.
    """

    critic: AbstractQFunction
    critic_target: AbstractQFunction

    def _value_estimate(self, trajectories):
        values = []
        for trajectory in trajectories:
            r = trajectory.reward
            next_v = self.critic(trajectory.next_state) * (1 - trajectory.done)
            v = self.critic(trajectory.state)
            values.append(r + self.gamma * next_v - v)
        return values

    def _td_base(self, state, action, reward, next_state, done, value_estimate=None):
        next_v = self.baseline(next_state) * (1 - done)
        target_v = reward + self.gamma * next_v
        return self.baseline(state), target_v.detach()

    def _td_critic(self, state, action, reward, next_state, done):
        pred_q = self.critic(state)
        next_v = self.critic_target(next_state) * (1 - done)
        target_q = reward + self.gamma * next_v
        return pred_q, target_q.detach()
