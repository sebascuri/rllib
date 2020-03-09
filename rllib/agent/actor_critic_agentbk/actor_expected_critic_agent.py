"""Implementation of Episodic Q Actor-Critic Algorithm."""

from .actor_critic_agent import ACAgent


class AECAgent(ACAgent):
    """Implementation of Q Actor-(expected)Critic algorithm.

    This Actor critic algorithm updates at the end of each episode only.

    References
    ----------
    Sutton, Richard S., et al. "Policy gradient methods for reinforcement learning with
    function approximation." Advances in neural information processing systems. 2000.

    """

    def _td_critic(self, state, action=None, reward=None, next_state=None, done=None,
                   *args, **kwargs):
        pred_q = self.critic(state, action)

        next_v = self.critic_target.value(next_state, self.policy) * (1 - done)

        target_q = reward + self.gamma * next_v

        return pred_q, target_q.detach()
