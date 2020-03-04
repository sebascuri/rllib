"""Implementation of Advantage Actor-Critic Algorithm."""

from .advantage_actor_critic_agent import A2CAgent


class A2ECAgent(A2CAgent):
    """Implementation of Advantage Actor-(expected)Critic Algorithm.

    References
    ----------
    Schulman, John, et al. (2015)
    "High-dimensional continuous control using generalized advantage estimation." ICLR.
    """

    def _td_critic(self, state, action=None, reward=None, next_state=None, done=None,
                   *args, **kwargs):
        pred_q = self.critic(state, action)

        next_v = self.critic_target.value(next_state, self.policy) * (1 - done)

        target_q = reward + self.gamma * next_v

        return pred_q, target_q.detach()
