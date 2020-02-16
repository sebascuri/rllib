"""Implementation of Advantage Actor-Critic Algorithm Algorithm."""

from .actor_critic_agent import ACAgent


class A2CAgent(ACAgent):
    """Implementation of Advantage Actor-Critic Algorithm.

    References
    ----------
    Schulman, John, et al. (2015)
    "High-dimensional continuous control using generalized advantage estimation." ICLR.
    """

    def _return(self, state, action=None, reward=None, next_state=None, done=None):
        value = self.critic_target.value(next_state, self.policy)
        return super()._return(state, action, reward, next_state, done) - value
