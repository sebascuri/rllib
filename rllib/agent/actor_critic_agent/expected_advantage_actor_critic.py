"""Implementation of Episodic Q Actor-Critic Algorithm."""

from .expected_actor_critic import EACAgent


class EA2CAgent(EACAgent):
    """Implementation of Expected Q Actor-Critic algorithm.

    This Actor critic algorithm updates at the end of each episode only.

    References
    ----------
    Sutton, Richard S., et al. "Policy gradient methods for reinforcement learning with
    function approximation." Advances in neural information processing systems. 2000.

    """

    def _return(self, state, action=None, reward=None, next_state=None, done=None):
        value = self.critic_target.value(next_state, self.policy)
        return super()._return(state, action, reward, next_state, done) - value
