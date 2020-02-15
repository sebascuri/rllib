"""Implementation of Expected Actor Critic Algorithm."""

from .actor_critic_agent import EpisodicACAgent
from rllib.value_function import AbstractQFunction


class EpisodicEACAgent(EpisodicACAgent):
    """Implementation of Expected Actor Critic algorithm.

    This is the Actor-Critic algorithm, except that the Q-Function is learned using
    a variant of Expected SARSA. Note that this is not the expected policy gradient
    algorithm.

    References
    ----------
    Sutton, Richard S., et al. "Policy gradient methods for reinforcement learning with
    function approximation." Advances in neural information processing systems. 2000.

    """

    critic: AbstractQFunction
    critic_target: AbstractQFunction

    def _td_critic(self, state, action, reward, next_state, done):
        pred_q = self.critic(state, action)
        next_v = self.critic_target.value(next_state, self.policy) * (1 - done)
        target_q = reward + self.gamma * next_v
        return pred_q, target_q.detach()
