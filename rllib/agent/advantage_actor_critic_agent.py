"""Implementation of Advantage-Actor Critic Agent."""

from rllib.algorithms.a2c import A2C
from .actor_critic_agent import ActorCriticAgent


class A2CAgent(ActorCriticAgent):
    """Implementation of the Advantage-Actor Critic.

    TODO: build compatible function approximation.

    References
    ----------
    Mnih, V., et al. (2016).
    Asynchronous methods for deep reinforcement learning. ICML.
    """

    def __init__(self, policy, actor_optimizer, critic, critic_optimizer, criterion,
                 num_rollouts=1, target_update_frequency=1,
                 gamma=1.0, exploration_steps=0, exploration_episodes=0):
        super().__init__(policy, actor_optimizer, critic, critic_optimizer, criterion,
                         num_rollouts, target_update_frequency, gamma,
                         exploration_steps, exploration_episodes)
        self.actor_critic = A2C(policy, critic, criterion(reduction='none'),
                                gamma)
        self.policy = self.actor_critic.policy
