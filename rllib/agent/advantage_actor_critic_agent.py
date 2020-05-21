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

    def __init__(self, env_name, policy, actor_optimizer, critic, critic_optimizer,
                 criterion, num_iter=1, target_update_frequency=1,
                 train_frequency=0, num_rollouts=1,
                 gamma=1.0, exploration_steps=0, exploration_episodes=0, comment=''):
        super().__init__(env_name, policy=policy, actor_optimizer=actor_optimizer,
                         critic=critic,
                         critic_optimizer=critic_optimizer,
                         criterion=criterion, num_iter=num_iter,
                         target_update_frequency=target_update_frequency,
                         train_frequency=train_frequency, num_rollouts=num_rollouts,
                         gamma=gamma, exploration_steps=exploration_steps,
                         exploration_episodes=exploration_episodes, comment=comment)

        self.algorithm = A2C(policy, critic, criterion(reduction='none'),
                             gamma)
        self.policy = self.algorithm.policy
