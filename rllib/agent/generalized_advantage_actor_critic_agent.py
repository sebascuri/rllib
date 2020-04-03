"""Implementation of Advantage-Actor Critic Agent."""

from rllib.algorithms.gaac import GAAC
from .actor_critic_agent import ActorCriticAgent


class GAACAgent(ActorCriticAgent):
    """Implementation of the Advantage-Actor Critic.

    TODO: build compatible function approximation.

    References
    ----------
    Mnih, V., et al. (2016).
    Asynchronous methods for deep reinforcement learning. ICML.
    """

    def __init__(self, environment, policy, actor_optimizer, critic, critic_optimizer,
                 criterion, num_rollouts=1, num_iter=1, target_update_frequency=1,
                 lambda_=0.97,
                 gamma=1.0, exploration_steps=0, exploration_episodes=0):
        super().__init__(environment, policy, actor_optimizer, critic, critic_optimizer,
                         criterion, num_rollouts, num_iter, target_update_frequency,
                         gamma, exploration_steps, exploration_episodes)
        self.actor_critic = GAAC(policy, critic, criterion(reduction='none'), lambda_,
                                 gamma)
        self.policy = self.actor_critic.policy
