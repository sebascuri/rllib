"""Implementation of Expected-Actor Critic Agent."""

from rllib.algorithms.eac import ExpectedActorCritic
from .actor_critic_agent import ActorCriticAgent


class ExpectedActorCriticAgent(ActorCriticAgent):
    """Implementation of the Advantage-Actor Critic.

    TODO: build compatible function approximation.

    References
    ----------
    Ciosek, K., & Whiteson, S. (2018).
    Expected policy gradients. AAAI.
    """

    def __init__(self, environment, policy, actor_optimizer, critic, critic_optimizer,
                 criterion, num_rollouts=1, num_iter=1, target_update_frequency=1,
                 gamma=1.0, exploration_steps=0, exploration_episodes=0):
        super().__init__(environment, policy, actor_optimizer, critic, critic_optimizer,
                         criterion, num_rollouts, num_iter, target_update_frequency,
                         gamma, exploration_steps, exploration_episodes)
        self.actor_critic = ExpectedActorCritic(policy, critic,
                                                criterion(reduction='none'), gamma)
        self.policy = self.actor_critic.policy
