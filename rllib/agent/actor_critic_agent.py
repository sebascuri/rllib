"""Implementation of Model-Free Policy Gradient Algorithms."""

from rllib.agent.on_policy_ac_agent import OnPolicyACAgent
from rllib.algorithms.ac import ActorCritic


class ActorCriticAgent(OnPolicyACAgent):
    """Abstract Implementation of the Actor-Critic Agent.

    The AbstractEpisodicPolicyGradient algorithm implements the Actor-Critic algorithms.

    TODO: build compatible function approximation.

    References
    ----------
    Sutton, R. S., McAllester, D. A., Singh, S. P., & Mansour, Y. (2000).
    Policy gradient methods for reinforcement learning with function approximation.NIPS.

    Konda, V. R., & Tsitsiklis, J. N. (2000).
    Actor-critic algorithms. NIPS.
    """

    eps = 1e-12

    def __init__(self, env_name, policy, actor_optimizer, critic, critic_optimizer,
                 criterion, num_iter=1, target_update_frequency=1,
                 train_frequency=0, num_rollouts=1,
                 gamma=1.0, exploration_steps=0, exploration_episodes=0, comment=''):
        super().__init__(env_name,
                         actor_optimizer=actor_optimizer,
                         critic_optimizer=critic_optimizer,
                         num_iter=num_iter,
                         target_update_frequency=target_update_frequency,
                         train_frequency=train_frequency, num_rollouts=num_rollouts,
                         gamma=gamma, exploration_steps=exploration_steps,
                         exploration_episodes=exploration_episodes, comment=comment)
        self.algorithm = ActorCritic(policy, critic, criterion(reduction='none'), gamma)
        self.policy = self.algorithm.policy
