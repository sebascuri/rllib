"""Implementation of Model-Free Policy Gradient Algorithms."""

from rllib.algorithms.ac import ActorCritic

from .on_policy_agent import OnPolicyAgent


class ActorCriticAgent(OnPolicyAgent):
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

    def __init__(self, policy, critic, optimizer, criterion,
                 num_iter=1, target_update_frequency=1,
                 train_frequency=0, num_rollouts=1,
                 gamma=1.0, exploration_steps=0, exploration_episodes=0,
                 tensorboard=False, comment=''):
        super().__init__(optimizer=optimizer,
                         num_iter=num_iter,
                         target_update_frequency=target_update_frequency,
                         train_frequency=train_frequency, num_rollouts=num_rollouts,
                         gamma=gamma, exploration_steps=exploration_steps,
                         exploration_episodes=exploration_episodes,
                         tensorboard=tensorboard, comment=comment)
        self.algorithm = ActorCritic(policy, critic, criterion(reduction='none'), gamma)
        self.policy = self.algorithm.policy
