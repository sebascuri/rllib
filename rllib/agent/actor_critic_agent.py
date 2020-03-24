"""Implementation of Model-Free Policy Gradient Algorithms."""

from rllib.agent.abstract_agent import AbstractAgent
from rllib.algorithms.ac import ActorCritic
from rllib.dataset.datatypes import Observation
from rllib.dataset.utilities import stack_list_of_tuples


class ActorCriticAgent(AbstractAgent):
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

    def __init__(self, policy, actor_optimizer, critic, critic_optimizer, criterion,
                 num_rollouts=1, target_update_frequency=1,
                 gamma=1.0, exploration_steps=0, exploration_episodes=0):
        super().__init__(gamma=gamma, exploration_steps=exploration_steps,
                         exploration_episodes=exploration_episodes)
        self.trajectories = []
        self.actor_critic = ActorCritic(policy, critic, criterion(reduction='none'),
                                        gamma)
        self.policy = self.actor_critic.policy

        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer

        self.num_rollouts = num_rollouts
        self.target_update_freq = target_update_frequency

    def observe(self, observation):
        """See `AbstractAgent.observe'."""
        super().observe(observation)
        self.trajectories[-1].append(observation)

    def start_episode(self):
        """See `AbstractAgent.start_episode'."""
        super().start_episode()
        self.trajectories.append([])

    def end_episode(self):
        """See `AbstractAgent.end_episode'."""
        if self.total_episodes % self.num_rollouts == 0:
            if self._training:
                self._train()
            self.trajectories = []

        if self.total_episodes % (self.target_update_freq * self.num_rollouts) == 0:
            self.actor_critic.update()

        super().end_episode()

    def _train(self):
        """Train Policy Gradient Agent."""
        trajectories = [Observation(*stack_list_of_tuples(t))
                        for t in self.trajectories]
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        losses = self.actor_critic(trajectories)

        total_loss = losses.actor_loss + losses.critic_loss
        total_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        # Update logs
        self.logger.update(actor_losses=losses.actor_loss.item(),
                           critic_losses=losses.critic_loss.item(),
                           td_errors=losses.td_error.abs().mean().item())
