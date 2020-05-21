"""On Policy Actor-Critic Agent."""

from .on_policy_agent import OnPolicyAgent
from rllib.dataset.utilities import stack_list_of_tuples


class OnPolicyACAgent(OnPolicyAgent):
    """Template for an on-policy algorithm."""

    def __init__(self, env_name,
                 actor_optimizer, critic_optimizer,
                 num_iter=1,
                 target_update_frequency=1,
                 num_rollouts=1,
                 gamma=1.0, exploration_steps=0,
                 exploration_episodes=0, comment=''):
        super().__init__(env_name,
                         num_rollouts=num_rollouts,
                         gamma=gamma, exploration_steps=exploration_steps,
                         exploration_episodes=exploration_episodes, comment=comment)
        self.target_update_frequency = target_update_frequency
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.num_iter = num_iter

    def _train(self):
        """Train Policy Gradient Agent."""
        trajectories = [stack_list_of_tuples(t) for t in self.trajectories]

        for _ in range(self.num_iter):
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()

            losses = self.algorithm(trajectories)

            total_loss = losses.actor_loss + losses.critic_loss
            total_loss.backward()

            self.actor_optimizer.step()
            self.critic_optimizer.step()

            # Update logs
            self.logger.update(actor_losses=losses.actor_loss.item(),
                               critic_losses=losses.critic_loss.item(),
                               td_errors=losses.td_error.abs().mean().item())

            self.train_iter += 1
            if self.train_iter % self.target_update_frequency == 0:
                self.algorithm.update()
