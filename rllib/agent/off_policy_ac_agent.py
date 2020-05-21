"""Off Policy Actor Critic Agent."""

from .off_policy_agent import OffPolicyAgent


class OffPolicyACAgent(OffPolicyAgent):
    """Template for an on-policy algorithm."""

    def __init__(self, env_name, actor_optimizer,
                 memory, critic_optimizer=None, batch_size=32,
                 num_iter=1,
                 target_update_frequency=1,
                 policy_update_frequency=1,
                 train_frequency=0, num_rollouts=1,
                 gamma=1.0, exploration_steps=0, exploration_episodes=0, comment=''):
        super().__init__(env_name,
                         memory=memory, batch_size=batch_size,
                         train_frequency=train_frequency, num_rollouts=num_rollouts,
                         gamma=gamma, exploration_steps=exploration_steps,
                         exploration_episodes=exploration_episodes, comment=comment)
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.target_update_frequency = target_update_frequency
        self.policy_update_frequency = policy_update_frequency
        self.num_iter = num_iter

    def _train(self):
        """Train the Actor-Critic Agent."""
        for _ in range(self.num_iter):
            observation, idx, weight = self.memory.get_batch(self.batch_size)

            if self.critic_optimizer is not None:
                self.critic_optimizer.zero_grad()
            self.actor_optimizer.zero_grad()

            ans = self.algorithm(
                observation.state, observation.action, observation.reward,
                observation.next_state, observation.done
            )

            # Back-propagate critic loss.
            critic_loss = (weight.detach() * ans.critic_loss).mean()
            critic_loss.backward()

            # Back-propagate actor loss.
            actor_loss = (weight.detach() * ans.actor_loss).mean()
            if self.total_steps % self.policy_update_frequency == 0:
                actor_loss.backward()

            # Update actor and critic.
            self.critic_optimizer.step()
            self.actor_optimizer.step()

            # Update memory
            self.memory.update(idx, ans.td_error.abs().detach())

            # Update logs
            self.logger.update(actor_losses=actor_loss.item(),
                               critic_losses=critic_loss.item(),
                               td_errors=ans.td_error.abs().mean().item())

            self.counters['train_steps'] += 1
            if self.train_steps % self.target_update_frequency == 0:
                self.algorithm.update()
