"""Off Policy Agent."""

from rllib.agent.abstract_agent import AbstractAgent
from rllib.dataset.utilities import average_named_tuple


class OffPolicyAgent(AbstractAgent):
    """Template for an on-policy algorithm."""

    def __init__(self, env_name, memory, optimizer,
                 target_update_frequency=1, num_iter=1, batch_size=64,
                 train_frequency=0, num_rollouts=1,
                 gamma=1.0, exploration_steps=0, exploration_episodes=0, comment=''):
        super().__init__(env_name,
                         train_frequency=train_frequency, num_rollouts=num_rollouts,
                         gamma=gamma, exploration_steps=exploration_steps,
                         exploration_episodes=exploration_episodes, comment=comment)

        self.batch_size = batch_size
        self.memory = memory

        self.optimizer = optimizer
        self.target_update_frequency = target_update_frequency
        self.num_iter = num_iter

    def observe(self, observation):
        """See `AbstractAgent.observe'."""
        super().observe(observation)
        self.memory.append(observation)
        if (self._training and  # training mode.
                len(self.memory) >= self.batch_size and  # enough data.
                self.train_frequency > 0 and  # train after train_frequency transitions.
                self.total_steps % self.train_frequency == 0):  # correct steps.
            self._train()

    def end_episode(self):
        """See `AbstractAgent.end_episode'."""
        if (self._training and  # training mode.
                len(self.memory) > self.batch_size and  # enough data
                self.num_rollouts > 0 and  # train after num_rollouts transitions.
                self.total_episodes % self.num_rollouts == 0):  # correct steps.
            self._train()

        super().end_episode()

    def _train(self):
        """Train the off-policy agent."""
        self.algorithm.reset()
        for _ in range(self.num_iter):
            obs, idx, weight = self.memory.get_batch(self.batch_size)

            def closure():
                """Gradient calculation."""
                self.optimizer.zero_grad()
                losses = self.algorithm(obs.state, obs.action, obs.reward,
                                        obs.next_state, obs.done)
                loss = (losses.loss * weight.detach()).mean()
                loss.backward()
                return losses

            losses = self.optimizer.step(closure=closure)

            # Update memory
            self.memory.update(idx, losses.td_error.abs().detach())

            # Update logs
            self.logger.update(**average_named_tuple(losses)._asdict())
            self.logger.update(**self.algorithm.info())

            self.counters['train_steps'] += 1
            if self.train_steps % self.target_update_frequency == 0:
                self.algorithm.update()
                for param in self.params.values():
                    param.update()

        self.algorithm.reset()
