"""Off Policy Agent."""

from .abstract_agent import AbstractAgent


class OffPolicyAgent(AbstractAgent):
    """Template for an on-policy algorithm."""

    def __init__(self, env_name,
                 memory, batch_size=64,
                 train_frequency=0, num_rollouts=1,
                 gamma=1.0, exploration_steps=0, exploration_episodes=0, comment=''):
        super().__init__(env_name,
                         train_frequency=train_frequency, num_rollouts=num_rollouts,
                         gamma=gamma, exploration_steps=exploration_steps,
                         exploration_episodes=exploration_episodes, comment=comment)

        self.batch_size = batch_size
        self.memory = memory

    def observe(self, observation):
        """See `AbstractAgent.observe'."""
        super().observe(observation)
        self.memory.append(observation)
        if (self._training and  # training mode.
                len(self.memory) > self.batch_size and  # enough data.
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
