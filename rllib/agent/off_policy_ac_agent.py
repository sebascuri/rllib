"""Off Policy Agent."""

from .abstract_agent import AbstractAgent
from rllib.algorithms.abstract_algorithm import AbstractAlgorithm


class OffPolicyAgent(AbstractAgent):
    """Template for an on-policy algorithm."""
    algorithm: AbstractAlgorithm

    def __init__(self, env_name,
                 memory, batch_size=32,
                 train_frequency=1,
                 target_update_frequency=1,
                 gamma=1.0, exploration_steps=0,
                 exploration_episodes=0, comment=''):
        super().__init__(env_name, gamma=gamma, exploration_steps=exploration_steps,
                         exploration_episodes=exploration_episodes, comment=comment)

        self.batch_size = batch_size
        self.memory = memory
        self.target_update_frequency = target_update_frequency
        self.train_frequency = train_frequency

    def observe(self, observation):
        """See `AbstractAgent.observe'."""
        super().observe(observation)
        self.memory.append(observation)
        if (self._training and
                len(self.memory) > self.batch_size and
                self.total_steps % self.train_frequency == 0):
            self._train()

        if self.total_steps % self.target_update_frequency == 0:
            self.algorithm.update()
