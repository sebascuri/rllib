"""On Policy Agent."""

from .abstract_agent import AbstractAgent
from rllib.algorithms.abstract_algorithm import AbstractAlgorithm


class OnPolicyAgent(AbstractAgent):
    """Template for an on-policy algorithm."""

    algorithm: AbstractAlgorithm

    def __init__(self, env_name,
                 num_rollouts=1,
                 gamma=1.0, exploration_steps=0,
                 exploration_episodes=0, comment=''):
        super().__init__(env_name, gamma=gamma, exploration_steps=exploration_steps,
                         exploration_episodes=exploration_episodes, comment=comment)
        self.trajectories = []
        self.num_rollouts = num_rollouts

    def observe(self, observation):
        """See `AbstractAgent.observe'."""
        super().observe(observation)
        self.trajectories[-1].append(observation)

    def start_episode(self, **kwargs):
        """See `AbstractAgent.start_episode'."""
        super().start_episode(**kwargs)
        self.trajectories.append([])

    def end_episode(self):
        """See `AbstractAgent.end_episode'."""
        if self.total_episodes % self.num_rollouts == self.num_rollouts - 1:
            if self._training:
                self._train()
            self.trajectories = list()

        super().end_episode()
