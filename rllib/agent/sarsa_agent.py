"""Implementation of SARSA Algorithms."""

from .abstract_agent import AbstractAgent
import torch
from torch.distributions import Categorical


class SarsaAgent(AbstractAgent):
    """Implementation of SARSA (On-Line)-Control."""

    def __init__(self, q_function, exploration, gamma=1.0, episode_length=None):
        super().__init__(gamma=gamma, episode_length=episode_length)
        self._q_function = q_function
        self._exploration = exploration

    def act(self, state):
        """See `AbstractAgent.act'."""
        logits = self._q_function(torch.tensor(state).float())
        action_distribution = Categorical(logits=logits)
        return self._exploration(action_distribution, self.total_steps)

    def observe(self, observation):
        """See `AbstractAgent.observe'."""
        super().observe(observation)

    def start_episode(self):
        """See `AbstractAgent.start_episode'."""
        super().start_episode()
        self.logs['td_errors'].append([])

    def end_episode(self):
        """See `AbstractAgent.end_episode'."""
        self.logs['q_function'].append(self._q_function.state_dict())

    @property
    def policy(self):
        """See `AbstractAgent.policy'."""
        return self._q_function.extract_policy(temperature=0.001)
