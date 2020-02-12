"""Implementation of SARSA Algorithms."""

from .abstract_agent import AbstractAgent
import torch
from torch.distributions import Categorical


class SarsaAgent(AbstractAgent):
    """Implementation of SARSA (On-Line)-Control."""

    def __init__(self, q_function, exploration, gamma=1.0):
        super().__init__(gamma=gamma)
        self.q_function = q_function
        self.exploration = exploration

    def act(self, state):
        """See `AbstractAgent.act'."""
        logits = self.q_function(torch.tensor(state).float())
        action_distribution = Categorical(logits=logits)
        return self.exploration(action_distribution, self.total_steps)

    def observe(self, observation):
        """See `AbstractAgent.observe'."""
        super().observe(observation)

    def start_episode(self):
        """See `AbstractAgent.start_episode'."""
        super().start_episode()
        self.logs['td_errors'].append([])

    def end_episode(self):
        """See `AbstractAgent.end_episode'."""
        self.logs['q_function'].append(self.q_function.state_dict())
