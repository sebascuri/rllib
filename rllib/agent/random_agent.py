"""Implementation of a random agent."""

from .abstract_agent import AbstractAgent
from rllib.dataset import TrajectoryDataset
from rllib.policy import RandomPolicy
import torch


class RandomAgent(AbstractAgent):
    """Agent that interacts randomly in an environment."""

    def __init__(self, dim_state, dim_action, num_actions=None, gamma=None,
                 episode_length=None):
        super().__init__(gamma=gamma, episode_length=episode_length)
        self._policy = RandomPolicy(dim_state, dim_action, num_actions=num_actions)
        self._trajectory = []
        self._dataset = TrajectoryDataset(sequence_length=1)

    def act(self, state):
        state = torch.tensor(state)
        action = self._policy(state).sample()
        return action.detach().numpy()

    def observe(self, observation):
        super().observe(observation)
        self._trajectory.append(observation)

    def start_episode(self):
        super().start_episode()
        self._trajectory = []

    def end_episode(self):
        self._dataset.append(self._trajectory)

    @property
    def policy(self):
        return self._policy
