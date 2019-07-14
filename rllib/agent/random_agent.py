from .abstract_agent import AbstractAgent
from rllib.policy import RandomPolicy
from rllib.dataset import Dataset
import torch


class RandomAgent(AbstractAgent):
    def __init__(self, state_dim, action_dim, action_space,
                 batch_size=1, sequence_length=1, seed=0):
        super(RandomAgent, self).__init__()
        self._policy = RandomPolicy(action_space, state_dim)
        self._trajectory = []
        self._dataset = Dataset(state_dim, action_dim, batch_size,
                                sequence_length)

    def act(self, state):
        state = torch.from_numpy(state).float()
        action = self._policy.action(state).sample()
        return action.detach().numpy()

    def observe(self, observation):
        self._trajectory.append(observation)

    def start_episode(self):
        self._trajectory = []

    def end_episode(self):
        self._dataset.append(self._trajectory)

    def end_interaction(self):
        pass
