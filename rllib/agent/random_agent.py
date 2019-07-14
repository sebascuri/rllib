from .abstract_agent import AbstractAgent
# from rllib.policy.nn_policy import NNPolicy
from rllib.dataset import Observation, Dataset
import torch


class RandomAgent(AbstractAgent):
    def __init__(self, config, seed=None):
        super(RandomAgent, self).__init__(config, seed=seed)
        state_dim = config.get('state_dim')
        action_dim = config.get('action_dim')
        action_space = config.get('action_space')
        discrete_action = config.get('discrete_action')
        batch_size = config.get('batch_size')
        sequence_length = config.get('sequence_length')

        self._policy = NNPolicy(state_dim, action_space, discrete_action)
        self._trajectory = []
        self._dataset = Dataset(state_dim, action_dim, batch_size,
                                sequence_length, seed=seed)

    def act(self, state):
        state = torch.from_numpy(state).float()
        action = self._policy.action(state)
        return action.detach().numpy()

    def observe(self, state, action, reward, next_state):
        observation = Observation(state, action, reward, next_state)
        self._trajectory.append(observation)

    def start_episode(self):
        self._trajectory = []

    def end_episode(self):
        self._dataset.append(self._trajectory)

    def end_interaction(self):
        pass
