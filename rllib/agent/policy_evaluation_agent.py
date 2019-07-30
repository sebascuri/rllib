from .abstract_agent import AbstractAgent
from abc import abstractmethod
import torch
from torch.utils.data import DataLoader


class OnlinePolicyEvaluation(AbstractAgent):
    def __init__(self, policy, value_function, target_function, criterion, optimizer,
                 hyper_params):
        super().__init__()
        self._policy = policy
        self._value_function = value_function
        self._target_function = target_function
        self._target_function.eval()

        self._criterion = criterion
        self._hyper_params = hyper_params
        self._optimizer = optimizer(self._value_function.parameters,
                                    lr=self._hyper_params['learning_rate'])
        self._data_loader = DataLoader(self._memory,
                                       batch_size=self._hyper_params['batch_size'])

        self._trajectory = []

    def act(self, state):
        action_distribution = self._policy(torch.from_numpy(state).float())
        return action_distribution.sample().item()

    def observe(self, observation):
        super().observe(observation)
        self._trajectory.append(observation)

    def start_episode(self):
        super().start_episode()
        self._trajectory = []

    def end_episode(self):
        for observation in self._trajectory:
            pass

    @property
    def policy(self):
        return self._policy

    def end_interaction(self):
        pass

    # def _train(self):
    #     self._memory.shuffle()
    #     for epoch in range(self._hyper_params['epochs']):
    #         epoch_loss = 0
    #         for i, observation in enumerate(self._data_loader):
    #             state, action, reward, next_state, done = observation
    #             loss = self._td(state.float(), action.float(),
    #                             reward.unsqueeze(-1).float(),
    #                             next_state.float(), done.unsqueeze(-1).float())
    #             self._optimizer.zero_grad()
    #             loss.backward()
    #             self._optimizer.step()
    #             epoch_loss += loss.detach().item()
    #
    #         print(epoch_loss)

    @abstractmethod
    def _td(self, state, action, reward, next_state, done):
        raise NotImplementedError


class TDAgent(OnlinePolicyEvaluation):
    def __init__(self, policy, value_function, criterion, optimizer, memory,
                 hyper_params):
        super().__init__(policy, value_function, criterion, optimizer, memory,
                         hyper_params)

    def __str__(self):
        return 'TD-{}'.format(self._hyper_params['lambda'])

    def _td(self, state, action, reward, next_state, done):
        pred_v = self._value_function(state)
        next_v = self._value_function(next_state)
        target_v = reward + self._hyper_params['gamma'] * next_v * (1 - done)
        return self._criterion(pred_v, target_v.detach())
