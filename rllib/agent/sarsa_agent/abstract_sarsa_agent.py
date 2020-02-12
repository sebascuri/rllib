"""Implementation of SARSA Algorithms."""

from rllib.agent.abstract_agent import AbstractAgent
from abc import abstractmethod
import copy
import numpy as np
import torch


class AbstractSARSAAgent(AbstractAgent):
    """Abstract base class for SARSA (On-Line)-Control."""

    def __init__(self, q_function, policy, criterion, optimizer,
                 target_update_frequency=1, gamma=1.0):
        super().__init__(gamma=gamma)
        self.q_function = q_function
        self.policy = policy
        self.q_target = copy.deepcopy(q_function)

        self.target_update_frequency = target_update_frequency
        self.criterion = criterion
        self.optimizer = optimizer
        self._last_observation = None

        self.logs['td_errors'] = []
        self.logs['episode_td_errors'] = []

    def act(self, state):
        """See `AbstractAgent.act'."""
        action = super().act(state)
        if self._last_observation is not None:
            self._train(self._last_observation, action)
        if self.total_steps % self.target_update_frequency == 0:
            self.q_target.parameters = self.q_function.parameters
        return action

    def observe(self, observation):
        """See `AbstractAgent.observe'."""
        super().observe(observation)
        self._last_observation = observation

    def start_episode(self):
        """See `AbstractAgent.start_episode'."""
        super().start_episode()
        self.logs['episode_td_errors'].append([])
        self._last_observation = None

    def end_episode(self):
        """See `AbstractAgent.end_episode'."""
        # The next action is irrelevant as the next value is zero for all actions.
        next_action = super().act(self._last_observation.state)
        self._train(self._last_observation, next_action)

        aux = self.logs['episode_td_errors'].pop(-1)
        if len(aux) > 0:
            self.logs['episode_td_errors'].append(np.abs(np.array(aux)).mean())

    def _train(self, observation, next_action):
        self.optimizer.zero_grad()

        pred_q, target_q = self._td(
            *map(lambda x: torch.tensor(x).float(), observation),
            torch.tensor(next_action).float())

        td_error = pred_q.detach() - target_q.detach()
        td_error_mean = td_error.mean().item()
        self.logs['td_errors'].append(td_error_mean)
        self.logs['episode_td_errors'][-1].append(td_error_mean)
        loss = self.criterion(pred_q, target_q, reduction='mean')
        loss.backward()

        self.optimizer.step()

    @abstractmethod
    def _td(self, state, action, reward, next_state, done, next_action):
        raise NotImplementedError
