"""Implementation of SARSA Algorithms."""

from rllib.agent.abstract_agent import AbstractAgent
from rllib.dataset import SARSAObservation
from rllib.dataset.utilities import stack_list_of_tuples
from abc import abstractmethod
import copy
import numpy as np


class AbstractSARSAAgent(AbstractAgent):
    """Abstract base class for SARSA (On-Line)-Control."""

    def __init__(self, q_function, policy, criterion, optimizer, batch_size=1,
                 target_update_frequency=1, gamma=1.0, exploration_steps=0,
                 exploration_episodes=0):
        super().__init__(gamma=gamma, exploration_steps=exploration_steps,
                         exploration_episodes=exploration_episodes)
        self.q_function = q_function
        self.policy = policy
        self.q_target = copy.deepcopy(q_function)

        self.target_update_frequency = target_update_frequency
        self.criterion = criterion(reduction='none')
        self.optimizer = optimizer
        self._last_observation = None
        self._batch_size = batch_size
        self._trajectory = list()

        self.logs['td_errors'] = []
        self.logs['episode_td_errors'] = []

    def act(self, state):
        """See `AbstractAgent.act'."""
        action = super().act(state)
        if self._last_observation:
            self._trajectory.append(SARSAObservation(*self._last_observation, action))
        return action

    def observe(self, observation):
        """See `AbstractAgent.observe'."""
        super().observe(observation)
        self._last_observation = observation

        if len(self._trajectory) >= self._batch_size:
            self._train(self._trajectory)
            self._trajectory = list()
        if self.total_steps % self.target_update_frequency == 0:
            self.q_target.parameters = self.q_function.parameters

    def start_episode(self):
        """See `AbstractAgent.start_episode'."""
        super().start_episode()
        self.logs['episode_td_errors'].append([])
        self._last_observation = None

    def end_episode(self):
        """See `AbstractAgent.end_episode'."""
        # The next action is irrelevant as the next value is zero for all actions.
        next_action = super().act(self._last_observation.state)
        self._trajectory.append(SARSAObservation(*self._last_observation, next_action))
        self._train(self._trajectory)

        aux = self.logs['episode_td_errors'].pop(-1)
        if len(aux) > 0:
            self.logs['episode_td_errors'].append(np.abs(np.array(aux)).mean())

    def _train(self, trajectory):
        trajectory = SARSAObservation(*stack_list_of_tuples(trajectory))

        self.optimizer.zero_grad()
        pred_q, target_q = self._td(*trajectory)

        td_error = pred_q.detach() - target_q.detach()
        td_error_mean = td_error.mean().item()
        self.logs['td_errors'].append(td_error_mean)
        self.logs['episode_td_errors'][-1].append(td_error_mean)
        loss = self.criterion(pred_q, target_q).mean()
        loss.backward()

        self.optimizer.step()

    @abstractmethod
    def _td(self, state, action, reward, next_state, done, next_action):
        raise NotImplementedError
