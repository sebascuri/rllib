"""Implementation of QLearning Algorithms."""
from rllib.agent.abstract_agent import AbstractAgent
from abc import abstractmethod
import torch
import numpy as np
import copy


class AbstractQLearningAgent(AbstractAgent):
    """Abstract Implementation of the Q-Learning Algorithm.

    The AbstractQLearning algorithm implements the Q-Learning algorithm except for the
    computation of the TD-Error, which leads to different algorithms.

    Parameters
    ----------
    q_function: AbstractQFunction
        q_function that is learned.
    policy: QFunctionPolicy.
        Q-function derived policy.
    criterion: nn.Module
    optimizer: nn.optim
    memory: ExperienceReplay
        memory where to store the observations.

    References
    ----------
    Watkins, C. J., & Dayan, P. (1992). Q-learning. Machine learning, 8(3-4), 279-292.

    """

    def __init__(self, q_function, policy, criterion, optimizer, memory,
                 target_update_frequency=4, gamma=1.0):
        super().__init__(gamma=gamma)
        self.q_function = q_function
        self.policy = policy
        self.q_target = copy.deepcopy(q_function)
        self.criterion = criterion
        self.memory = memory
        self.target_update_frequency = target_update_frequency
        self.optimizer = optimizer

        # self.logs['q_function'] = []
        self.logs['td_errors'] = []
        self.logs['episode_td_errors'] = []

    def observe(self, observation):
        """See `AbstractAgent.observe'."""
        super().observe(observation)
        self.memory.append(observation)
        if self.memory.has_batch:
            self._train()
            if self.total_steps % self.target_update_frequency == 0:
                self.q_target.parameters = self.q_function.parameters

    def start_episode(self):
        """See `AbstractAgent.start_episode'."""
        super().start_episode()
        self.logs['episode_td_errors'].append([])

    def end_episode(self):
        """See `AbstractAgent.end_episode'."""
        aux = self.logs['episode_td_errors'].pop(-1)
        if len(aux) > 0:
            self.logs['episode_td_errors'].append(np.abs(np.array(aux)).mean())

    def _train(self, batches=1):
        """Train the DQN for `batches' batches.

        Parameters
        ----------
        batches: int

        """
        for batch in range(batches):
            (state, action, reward, next_state, done), idx, w = self.memory.get_batch()
            self.optimizer.zero_grad()
            pred_q, target_q = self._td(state.float(), action.float(), reward.float(),
                                        next_state.float(), done.float())

            td_error = pred_q.detach() - target_q.detach()
            td_error_mean = td_error.mean().item()
            self.logs['td_errors'].append(td_error_mean)
            self.logs['episode_td_errors'][-1].append(td_error_mean)
            loss = self.criterion(pred_q, target_q, reduction='none')
            loss = torch.tensor(w).float() * loss
            loss.mean().backward()

            self.optimizer.step()
            self.memory.update(idx, td_error)

    @abstractmethod
    def _td(self, state, action, reward, next_state, done):
        raise NotImplementedError
