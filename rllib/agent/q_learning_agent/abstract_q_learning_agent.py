"""Implementation of QLearning Algorithms."""
from rllib.agent.abstract_agent import AbstractAgent
from rllib.dataset import Observation
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
                 target_update_frequency=4, gamma=1.0,
                 exploration_steps=0, exploration_episodes=0):
        super().__init__(gamma=gamma, exploration_steps=exploration_steps,
                         exploration_episodes=exploration_episodes)
        self.q_function = q_function
        self.policy = policy
        self.q_target = copy.deepcopy(q_function)
        self.criterion = criterion(reduction='none')
        self.memory = memory
        self.target_update_frequency = target_update_frequency
        self.optimizer = optimizer

        self.logs['td_errors'] = []
        self.logs['episode_td_errors'] = []

    def observe(self, observation):
        """See `AbstractAgent.observe'."""
        super().observe(observation)
        self.memory.append(observation)
        if self.memory.has_batch:
            self._train()
            if self.total_steps % self.target_update_frequency == 0:
                self.q_target.update_parameters(self.q_function.parameters())

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
            observation, idx, weight = self.memory.get_batch()
            weight = torch.tensor(weight).float()
            observation = Observation(*map(lambda x: x.float(), observation))

            self.optimizer.zero_grad()
            pred_q, target_q = self._td(*observation)

            td_error = pred_q.detach() - target_q.detach()
            td_error_mean = td_error.mean().item()
            self.logs['td_errors'].append(td_error_mean)
            self.logs['episode_td_errors'][-1].append(td_error_mean)
            loss = weight * self.criterion(pred_q, target_q)
            loss.mean().backward()

            self.optimizer.step()
            self.memory.update(idx, td_error.numpy())

    @abstractmethod
    def _td(self, state, action, reward, next_state, done):
        raise NotImplementedError
