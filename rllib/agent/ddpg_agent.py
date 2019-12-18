"""Implementation of DDPG Algorithms."""
from .abstract_agent import AbstractAgent
# from abc import abstractmethod
import torch
import numpy as np
import copy


class DDPGAgent(AbstractAgent):
    """Abstract Implementation of the DDPG Algorithm.

    The AbstractDDPGAgent algorithm implements the DDPT-Learning algorithm except for
    the computation of the TD-Error, which leads to different algorithms.

    Parameters
    ----------
    q_function: AbstractQFunction
        q_function that is learned.
    policy: AbstractPolicy
        policy that is learned.
    exploration: AbstractExplorationStrategy.
        exploration strategy that returns the actions.
    criterion: nn.Module
    critic_optimizer: nn.Optimizer
        q_function optimizer.
    actor_optimizer: nn.Optimizer
        policy optimizer.
    memory: ExperienceReplay
        memory where to store the observations.

    References
    ----------
    Lillicrap et. al. (2016). CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING. ICLR.

    """

    def __init__(self, q_function, policy, exploration, criterion, critic_optimizer,
                 actor_optimizer, memory, target_update_frequency=4, gamma=1.0,
                 episode_length=None):
        super().__init__(gamma=gamma, episode_length=episode_length)

        self._q_function = q_function
        self._q_target = copy.deepcopy(q_function)
        self._policy = policy
        self._policy_target = copy.deepcopy(policy)

        self._exploration = exploration
        self._criterion = criterion
        self._memory = memory
        self._target_update_frequency = target_update_frequency
        self._critic_optimizer = critic_optimizer
        self._actor_optimizer = actor_optimizer

        self.logs['td_errors'] = []
        self.logs['episode_td_errors'] = []

    def act(self, state):
        """See `AbstractAgent.act'."""
        action_distribution = self._policy(torch.tensor(state).float())
        return np.clip(self._exploration(action_distribution, self.total_steps), -1, 1)

    def observe(self, observation):
        """See `AbstractAgent.observe'."""
        super().observe(observation)
        self._memory.append(observation)
        if self._memory.has_batch:
            self._train()
            if self.total_steps % self._target_update_frequency == 0:
                self._q_target.parameters = self._q_function.parameters
                self._policy_target.parameters = self._policy.parameters

    def start_episode(self):
        """See `AbstractAgent.start_episode'."""
        super().start_episode()
        self.logs['episode_td_errors'].append([])

    def end_episode(self):
        """See `AbstractAgent.end_episode'."""
        aux = self.logs['episode_td_errors'].pop(-1)
        if len(aux) > 0:
            self.logs['episode_td_errors'].append(np.abs(np.array(aux)).mean())

    @property
    def policy(self):
        """See `AbstractAgent.policy'."""
        return self._policy

    def _train(self, batches=1):
        """Train the DDPG for `batches' batches.

        Parameters
        ----------
        batches: int

        """
        for batch in range(batches):
            state, action, reward, next_state, done = self._memory.get_batch()

            # optimize critic
            self._critic_optimizer.zero_grad()
            pred_q, target_q = self._td(state.float(), action.float(), reward.float(),
                                        next_state.float(), done.float())
            td_error = (pred_q.detach() - target_q.detach()).mean().item()
            self.logs['td_errors'].append(td_error)
            self.logs['episode_td_errors'][-1].append(td_error)

            loss = self._criterion(pred_q, target_q)
            loss.backward()
            self._critic_optimizer.step()

            # optimize actor
            self._actor_optimizer.zero_grad()
            policy_action = self._policy(state.float()).loc
            q = -(self._q_function(state.float(), policy_action)).mean()
            q.backward()
            self._actor_optimizer.step()

    def _td(self, state, action, reward, next_state, done):
        pred_q = self._q_function(state, action)

        # target = r + gamma * Q_target(x', \pi_target(x'))
        next_policy_action = self._policy_target(next_state).loc
        next_q = self._q_target(next_state, next_policy_action)
        target_q = reward + self.gamma * next_q * (1 - done)

        return pred_q, target_q.detach()