"""Implementation of Model-Free Policy Gradient Algorithms."""

from rllib.agent.abstract_agent import AbstractAgent
from abc import abstractmethod
import torch
import copy
from rllib.dataset import Observation
from rllib.dataset.utilities import stack_list_of_tuples


class AbstractPolicyGradient(AbstractAgent):
    """Abstract Implementation of the Policy-Gradient Algorithm.

    The AbstractPolicyGradient algorithm implements the Policy-Gradient algorithm except
    for the computation of the rewards, which leads to different algorithms.

    Parameters
    ----------
    policy: AbstractPolicy
        learnable policy.

    References
    ----------
    Williams, Ronald J. "Simple statistical gradient-following algorithms for
    connectionist reinforcement learning." Machine learning 8.3-4 (1992): 229-256.
    """

    eps = 1e-12

    def __init__(self, policy, policy_optimizer, baseline=None, critic=None,
                 baseline_optimizer=None, critic_optimizer=None, criterion=None,
                 num_rollouts=1, target_update_frequency=1,
                 gamma=1.0, exploration_steps=0, exploration_episodes=0):
        super().__init__(gamma=gamma, exploration_steps=exploration_steps,
                         exploration_episodes=exploration_episodes)
        self.trajectories = []
        self.policy = policy
        self.policy_target = copy.deepcopy(policy)
        self.policy_optimizer = policy_optimizer
        self.baseline = baseline
        self.baseline_optimizer = baseline_optimizer
        self.critic = critic
        self.critic_target = copy.deepcopy(critic)
        self.critic_optimizer = critic_optimizer
        self.criterion = criterion
        self.num_rollouts = num_rollouts
        self.target_update_freq = target_update_frequency

    def observe(self, observation):
        """See `AbstractAgent.observe'."""
        super().observe(observation)
        self.trajectories[-1].append(observation)

    def start_episode(self):
        """See `AbstractAgent.start_episode'."""
        super().start_episode()
        self.trajectories.append([])

    def end_episode(self):
        """See `AbstractAgent.end_episode'."""
        if self.total_episodes % self.num_rollouts == 0:
            self._train()
            self.trajectories = []

        if self.total_episodes % (self.target_update_freq * self.num_rollouts) == 0:
            self.policy_target.parameters = self.policy.parameters

            if self.critic:
                self.critic_target.parameters = self.critic.parameters

    def _train(self):
        trajectories = [Observation(*stack_list_of_tuples(t))
                        for t in self.trajectories]

        value_estimates = self._value_estimate(trajectories)
        self._train_actor(trajectories, value_estimates)

        # This could be trained from an off-line dataset ?.
        if self.baseline:
            self._train_baseline(trajectories, value_estimates)
        if self.critic:
            self._train_critic(trajectories)

    def _train_actor(self, observations, value_estimates):
        self.policy_optimizer.zero_grad()
        for observation, value_estimate in zip(observations, value_estimates):
            if self.baseline is not None:
                baseline = self.baseline(observation.state).detach()
            else:
                baseline = torch.zeros_like(observation.reward)

            pi = self.policy(observation.state)
            action = observation.action
            if self.policy.discrete_action:
                action = observation.action.long()
            loss = - pi.log_prob(action) * (value_estimate - baseline)

            loss.sum().backward()
        self.policy_optimizer.step()

    def _train_baseline(self, observations, value_estimates):
        self.baseline_optimizer.zero_grad()

        for observation, value_estimate in zip(observations, value_estimates):
            pred_v, target_v = self._td_base(*observation, value_estimate)
            loss = self.criterion(pred_v, target_v, reduction='none')
            loss.mean().backward()

        self.baseline_optimizer.step()

    def _train_critic(self, observations):
        self.critic_optimizer.zero_grad()

        for observation in observations:
            pred_q, target_q = self._tdq(*observation)
            loss = self.criterion(pred_q, target_q, reduction='none')
            loss.mean().backward()
        self.critic_optimizer.step()

    @abstractmethod
    def _value_estimate(self, trajectories):
        raise NotImplementedError

    @abstractmethod
    def _td_base(self, state, action, reward, next_state, done, value_estimates=None):
        raise NotImplementedError

    @abstractmethod
    def _td_critic(self, state, action, reward, next_state, done):
        raise NotImplementedError
