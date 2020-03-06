"""Implementation of REINFORCE Algorithms."""

from rllib.agent.abstract_agent import AbstractAgent
import torch
import copy
from rllib.dataset import Observation
from rllib.dataset.utilities import stack_list_of_tuples
from rllib.util import discount_cumsum
from rllib.util.logger import Logger


class REINFORCE(AbstractAgent):
    """Implementation of the REINFORCE algorithm.

    The REINFORCE algorithm computes the policy gradient using MC
    approximation for the returns (sum of discounted rewards).

    References
    ----------
    Williams, Ronald J. "Simple statistical gradient-following algorithms for
    connectionist reinforcement learning." Machine learning 8.3-4 (1992): 229-256.
    """

    eps = 1e-12

    def __init__(self, policy, policy_optimizer, baseline=None, baseline_optimizer=None,
                 criterion=None, num_rollouts=1, target_update_frequency=1, gamma=1.0,
                 exploration_steps=0, exploration_episodes=0):
        super().__init__(gamma=gamma, exploration_steps=exploration_steps,
                         exploration_episodes=exploration_episodes)
        self.trajectories = []
        self.policy = policy
        self.policy_target = copy.deepcopy(policy)
        self.policy_optimizer = policy_optimizer
        self.baseline = baseline
        self.baseline_optimizer = baseline_optimizer
        self.criterion = criterion(reduction='none')
        self.num_rollouts = num_rollouts
        self.target_update_freq = target_update_frequency

        self.logs['losses'] = Logger('mean')

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
            self.policy_target.update_parameters(self.policy.parameters())

        super().end_episode()

    def _train(self):
        trajectories = [Observation(*stack_list_of_tuples(t))
                        for t in self.trajectories]

        value_estimates = self._value_estimate(trajectories)
        self._train_actor(trajectories, value_estimates)

        if self.baseline:
            self._train_baseline(trajectories, value_estimates)

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
            loss = self.criterion(self.baseline(observation.state), value_estimate)
            loss.mean().backward()

        self.baseline_optimizer.step()

    def _value_estimate(self, trajectories):
        values = []
        for trajectory in trajectories:
            val = discount_cumsum(trajectory.reward, self.gamma)
            values.append((val - val.mean()) / (val.std() + self.eps))

        return values
