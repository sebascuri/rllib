"""Implementation of Model-Free Policy Gradient Algorithms."""

from .abstract_agent import AbstractAgent
from abc import abstractmethod
import numpy as np
import torch


__all__ = ['REINFORCE', 'ActorCritic']


class AbstractPolicyGradient(AbstractAgent):
    """Abstract Implementation of the Policy-Gradient Algorithm.

    The AbstractPolicyGradient algorithm implements the Policy-Gradient algorithm except
    for the computation of the rewards, which leads to different algorithms.

    Parameters
    ----------
    policy: AbstractPolicy
        learnable policy.
    optimizer: nn.optim
    hyper_params:
        algorithm hyperparameters.

    References
    ----------
    Williams, Ronald J. "Simple statistical gradient-following algorithms for
    connectionist reinforcement learning." Machine learning 8.3-4 (1992): 229-256.
    """

    def __init__(self, policy, optimizer, hyper_params, gamma=1.0, episode_length=None):
        super().__init__(gamma=gamma, episode_length=episode_length)
        self._trajectory = []
        self.policy = policy
        self.hyper_params = hyper_params
        self._optimizer = optimizer(self._policy.parameters,
                                    lr=self.hyper_params['learning_rate'])

    def act(self, state):
        """See `AbstractAgent.act'."""
        action_distribution = self._policy(torch.from_numpy(state).float())
        return action_distribution.sample().item()

    def observe(self, observation):
        """See `AbstractAgent.observe'."""
        super().observe(observation)
        self._trajectory.append(observation)

    def start_episode(self):
        """See `AbstractAgent.start_episode'."""
        super().start_episode()
        self._trajectory = []

    def end_episode(self):
        """See `AbstractAgent.end_episode'."""
        self._train()

    def end_interaction(self):
        """See `AbstractAgent.end_interaction'."""
        pass

    def _train(self):
        q_estimate = self._q_estimate(self._trajectory)

        loss = 0
        self._optimizer.zero_grad()
        for i, observation in enumerate(self._trajectory):
            state = observation.state
            action = observation.action
            pi = self._policy(torch.from_numpy(state).float())
            loss -= pi.log_prob(torch.tensor(action)) * q_estimate[i]
        loss.backward()
        self._optimizer.step()

    @abstractmethod
    def _q_estimate(self, trajectory):
        raise NotImplementedError


class REINFORCE(AbstractPolicyGradient):
    """Implementation of REINFORCE algorithm.

    References
    ----------
    Williams, Ronald J. "Simple statistical gradient-following algorithms for
    connectionist reinforcement learning." Machine learning 8.3-4 (1992): 229-256.

    """

    def _q_estimate(self, trajectory):
        q_estimate = []
        for t in range(len(trajectory)):
            q_t = 0
            for i, observation in enumerate(trajectory[t:]):
                q_t = q_t + self.gamma ** i * observation.reward
            q_estimate.append(q_t)

        q_estimate = np.array(q_estimate)

        return (q_estimate - q_estimate.mean()) / q_estimate.std()


class ActorCritic(AbstractPolicyGradient):
    """Implementation of ACTOR-CRITIC algorithm.

    References
    ----------
    Sutton, Richard S., et al. "Policy gradient methods for reinforcement learning with
    function approximation." Advances in neural information processing systems. 2000.

    """

    def __init__(self, actor, critic, optimizer, hyper_params):
        super().__init__(policy=actor, optimizer=optimizer, hyper_params=hyper_params)
        self._critic = critic
        self._critic_optimizer = optimizer(
            self._critic.parameters, lr=self.hyper_params['critic_learning_rate']
        )

    def _q_estimate(self, trajectory):
        q_estimate = []
        for observation in trajectory:
            q_estimate.append(
                self._critic(torch.tensor(observation.state),
                             torch.tensor(observation.action)))
