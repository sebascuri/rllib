from .abstract_agent import AbstractAgent
from abc import abstractmethod
import numpy as np
import torch


class AbstractPolicyGradient(AbstractAgent):
    def __init__(self, policy, optimizer, hyper_params):
        super().__init__()
        self._trajectory = []
        self._policy = policy
        self.hyper_params = hyper_params
        self._optimizer = optimizer(self._policy.parameters,
                                    lr=self.hyper_params['learning_rate'])

    def act(self, state):
        """Ask the agent for an action to interact with the environment.

        Parameters
        ----------
        state: ndarray

        Returns
        -------
        action: ndarray

        """
        action_distribution = self._policy(torch.from_numpy(state).float())
        return action_distribution.sample().item()

    def observe(self, observation):
        super().observe(observation)
        self._trajectory.append(observation)

    def start_episode(self):
        super().start_episode()
        self._trajectory = []

    def end_episode(self):
        self._train()

    def end_interaction(self):
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

    @property
    def policy(self):
        return self._policy

    @abstractmethod
    def _q_estimate(self, trajectory):
        raise NotImplementedError


class REINFORCE(AbstractPolicyGradient):
    def __str__(self):
        return "REINFORCE"

    def _q_estimate(self, trajectory):
        q_estimate = []
        for t in range(len(trajectory)):
            q_t = 0
            for i, observation in enumerate(trajectory[t:]):
                q_t = q_t + self.hyper_params['gamma'] ** i * observation.reward
            q_estimate.append(q_t)

        q_estimate = np.array(q_estimate)

        return (q_estimate - q_estimate.mean()) / q_estimate.std()


class ActorCritic(AbstractPolicyGradient):
    def __init__(self, actor, critic, optimizer, hyper_params):
        super().__init__(policy=actor, optimizer=optimizer, hyper_params=hyper_params)
        self._critic = critic
        self._critic_optimizer = optimizer(
            self._critic.parameters, lr=self.hyper_params['critic_learning_rate']
        )

    def __str__(self):
        return "Vanilla Actor Critic"

    def _q_estimate(self, trajectory):
        q_estimate = []
        for observation in trajectory:
            q_estimate.append(
                self._critic(torch.tensor(observation.state),
                             torch.tensor(observation.action)))
