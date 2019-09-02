"""Implementation of Policy Evaluation Algorithms."""


from .abstract_agent import AbstractAgent
from abc import abstractmethod
import torch
from rllib.util import sum_discounted_rewards


__all__ = ['TDAgent', 'MCAgent', 'OnLineTDLearning']


class EpisodicPolicyEvaluation(AbstractAgent):
    """Abstract Implementation of Policy-Evaluation Algorithms.

    The EpisodicPolicyEvaluation algorithm implements the Policy-Evaluation algorithm
    except for the computation of the estimate of the Value Function, which leads to
    Temporal Difference or Monte Carlo algorithms.

    Parameters
    ----------
    policy: AbstractPolicy
        Policy used to interact with the environment (fixed).
    value_function: AbstractValueFunction
        Value function to be learned.
    criterion: nn.Module
    optimizer: nn.optim
    hyper_params: dict
        algorithm hyperparameters
    """

    def __init__(self, policy, value_function, criterion, optimizer, hyper_params,
                 gamma=1.0, episode_length=None):
        super().__init__(gamma=gamma, episode_length=episode_length)
        self._policy = policy
        self._value_function = value_function

        self._criterion = criterion
        self._hyper_params = hyper_params
        self._optimizer = optimizer(self._value_function.parameters,
                                    lr=self._hyper_params['learning_rate'])

        self._trajectory = []
        self.logs['value_function'] = []
        self.logs['td_error'] = []

    def act(self, state):
        action_distribution = self._policy(torch.tensor(state))
        return action_distribution.sample().item()

    def observe(self, observation):
        super().observe(observation)
        self._trajectory.append(observation)

    def start_episode(self):
        super().start_episode()
        self._trajectory = []
        self.logs['td_error'].append(0)

    def end_episode(self):
        self._train()
        self.logs['value_function'].append(
            [param.detach().clone() for param in self._value_function.parameters])

    @property
    def policy(self):
        return self._policy

    def end_interaction(self):
        pass

    def _train(self):
        for t, observation in enumerate(self._trajectory):
            expected_value = self._value_function(
                torch.tensor(observation.state))
            target_value = self._value_estimate(self._trajectory[t:])

            loss = self._criterion(expected_value, target_value)

            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()
            self.logs['td_error'][-1] += loss.item()

    @abstractmethod
    def _value_estimate(self, trajectory):
        raise NotImplementedError


class TDAgent(EpisodicPolicyEvaluation):
    """Implementation of TD algorithm."""

    def _value_estimate(self, trajectory):
        state, action, reward, next_state, done = trajectory[0]
        return reward + self.gamma * self._value_function(
            torch.tensor(next_state)).detach()


class MCAgent(EpisodicPolicyEvaluation):
    """Implementation of Monte-Carlo algorithm."""

    def _value_estimate(self, trajectory):
        last_state = torch.tensor(trajectory[-1].next_state)
        estimate = sum_discounted_rewards(trajectory, self.gamma)
        estimate += (self.gamma ** len(trajectory)
                     * self._value_function(last_state).detach())

        return estimate


class OnLineTDLearning(AbstractAgent):
    """Implementation of online-TD(lambda)-Learning."""
    def __init__(self, policy, value_function, criterion, optimizer, hyper_params,
                 lamda_=0.0, gamma=1.0, episode_length=None):
        super().__init__(gamma=gamma, episode_length=episode_length)
        self._policy = policy
        self._value_function = value_function

        self._criterion = criterion
        self._hyper_params = hyper_params
        self._optimizer = optimizer(self._value_function.parameters,
                                    lr=self._hyper_params['learning_rate'])
        self._lambda = lamda_

        self.logs['value_function'] = []
        self.logs['td_error'] = []

        self._eligibility_trace = []
        for param in self._value_function.parameters:
            self._eligibility_trace.append(torch.zeros_like(param))

    def act(self, state):
        action_distribution = self._policy(torch.tensor(state))
        return action_distribution.sample().item()

    def observe(self, observation):
        super().observe(observation)
        self._train(observation)

    def start_episode(self):
        super().start_episode()
        self.logs['td_error'].append(0)

    def end_episode(self):
        self.logs['value_function'].append(
            [param.detach().clone() for param in self._value_function.parameters])

    @property
    def policy(self):
        return self._policy

    def end_interaction(self):
        pass

    def _train(self, observation):
        value_ = self._value_function(torch.tensor(observation.state))
        next_value = self._value_function(torch.tensor(observation.next_state))
        target = observation.reward + self.gamma * next_value.detach()

        self._optimizer.zero_grad()
        loss = self._criterion(value_, target)
        self.logs['td_error'][-1] += loss.item()
        loss.backward()

        for i, param in enumerate(self._value_function.parameters):
            self._eligibility_trace[i] *= self.gamma * self._lambda
            self._eligibility_trace[i] += param.grad
            param.grad.data.set_ = self._eligibility_trace[i]
        self._optimizer.step()
