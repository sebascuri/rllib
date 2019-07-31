from .abstract_agent import AbstractAgent
from abc import abstractmethod
import torch
from rllib.util import sum_discounted_rewards


class EpisodicPolicyEvaluation(AbstractAgent):
    def __init__(self, policy, value_function, criterion, optimizer, hyper_params):
        super().__init__()
        self._policy = policy
        self._value_function = value_function
        # self._target_function = target_function
        # self._target_function.eval()

        self._criterion = criterion
        self._hyper_params = hyper_params
        self._optimizer = optimizer(self._value_function.parameters,
                                    lr=self._hyper_params['learning_rate'])

        self._trajectory = []

    def act(self, state):
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

    @property
    def policy(self):
        return self._policy

    def end_interaction(self):
        pass

    def _train(self):
        for t, observation in enumerate(self._trajectory):
            expected_value = self._value_function(
                torch.tensor(observation.state).float())
            target_value = self._value_estimate(self._trajectory[t:])

            self._optimizer.zero_grad()
            loss = self._criterion(expected_value, target_value)
            loss.backward()
            self._optimizer.step()

    @abstractmethod
    def _value_estimate(self, trajectory):
        raise NotImplementedError


class TDAgent(EpisodicPolicyEvaluation):
    def __str__(self):
        return 'TD-{}'.format(self._hyper_params.get('lambda', 0))

    def _value_estimate(self, trajectory):
        state, action, reward, next_state, done = trajectory[0]
        return reward + self._hyper_params['gamma'] * self._value_function(
            torch.tensor(next_state).float()).detach() * (1.-float(done))


class MCAgent(EpisodicPolicyEvaluation):
    def __str__(self):
        return 'Monte Carlo Agent'

    def _value_estimate(self, trajectory):
        estimate = sum_discounted_rewards(trajectory, self._hyper_params['gamma'])
        return torch.tensor([estimate])
