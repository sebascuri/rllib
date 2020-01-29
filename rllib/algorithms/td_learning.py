"""Implementation of LSTD algorithm."""

from abc import ABC, abstractmethod
import torch
from torch.distributions import Bernoulli
import torch.testing


class TDLearning(ABC):
    """Abstract Base Class for TD Learning algorithm family."""

    double_sample = False

    def __init__(self, environment, agent, sampler, value_function, gamma,
                 lr_theta=0.1, lr_omega=0.1):
        self.dimension = value_function.dimension
        self.omega = torch.zeros(self.dimension)
        self.theta = torch.zeros(self.dimension)
        self.value_function = value_function

        self.environment = environment
        self.agent = agent
        self.sampler = sampler
        self.gamma = gamma
        self.lr_theta = lr_theta
        self.lr_omega = lr_omega

    def _step(self, state):
        try:
            self.environment.state = state.item()
        except ValueError:
            self.environment.state = state.numpy()

        action = self.agent.act(state)
        next_state, reward, done, _ = self.environment.step(action)
        next_state = torch.tensor(next_state)
        reward = torch.tensor(reward).float()
        return next_state, reward

    def simulate(self, observation):
        """Run simulator in batch mode for a batch of observations."""
        if self.environment:
            batch_size = self.sampler.batch_size
            state = torch.zeros((batch_size, self.environment.dim_state))
            next_state = torch.zeros((batch_size, self.environment.dim_state))
            reward = torch.zeros((batch_size, ))

            for i in range(batch_size):
                s = observation.state[i]
                ns, r = self._step(s)
                state[i] = s
                next_state[i] = ns
                reward[i] = r
        else:
            state = observation.state
            next_state = observation.next_state
            reward = observation.reward
        return state, next_state, reward

    def train(self, epochs):
        """Train using TD Learning."""
        mspbe = []

        for _ in range(epochs):
            for i in range(len(self.sampler) // self.sampler.batch_size):
                observation, idx, weight = self.sampler.get_batch()

                state, next_state, reward = self.simulate(observation)

                # Get embeddings of value function.
                phi = self.value_function.embeddings(state)
                next_phi = self.value_function.embeddings(next_state)

                # TD
                td = reward + self.gamma * next_phi @ self.theta - phi @ self.theta
                mspbe.append(td.mean().item() ** 2)
                if self.double_sample:
                    aux_state, next_state, reward = self.simulate(observation)
                    torch.testing.assert_allclose(aux_state, state)
                    next_phi = self.value_function.embeddings(next_state)
                    td2 = reward + self.gamma * next_phi @ self.theta - phi @ self.theta
                    td = td * td2

                self._update(td, phi, next_phi, weight)
                self.sampler.update(idx, td.detach().numpy())
        return mspbe

    @abstractmethod
    def _update(self, td_error, phi, next_phi, weight):
        raise NotImplementedError


class TD(TDLearning):
    """TD Learning algorithm."""

    def _update(self, td_error, phi, next_phi, weight):
        self.theta += self.lr_theta * td_error @ phi


class GTD(TDLearning):
    """GTD Learning algorithm."""

    def _update(self, td_error, phi, next_phi, weight):
        phitw = phi @ self.omega

        self.theta += self.lr_theta * phitw @ (phi - self.gamma * next_phi)
        self.omega += self.lr_omega * (td_error @ phi - self.omega)


class GTD2(TDLearning):
    """GTD2 Learning algorithm."""

    def _update(self, td_error, phi, next_phi, weight):
        phitw = phi @ self.omega

        self.theta += self.lr_theta * phitw @ (phi - self.gamma * next_phi)
        self.omega += self.lr_omega * (td_error - phitw) @ phi


class TDC(TDLearning):
    """TDC Learning algorithm."""

    def _update(self, td_error, phi, next_phi, weight):
        phitw = phi @ self.omega

        self.theta += self.lr_theta * td_error @ phi
        self.theta -= self.lr_theta * self.gamma * phitw @ next_phi  # Correction term.
        self.omega += self.lr_omega * (td_error - phitw) @ phi


class TDLinf(TDLearning):
    """TD-Linf Learning algorithm."""

    double_sample = True

    def _update(self, td_error, phi, next_phi, weight):
        self.theta += self.lr_theta * td_error @ phi
        self.theta -= self.lr_theta * self.gamma * td_error @ next_phi


class TDL1(TDLearning):
    """TD-L1 Learning algorithm."""

    def _update(self, td_error, phi, next_phi, weight):
        weight_minus = 1 / weight
        s = Bernoulli(torch.tensor(weight / (weight_minus + weight))).sample().float()
        self.theta += self.lr_theta * s @ (phi - self.gamma * next_phi)
