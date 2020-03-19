"""Abstract TD Learning file."""

from abc import ABCMeta, abstractmethod

import torch
import torch.testing

from rllib.util.utilities import tensor_to_distribution


class AbstractTDLearning(object, metaclass=ABCMeta):
    """Abstract Base Class for TD Learning algorithm family.

    V = theta * phi.
    td = r + gamma * V' - V.
    """

    double_sample = False

    def __init__(self, environment, policy, sampler, value_function, gamma,
                 lr_theta=0.1, lr_omega=0.1, exact_value_function=None):
        self.dimension = value_function.dimension
        self.omega = torch.zeros(self.dimension)
        self.theta = torch.zeros(self.dimension)
        self.value_function = value_function

        self.environment = environment
        self.policy = policy
        self.sampler = sampler
        self.gamma = gamma
        self.lr_theta = lr_theta
        self.lr_omega = lr_omega

        self.exact_value_function = exact_value_function

    def _step(self, state):
        try:
            self.environment.state = state.item()
        except ValueError:
            self.environment.state = state.numpy()

        action = tensor_to_distribution(self.policy(state)).sample()
        next_state, reward, done, _ = self.environment.step(action)
        next_state = torch.tensor(next_state)
        reward = torch.tensor(reward, dtype=torch.get_default_dtype())
        return next_state, reward, done

    def simulate(self, observation):
        """Run simulator in batch mode for a batch of observations."""
        if self.environment:
            batch_size = self.sampler.batch_size
            state = torch.zeros((batch_size, self.environment.dim_state))
            next_state = torch.zeros((batch_size, self.environment.dim_state))
            reward = torch.zeros((batch_size,))
            done = torch.zeros((batch_size,))

            for i in range(batch_size):
                s = observation.state[i]
                ns, r, d = self._step(s)
                state[i] = s
                next_state[i] = ns
                reward[i] = r
                done[i] = d
        else:
            state = observation.state
            next_state = observation.next_state
            reward = observation.reward
            done = observation.done
        return state, next_state, reward, done

    def train(self, epochs):
        """Train using TD Learning."""
        mspbe = []

        for _ in range(epochs):
            for i in range(len(self.sampler) // self.sampler.batch_size):
                observation, idx, weight = self.sampler.get_batch()

                state, next_state, reward, done = self.simulate(observation)

                # Get embeddings of value function.
                phi = self.value_function.embeddings(state)
                next_phi = self.value_function.embeddings(next_state)

                # TD
                td = reward + self.gamma * (next_phi @ self.theta) - phi @ self.theta
                mspbe.append(td.mean().item() ** 2)
                if self.double_sample:
                    aux_state, next_state, reward, done = self.simulate(observation)
                    torch.testing.assert_allclose(aux_state, state)
                    next_phi = self.value_function.embeddings(next_state)
                    td2 = reward + self.gamma * next_phi @ self.theta - phi @ self.theta
                    td = td * td2

                self._update(td, phi, next_phi, weight)
                self.sampler.update(idx, td.detach().numpy())

        self.value_function.value_function.head.weight.data = self.theta
        return mspbe

    @abstractmethod
    def _update(self, td_error, phi, next_phi, weight):
        raise NotImplementedError
