"""Policies parametrized with Neural Networks."""

import torch
import torch.nn as nn
import torch.nn.functional

from rllib.util.neural_networks import HeteroGaussianNN, CategoricalNN
from rllib.util.neural_networks import one_hot_encode
from .abstract_policy import AbstractPolicy


class NNPolicy(AbstractPolicy):
    """Implementation of a Policy implemented with a Neural Network.

    Parameters
    ----------
    dim_state: int
        dimension of state.
    dim_action: int
        dimension of action.
    num_states: int, optional
        number of discrete states (None if state is continuous).
    num_actions: int, optional
        number of discrete actions (None if action is continuous).
    layers: list, optional
        width of layers, each layer is connected with ReLUs non-linearities.
    biased_head: bool, optional
        flag that indicates if head of NN has a bias term or not.

    """

    def __init__(self, dim_state, dim_action, num_states=None, num_actions=None,
                 layers=None, biased_head=True, tau=1.0, deterministic=False,
                 squashed_output=True, input_transform=None):
        super().__init__(dim_state, dim_action, num_states, num_actions, tau,
                         deterministic)
        if self.discrete_state:
            in_dim = self.num_states
        else:
            in_dim = self.dim_state

        self.input_transform = input_transform
        if self.discrete_action:
            self.nn = CategoricalNN(in_dim, self.num_actions, layers,
                                    biased_head=biased_head)
        else:
            self.nn = HeteroGaussianNN(in_dim, self.dim_action, layers,
                                       biased_head=biased_head,
                                       squashed_output=squashed_output)

    def forward(self, state):
        """Get distribution over actions."""
        if self.input_transform is not None:
            state = self.input_transform(state)

        if isinstance(self.num_states, int):
            state = one_hot_encode(state.long(), torch.tensor(self.num_states))
        out = self.nn(state)

        if self.deterministic:
            return out[0], torch.zeros(1)
        else:
            return out

    def embeddings(self, state):
        """Get embeddings of the value-function at a given state."""
        if self.discrete_state:
            state = one_hot_encode(state.long(), self.num_states)

        features = self.nn.last_layer_embeddings(state)
        return features.squeeze()


class TabularPolicy(NNPolicy):
    """Implement tabular policy."""

    def __init__(self, num_states, num_actions):
        super().__init__(dim_state=1, dim_action=1,
                         num_states=num_states, num_actions=num_actions,
                         biased_head=False)
        nn.init.ones_(self.nn.head.weight)

    @property
    def table(self):
        """Get table representation of policy."""
        return self.nn.head.weight

    def set_value(self, state, new_value):
        """Set value to value function at a given state."""
        try:
            new_value = torch.log(
                one_hot_encode(new_value, num_classes=self.num_actions) + 1e-12
            )
        except TypeError:
            pass
        self.nn.head.weight[:, state] = new_value


class FelixPolicy(AbstractPolicy):
    """Implementation of a NN Policy using FelixNet (designed by Felix Berkenkamp).

    Parameters
    ----------
    dim_state: int
        dimension of state.
    dim_action: int
        dimension of action.
    num_states: int, optional
        number of discrete states (None if state is continuous).
    num_actions: int, optional
        number of discrete actions (None if action is continuous).

    Notes
    -----
    This class is only implemented for continuous state and action spaces.

    """

    def __init__(self, dim_state, dim_action, num_states=None, num_actions=None,
                 tau=1.0, deterministic=False):
        super().__init__(dim_state, dim_action, num_states, num_actions, tau=tau,
                         deterministic=deterministic)
        self.layers = [64, 64]

        self.linear1 = nn.Linear(dim_state, 64, bias=True)
        nn.init.zeros_(self.linear1.bias)

        self.linear2 = nn.Linear(64, 64, bias=True)
        nn.init.zeros_(self.linear2.bias)

        self._mean = nn.Linear(64, dim_action, bias=False)
        # torch.nn.init.uniform_(self._mean.weight, -0.1, 0.1)

        self._covariance = nn.Linear(64, dim_action, bias=False)
        # torch.nn.init.uniform_(self._covariance.weight, -0.01, 0.01)

        if self.discrete_state or self.discrete_action:
            raise ValueError("Felix Policy is for Continuous Problems")

    def forward(self, state):
        """Execute felix network."""
        x = torch.relu(self.linear1(state))
        x = torch.relu(self.linear2(x))

        mean = torch.tanh(self._mean(x))
        covariance = torch.nn.functional.softplus(self._covariance(x))
        covariance = torch.diag_embed(covariance)

        if self.deterministic:
            return mean, torch.zeros(1)
        else:
            return mean, covariance

    def embeddings(self, state):
        """Get embeddings of the value-function at a given state."""
        if self.discrete_state:
            state = one_hot_encode(state.long(), self.num_states)

        x = torch.relu(self.linear1(state))
        x = torch.relu(self.linear2(x))

        return x.squeeze()
