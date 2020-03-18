"""Value and Q-Functions parametrized with Neural Networks."""

import torch
import torch.nn as nn
from .abstract_value_function import AbstractValueFunction, AbstractQFunction
from rllib.util.neural_networks import DeterministicNN
from rllib.util.neural_networks import one_hot_encode


class NNValueFunction(AbstractValueFunction):
    """Implementation of a Value Function implemented with a Neural Network.

    Parameters
    ----------
    dim_state: int
        dimension of state.
    num_states: int, optional
        number of discrete states (None if state is continuous).
    layers: list, optional
        width of layers, each layer is connected with ReLUs non-linearities.
    tau: float, optional
        when a new parameter is set, tau low-passes the new parameter with the old one.
    biased_head: bool, optional
        flag that indicates if head of NN has a bias term or not.

    """

    def __init__(self, dim_state, num_states=None, layers=None, tau=1.0,
                 biased_head=True, input_transform=None):
        super().__init__(dim_state, num_states, tau=tau)

        if self.discrete_state:
            num_inputs = self.num_states
        else:
            num_inputs = self.dim_state

        self.input_transform = input_transform
        self.nn = DeterministicNN(num_inputs, 1, layers, biased_head=biased_head)
        self.dimension = self.nn.embedding_dim

    def forward(self, state, action=torch.empty(1)):
        """Get value of the value-function at a given state."""
        if self.input_transform is not None:
            state = self.input_transform(state)

        if isinstance(self.num_states, int):
            state = one_hot_encode(state.long(), torch.tensor(self.num_states))
        return self.nn(state).squeeze(-1)

    def embeddings(self, state):
        """Get embeddings of the value-function at a given state."""
        if isinstance(self.num_states, int):
            state = one_hot_encode(state.long(), torch.tensor(self.num_states))
        return self.value_function.last_layer_embeddings(state).squeeze()


class NNQFunction(AbstractQFunction):
    """Implementation of a Q-Function implemented with a Neural Network.

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
    tau: float, optional
        when a new parameter is set, tau low-passes the new parameter with the old one.
    biased_head: bool, optional
        flag that indicates if head of NN has a bias term or not.
    """

    def __init__(self, dim_state, dim_action, num_states=None, num_actions=None,
                 layers=None, tau=1.0, biased_head=True, input_transform=None):
        super().__init__(dim_state, dim_action, num_states, num_actions, tau)

        if not self.discrete_state and not self.discrete_action:
            num_inputs = self.dim_state + self.dim_action
            num_outputs = 1
        elif self.discrete_state and self.discrete_action:
            num_inputs = self.num_states
            num_outputs = self.num_actions
        elif not self.discrete_state and self.discrete_action:
            num_inputs = self.dim_state
            num_outputs = self.num_actions
        else:
            raise NotImplementedError("If states are discrete, so should be actions.")

        self.input_transform = input_transform
        self.nn = DeterministicNN(num_inputs, num_outputs, layers,
                                  biased_head=biased_head)

    def forward(self, state, action=None):
        """Get value of the value-function at a given state.

        Parameters
        ----------
        state: torch.Tensor
        action: torch.Tensor

        Returns
        -------
        value: torch.Tensor

        """
        if isinstance(self.num_states, int):
            state = one_hot_encode(state.long(), torch.tensor(self.num_states))

        if self.input_transform is not None:
            state, action = self.input_transform(state, action)

        if action is None:
            if not self.discrete_action:
                raise NotImplementedError
            action_value = self.nn(state)
            return action_value
        elif action.dim() == 0:
            action.unsqueeze(0)

        if self.discrete_action:
            action = action.unsqueeze(-1).long()

        if not self.discrete_action:
            state_action = torch.cat((state, action), dim=-1)
            return self.nn(state_action).squeeze(-1)
        else:
            return self.nn(state).gather(-1, action).squeeze(-1)


class TabularValueFunction(NNValueFunction):
    """Implement tabular value function."""

    def __init__(self, num_states, tau=1.0, biased_head=False):
        super().__init__(dim_state=1, num_states=num_states, tau=tau,
                         biased_head=biased_head)
        nn.init.zeros_(self.nn.head.weight)

    @property
    def table(self):
        """Get table representation of value function."""
        return self.nn.head.weight

    def set_value(self, state, new_value):
        """Set value to value function at a given state.

        Parameters
        ----------
        state: int
            State number.
        new_value: float
            value of state.

        """
        self.nn.head.weight[0, state] = new_value


class TabularQFunction(NNQFunction):
    """Implement tabular value function."""

    def __init__(self, num_states, num_actions, tau=1.0, biased_head=False):
        super().__init__(dim_state=1, dim_action=1,
                         num_states=num_states, num_actions=num_actions,
                         tau=tau, biased_head=biased_head)

        nn.init.zeros_(self.nn.head.weight)

    @property
    def table(self):
        """Get table representation of Q-function."""
        return self.nn.head.weight

    def set_value(self, state, action, new_value):
        """Set value to q-function at a given state-action pair.

        Parameters
        ----------
        state: int
            State number.
        action: int
            Action number.
        new_value: float
            value of state.

        """
        self.nn.head.weight[action, state] = new_value
