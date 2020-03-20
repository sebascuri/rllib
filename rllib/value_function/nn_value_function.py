"""Value and Q-Functions parametrized with Neural Networks."""

import torch
import torch.jit

from rllib.util.neural_networks import DeterministicNN
from rllib.util.neural_networks import one_hot_encode
from .abstract_value_function import AbstractValueFunction, AbstractQFunction


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

    def __init__(self, dim_state, num_states=-1, layers=None, biased_head=True,
                 non_linearity='ReLU', tau=1.0, input_transform=None):
        super().__init__(dim_state, num_states, tau=tau)

        if self.discrete_state:
            num_inputs = self.num_states
        else:
            num_inputs = self.dim_state

        self.input_transform = input_transform
        if hasattr(input_transform, 'extra_dim'):
            num_inputs += getattr(input_transform, 'extra_dim')

        self.nn = DeterministicNN(num_inputs, 1, layers=layers,
                                  non_linearity=non_linearity, biased_head=biased_head)
        self.dimension = self.nn.embedding_dim

    @classmethod
    def from_other(cls, other):
        """Create new Value Function from another Value Function."""
        new = cls(dim_state=other.dim_state, num_states=other.num_states,
                  tau=other.tau, input_transform=other.input_transform)
        new.nn = other.nn
        return new

    @classmethod
    def from_nn(cls, module, dim_state, num_states=-1, tau=1.0, input_transform=None):
        """Create new Value Function from a Neural Network Implementation."""
        new = cls(dim_state=dim_state, num_states=num_states, tau=tau,
                  input_transform=input_transform)
        new.nn = module
        return new

    def forward(self, state, action=torch.tensor(float('nan'))):
        """Get value of the value-function at a given state."""
        if self.input_transform is not None:
            state = self.input_transform(state)

        if self.discrete_state:
            state = one_hot_encode(state.long(), self.num_states)
        return self.nn(state).squeeze(-1)

    @torch.jit.export
    def embeddings(self, state):
        """Get embeddings of the value-function at a given state."""
        if self.discrete_state:
            state = one_hot_encode(state.long(), self.num_states)
        return self.nn.last_layer_embeddings(state).squeeze()


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

    def __init__(self, dim_state, dim_action, num_states=-1, num_actions=-1,
                 layers=None, biased_head=True, non_linearity='ReLU',
                 tau=1.0, input_transform=None):
        super().__init__(dim_state, dim_action, num_states, num_actions,
                         tau=tau)

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
        if hasattr(input_transform, 'extra_dim'):
            num_inputs += getattr(input_transform, 'extra_dim')

        self.nn = DeterministicNN(num_inputs, num_outputs, layers=layers,
                                  non_linearity=non_linearity, biased_head=biased_head)

    @classmethod
    def from_other(cls, other, copy=True):
        """Create new Value Function from another Value Function."""
        new = cls(dim_state=other.dim_state, dim_action=other.dim_action,
                  num_states=other.num_states, num_actions=other.num_actions,
                  tau=other.tau, input_transform=other.input_transform)
        if copy:
            new.nn = other.nn.__class__.from_other_with_copy
        return new

    @classmethod
    def from_nn(cls, module, dim_state, num_states=-1, tau=1.0, input_transform=None):
        """Create new Value Function from a Neural Network Implementation."""
        new = cls(dim_state=dim_state, num_states=num_states, tau=tau,
                  input_transform=input_transform)
        new.nn = module
        return new

    def forward(self, state, action=torch.tensor(float('nan'))):
        """Get value of the value-function at a given state.

        Parameters
        ----------
        state: torch.Tensor
        action: torch.Tensor

        Returns
        -------
        value: torch.Tensor

        """
        if self.discrete_state:
            state = one_hot_encode(state.long(), self.num_states)

        if self.input_transform is not None:
            state, action = self.input_transform(state, action)

        if torch.isnan(action).all():
            if not self.discrete_action:
                raise NotImplementedError
            action_value = self.nn(state)
            return action_value

        if self.discrete_action:
            action = action.unsqueeze(-1).long()

        if action.ndim < state.ndim:
            resqueeze = True
            action = action.unsqueeze(0)
        else:
            resqueeze = False

        if not self.discrete_action:
            state_action = torch.cat((state, action), dim=-1)
            return self.nn(state_action).squeeze(-1)
        else:
            out = self.nn(state).gather(-1, action).squeeze(-1)
            if resqueeze:
                return out.squeeze(0)
            else:
                return out
