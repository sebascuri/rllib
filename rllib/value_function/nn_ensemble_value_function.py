"""Value and Q-Functions parametrized with ensembles of Neural Networks."""

import torch
import torch.nn as nn

from .nn_value_function import NNValueFunction, NNQFunction


class NNEnsembleValueFunction(NNValueFunction):
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

    def __init__(self, dim_state, num_heads, num_states=-1, layers=None,
                 biased_head=True, non_linearity='ReLU', tau=0.0, input_transform=None):
        assert num_heads > 0
        self.num_heads = num_heads

        super().__init__(dim_state, num_states)
        self.nn = nn.ModuleList(
            [NNValueFunction(dim_state=dim_state, num_states=num_states,
                             layers=layers, biased_head=biased_head,
                             non_linearity=non_linearity, tau=tau,
                             input_transform=input_transform)
             for _ in range(num_heads)])

    @classmethod
    def from_q_function(cls, value_function, num_heads: int):
        """Create ensemble form q-funciton."""
        out = cls(dim_state=value_function.dim_state,
                  num_heads=num_heads, num_states=value_function.num_states,
                  tau=value_function.tau,
                  input_transform=value_function.input_transform
                  )

        out.nn = nn.ModuleList([
            value_function.__class__.from_other(value_function, copy=False)
            for _ in range(num_heads)])
        return out

    def forward(self, state, action=torch.tensor(float('nan'))):
        """Get value of the value-function at a given state."""
        return [value_function(state, action) for value_function in self.nn]


class NNEnsembleQFunction(NNQFunction):
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

    def __init__(self, dim_state, dim_action, num_heads, num_states=-1, num_actions=-1,
                 layers=None, biased_head=True, non_linearity='ReLU',
                 tau=0.0, input_transform=None):
        self.num_heads = num_heads
        assert num_heads > 0
        # Initialize from q-function.

        super().__init__(dim_state, dim_action, num_states, num_actions)

        self.nn = nn.ModuleList(
            [NNQFunction(dim_state=dim_state, dim_action=dim_action,
                         num_states=num_states, num_actions=num_actions, layers=layers,
                         biased_head=biased_head, non_linearity=non_linearity, tau=tau,
                         input_transform=input_transform)
             for _ in range(self.num_heads)]
        )

    @classmethod
    def from_q_function(cls, q_function, num_heads: int):
        """Create ensemble form q-funciton."""
        out = cls(dim_state=q_function.dim_state, dim_action=q_function.dim_action,
                  num_heads=num_heads, num_states=q_function.num_states,
                  num_actions=q_function.num_actions,
                  tau=q_function.tau, input_transform=q_function.input_transform)

        out.nn = nn.ModuleList([
            q_function.__class__.from_other(q_function, copy=False)
            for _ in range(num_heads)])
        return out

    def forward(self, state, action=torch.tensor(float('nan'))):
        """Get value of the q-function at a given state-action pair."""
        return [q_function(state, action) for q_function in self.nn]
