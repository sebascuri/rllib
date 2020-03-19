"""Value and Q-Functions parametrized with ensembles of Neural Networks."""

import torch.nn as nn

from .abstract_value_function import AbstractValueFunction, AbstractQFunction
from .nn_value_function import NNValueFunction, NNQFunction


class NNEnsembleValueFunction(AbstractValueFunction):
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

    def __init__(self, value_function=None, dim_state=1, num_states=None, layers=None,
                 tau=1.0, biased_head=True, num_heads=1):
        assert num_heads > 0
        # Initialize from value-function.
        if value_function is not None:
            dim_state = value_function.dim_state
            num_states = value_function.num_states

        super().__init__(dim_state, num_states)
        if value_function is not None:
            layers = value_function.nn.layers
            tau = value_function.tau
            biased_head = value_function.nn.head.bias is not None

        self.ensemble = nn.ModuleList(
            [NNValueFunction(dim_state, num_states, layers, tau, biased_head
                             ) for _ in range(num_heads)])

        self.dimension = self.ensemble[0].dimension

    def __len__(self):
        """Get length of ensemble."""
        return len(self.ensemble)

    def __getitem__(self, item):
        """Get ensemble item."""
        return self.ensemble[item]

    def __call__(self, state, action=None):
        """Get value of the value-function at a given state."""
        return [value_function(state, action) for value_function in self.ensemble]


class NNEnsembleQFunction(AbstractQFunction):
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

    def __init__(self, q_function=None, dim_state=1, dim_action=1,
                 num_states=None, num_actions=None,
                 layers=None, tau=1.0, biased_head=True, num_heads=1):

        assert num_heads > 0
        # Initialize from q-function.
        if q_function is not None:
            dim_state = q_function.dim_state
            num_states = q_function.num_states

        super().__init__(dim_state, num_states)
        if q_function is not None:
            layers = q_function.nn.layers
            tau = q_function.tau
            biased_head = q_function.nn.head.bias is not None

        self.ensemble = nn.ModuleList(
            [NNQFunction(dim_state, dim_action, num_states, num_actions, layers, tau,
                         biased_head) for _ in range(num_heads)])

    def __len__(self):
        """Get size of ensemble."""
        return len(self.ensemble)

    def __getitem__(self, item):
        """Get ensemble item."""
        return self.ensemble[item]

    def forward(self, state, action=None):
        """Get value of the q-function at a given state-action pair."""
        return [q_function(state, action) for q_function in self.ensemble]

    def value(self, state, policy, num_samples=0):
        """See `AbstractQFunction.value'."""
        return [q_function.value(state, policy, num_samples)
                for q_function in self.ensemble]
