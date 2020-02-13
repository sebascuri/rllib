"""Value and Q-Functions parametrized with ensembles of Neural Networks."""

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
        # Initialize from value-function.
        if value_function is not None:
            dim_state = value_function.dim_state
            num_states = value_function.num_states
            layers = value_function.value_function.layers
            tau = value_function.tau
            biased_head = value_function.value_function.head.bias is not None

        super().__init__(dim_state, num_states)

        assert num_heads > 0

        self.ensemble = [NNValueFunction(dim_state, num_states, layers, tau, biased_head
                                         ) for _ in range(num_heads)]

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

    @property
    def parameters(self):
        """Get iterator of value function parameters."""
        return [value_function.parameters for value_function in self.ensemble]

    @parameters.setter
    def parameters(self, new_params):
        """Set value function parameters."""
        for value_function, new_param in zip(self.ensemble, new_params):
            value_function.parameters = new_param

    def embeddings(self, state):
        """Get embeddings of the value-function at a given state."""
        return [value_function.embeddings(state) for value_function in self.ensemble]


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

        # Initialize from value-function.
        if q_function is not None:
            dim_state = q_function.dim_state
            num_states = q_function.num_states
            dim_action = q_function.dim_action
            num_actions = q_function.num_actions
            layers = q_function.q_function.layers
            tau = q_function.tau
            biased_head = q_function.q_function.head.bias is not None

        super().__init__(dim_state, dim_action, num_states, num_actions)

        assert num_heads > 0

        self.ensemble = [NNQFunction(
            dim_state, dim_action, num_states, num_actions, layers, tau, biased_head
        ) for _ in range(num_heads)]

    def __len__(self):
        """Get size of ensemble."""
        return len(self.ensemble)

    def __getitem__(self, item):
        """Get ensemble item."""
        return self.ensemble[item]

    def __call__(self, state, action=None):
        """Get value of the q-function at a given state-action pair."""
        return [q_function(state, action) for q_function in self.ensemble]

    @property
    def parameters(self):
        """Get iterator of q function parameters."""
        return [q_function.parameters for q_function in self.ensemble]

    @parameters.setter
    def parameters(self, new_params):
        """Set q-function parameters."""
        for q_function, new_param in zip(self.ensemble, new_params):
            q_function.parameters = new_param
