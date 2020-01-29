"""Value and Q-Functions parametrized with Neural Networks."""

import torch
from .abstract_value_function import AbstractValueFunction, AbstractQFunction
from rllib.util.neural_networks import DeterministicNN
from rllib.util.neural_networks import update_parameters, one_hot_encode
from rllib.policy import NNPolicy


__all__ = ['NNValueFunction', 'NNQFunction']


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

    def __init__(self, dim_state, num_states=None, layers: list = None, tau=1.0,
                 biased_head=True):
        super().__init__(dim_state, num_states)

        if self.discrete_state:
            num_inputs = self.num_states
        else:
            num_inputs = self.dim_state

        self.value_function = DeterministicNN(num_inputs, 1, layers,
                                              biased_head=biased_head)
        self._tau = tau

        if not layers:
            layers = [num_inputs]
        self.dimension = layers[-1] + 1

    def __call__(self, state, action=None):
        """Get value of the value-function at a given state."""
        if self.discrete_state:
            state = one_hot_encode(state.long(), self.num_states)
        return self.value_function(state).squeeze(-1)

    @property
    def parameters(self):
        """Get iterator of value function parameters."""
        return self.value_function.parameters()

    @parameters.setter
    def parameters(self, new_params):
        """Set value function parameters."""
        update_parameters(self.value_function.parameters(), new_params, self._tau)

    def embeddings(self, state):
        """Get embeddings of the value-function at a given state."""
        if self.discrete_state:
            state = one_hot_encode(state.long(), self.num_states)
        if self.value_function.layers is None:
            embeddings = state.squeeze(0)
        else:
            embeddings = self.value_function.hidden_layers(state).squeeze(0)

        return torch.cat((embeddings, torch.ones(embeddings.shape[0], 1)), dim=1)

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
                 layers: list = None, tau=1.0, biased_head=True):
        super().__init__(dim_state, dim_action, num_states, num_actions)

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

        self.q_function = DeterministicNN(num_inputs, num_outputs, layers,
                                          biased_head=biased_head)
        self._tau = tau

    def __call__(self, state, action=None):
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

        if action is None:
            if not self.discrete_action:
                raise NotImplementedError
            action_value = self.q_function(state)
            return action_value
        elif action.dim() == 0:
            action.unsqueeze(0)

        if self.discrete_action:
            action = action.unsqueeze(-1).long()

        if not self.discrete_state and not self.discrete_action:
            state_action = torch.cat((state, action), dim=-1)
            return self.q_function(state_action).squeeze(-1)
        elif self.discrete_state and self.discrete_action:
            return self.q_function(state).gather(-1, action).squeeze(-1)
        elif not self.discrete_state and self.discrete_action:
            return self.q_function(state).gather(-1, action).squeeze(-1)

    @property
    def parameters(self):
        """Get iterator of q-function parameters."""
        return self.q_function.parameters()

    @parameters.setter
    def parameters(self, new_params):
        """Set q-function parameters."""
        update_parameters(self.q_function.parameters(), new_params, self._tau)

    def max(self, state):
        """Get the maximum over actions of the q-function at a given state."""
        if not self.discrete_action:
            raise NotImplementedError
        else:
            return self(state).max(dim=-1)[0]

    def argmax(self, state):
        """Get the action that maximizes the q-function at a given state."""
        if not self.discrete_action:
            raise NotImplementedError
        else:
            return self(state).argmax(dim=-1)

    def extract_policy(self, temperature=1.0):
        """Extract the policy induced by the Q-Value function."""
        if not self.discrete_action:
            raise NotImplementedError
        else:
            policy = NNPolicy(self.dim_state, self.dim_action,
                              num_states=self.num_states,
                              num_actions=self.num_actions,
                              layers=self.q_function.layers,
                              temperature=temperature)
            policy.parameters = self.q_function.parameters()
            return policy
