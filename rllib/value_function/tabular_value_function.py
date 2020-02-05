"""Value Functions parametrized with tables."""

from . import NNValueFunction, NNQFunction
import torch.nn as nn


class TabularValueFunction(NNValueFunction):
    """Implement tabular value function."""

    def __init__(self, num_states, tau=1.0, biased_head=False):
        super().__init__(dim_state=1, num_states=num_states, tau=tau,
                         biased_head=biased_head)
        nn.init.zeros_(self.value_function.head.weight)

    @property
    def table(self):
        """Get table representation of value function."""
        return self.value_function.head.weight

    def set_value(self, state, new_value):
        """Set value to value function at a given state.

        Parameters
        ----------
        state: int
            State number.
        new_value: float
            value of state.

        """
        self.value_function.head.weight[0, state] = new_value

    def get_nn(self):
        """Get a NNValueFunction."""
        val = NNValueFunction(self.dim_state, self.num_states, layers=[], tau=self._tau,
                              biased_head=self.value_function.head.bias is not None)
        val.value_function.head.weight.data = self.value_function.head.weight.data

        return val


class TabularQFunction(NNQFunction):
    """Implement tabular value function."""

    def __init__(self, num_states, num_actions, tau=1.0, biased_head=False):
        super().__init__(dim_state=1, dim_action=1,
                         num_states=num_states, num_actions=num_actions,
                         tau=tau, biased_head=biased_head)

        nn.init.zeros_(self.q_function.head.weight)

    @property
    def table(self):
        """Get table representation of Q-function."""
        return self.q_function.head.weight

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
        self.q_function.head.weight[action, state] = new_value
