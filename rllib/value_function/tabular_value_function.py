"""Value Functions parametrized with tables."""


from . import NNValueFunction, NNQFunction
import torch.nn as nn


class TabularValueFunction(NNValueFunction):
    """Implement tabular value function."""

    def __init__(self, num_states, tau=1.0):
        super().__init__(dim_state=1, num_states=num_states, tau=tau, biased_head=False)
        nn.init.zeros_(self._value_function.head.weight)

    @property
    def table(self):
        return self._value_function.head.weight

    def set_value(self, state, new_value):
        self._value_function.head.weight[0, state] = new_value


class TabularQFunction(NNQFunction):
    """Implement tabular value function."""

    def __init__(self, num_states, num_actions, tau=1.0):
        super().__init__(dim_state=1, dim_action=1,
                         num_states=num_states, num_actions=num_actions,
                         tau=tau, biased_head=False)

        nn.init.zeros_(self._q_function.head.weight)

    @property
    def table(self):
        return self._q_function.head.weight

    def set_value(self, state, action, new_value):
        self._q_function.head.weight[action, state] = new_value
