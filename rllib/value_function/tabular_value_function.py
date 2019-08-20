"""Value Functions parametrized with tables."""


from .abstract_value_function import AbstractValueFunction
import torch


class TabularValueFunction(AbstractValueFunction):
    """Implement value function."""

    def __init__(self, num_states,):
        super().__init__(dim_state=1, num_states=num_states)
        self._value_function = torch.zeros(num_states)

    def __call__(self, state, action=None):
        return self._value_function[state]

    @property
    def parameters(self):
        return self._value_function

    @parameters.setter
    def parameters(self, new_params):
        self._value_function = new_params

    def set_value(self, state, new_value):
        self._value_function[state] = new_value
