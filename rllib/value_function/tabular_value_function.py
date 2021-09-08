"""Tabular Value and Q Function Implementations."""

import torch
import torch.nn as nn

from .nn_value_function import NNQFunction, NNValueFunction


class TabularValueFunction(NNValueFunction):
    """Implement tabular value function."""

    def __init__(self, *args, **kwargs):
        kwargs.pop("layers", [])
        super().__init__(
            dim_state=kwargs.pop("dim_state", ()),
            biased_head=kwargs.pop("biased_head", False),
            layers=[],
            *args,
            **kwargs,
        )
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
        with torch.no_grad():
            self.nn.head.weight[0, state] = new_value


class TabularQFunction(NNQFunction):
    """Implement tabular value function."""

    def __init__(self, *args, **kwargs):
        kwargs.pop("layers", [])

        super().__init__(
            dim_state=kwargs.pop("dim_state", ()),
            dim_action=kwargs.pop("dim_action", ()),
            biased_head=kwargs.pop("biased_head", False),
            layers=[],
            *args,
            **kwargs,
        )

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
        with torch.no_grad():
            self.nn.head.weight[action, state] = new_value
