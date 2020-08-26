"""Utilities for tabular planning functions."""

from rllib.value_function import TabularValueFunction


def init_value_function(num_states, terminal_states=None):
    """Initialize value function."""
    value_function = TabularValueFunction(num_states=num_states)
    terminal_states = [] if terminal_states is None else terminal_states
    for terminal_state in terminal_states:
        value_function.set_value(terminal_state, 0)

    return value_function
