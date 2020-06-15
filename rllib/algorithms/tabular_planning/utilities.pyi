"""Utilities for tabular planning functions."""

from typing import List

from rllib.value_function import TabularValueFunction

def init_value_function(
    num_states: int, terminal_states: List[int]
) -> TabularValueFunction: ...
