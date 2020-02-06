"""Utilities for tabular planning functions."""

from rllib.value_function import TabularValueFunction
from typing import List


def init_value_function(num_states: int, terminal_states: List[int]
                         ) -> TabularValueFunction: ...
