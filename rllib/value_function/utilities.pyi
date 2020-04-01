from typing import Tuple
from rllib.environment import AbstractEnvironment


def get_input_size_value_function(environment: AbstractEnvironment) -> int: ...


def get_input_output_size_q_function(environment: AbstractEnvironment) -> Tuple[int, int]: ...
