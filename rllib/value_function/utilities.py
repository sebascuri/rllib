"""Python Script Template."""
from rllib.environment import AbstractEnvironment


def get_input_size_value_function(environment: AbstractEnvironment):
    """Get the input size for a value function."""
    if environment.discrete_state:
        return environment.num_states
    else:
        return environment.dim_state


def get_input_output_size_q_function(environment: AbstractEnvironment):
    """Get the input and output size for Q-Function."""
    if not environment.discrete_state and not environment.discrete_action:
        num_inputs = environment.dim_state + environment.dim_action
        num_outputs = 1
    elif environment.discrete_state and environment.discrete_action:
        num_inputs = environment.num_states
        num_outputs = environment.num_actions
    elif not environment.discrete_state and environment.discrete_action:
        num_inputs = environment.dim_state
        num_outputs = environment.num_actions
    else:
        raise NotImplementedError("If states are discrete, so should be actions.")

    return num_inputs, num_outputs
