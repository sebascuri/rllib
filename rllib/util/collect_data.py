"""Utilities to collect data."""

from rllib.dataset.datatypes import Observation
from rllib.policy import AbstractPolicy
from rllib.util.utilities import tensor_to_distribution


def collect_environment_transitions(state_dist, policy, environment, num_samples):
    """Collect transitions by interacting with an environment.

    Parameters
    ----------
    state_dist: Distribution.
        State distribution.
    policy: AbstractPolicy or Distribution.
        Policy to interact with the environment.
    environment: AbstractEnvironment.
        Environment with which to interact.
    num_samples: int.
        Number of transitions.

    Returns
    -------
    transitions: List[Observation]
        List of 1-step transitions.

    """
    transitions = []
    for i in range(num_samples):
        state = state_dist.sample()
        if isinstance(policy, AbstractPolicy):
            action_dist = tensor_to_distribution(policy(state))
        else:  # random action_distribution
            action_dist = policy
        action = action_dist.sample()

        state = state.numpy()
        action = action.numpy()
        environment.state = state
        next_state, reward, done, _ = environment.step(action)
        transitions.append(
            Observation(state, action, reward, next_state).to_torch())

    return transitions


def collect_model_transitions(state_dist, policy, dynamic_model, reward_model,
                              num_samples):
    """Collect transitions by interacting with an environment.

    Parameters
    ----------
    state_dist: Distribution.
        State distribution.
    policy: AbstractPolicy or Distribution.
        Policy to interact with the environment.
    dynamic_model: AbstractModel.
        Model with which to interact.
    reward_model: AbstractReward.
        Reward model with which to interact.
    num_samples: int.
        Number of transitions.

    Returns
    -------
    transitions: List[Observation]
        List of 1-step transitions.

    """
    states = state_dist.sample((num_samples,))
    if isinstance(policy, AbstractPolicy):
        action_dist = tensor_to_distribution(policy(states))
        actions = action_dist.sample()
    else:  # action_distribution
        action_dist = policy
        actions = action_dist.sample((num_samples,))

    next_states = tensor_to_distribution(dynamic_model(states, actions)).sample()
    rewards = tensor_to_distribution(reward_model(states, actions)).sample()

    transitions = []
    for state, action, reward, next_state in zip(states, actions, rewards, next_states):
        transitions.append(
            Observation(state, action, reward, next_state).to_torch())
    return transitions
