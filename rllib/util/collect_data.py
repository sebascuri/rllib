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
    for _ in range(num_samples):
        state = state_dist.sample()
        if isinstance(policy, AbstractPolicy):
            action_dist = tensor_to_distribution(policy(state), **policy.dist_params)
        else:  # random action_distribution
            action_dist = policy
        action = action_dist.sample()

        state = state.numpy()
        action = action.numpy()
        environment.state = state
        next_state, reward, done, _ = environment.step(action)
        transitions.append(Observation(state, action, reward, next_state).to_torch())

    return transitions


def collect_model_transitions(
    state_dist, policy, dynamical_model, reward_model, num_samples
):
    """Collect transitions by interacting with an environment.

    Parameters
    ----------
    state_dist: Distribution.
        State distribution.
    policy: AbstractPolicy or Distribution.
        Policy to interact with the environment.
    dynamical_model: AbstractModel.
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
    state = state_dist.sample((num_samples,))
    if isinstance(policy, AbstractPolicy):
        action_dist = tensor_to_distribution(policy(state), **policy.dist_params)
        action = action_dist.sample()
    else:  # action_distribution
        action_dist = policy
        action = action_dist.sample((num_samples,))

    next_state = tensor_to_distribution(dynamical_model(state, action)).sample()
    reward = tensor_to_distribution(reward_model(state, action, next_state)).sample()

    transitions = []
    for state_, action_, reward_, next_state_ in zip(state, action, reward, next_state):
        transitions.append(
            Observation(state_, action_, reward_, next_state_).to_torch()
        )
    return transitions
