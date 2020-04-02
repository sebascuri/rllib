"""Helper functions to conduct a rollout with policies or agents."""

import pickle

import torch
from tqdm import tqdm

from rllib.dataset.datatypes import RawObservation
from rllib.util.utilities import tensor_to_distribution


def step(environment, state, action, pi, render):
    """Perform a single step in an environment."""
    try:
        next_state, reward, done, _ = environment.step(action)
    except TypeError:
        next_state, reward, done, _ = environment.step(action.item())

    if not isinstance(action, torch.Tensor):
        action = torch.tensor(action, dtype=torch.get_default_dtype())

    observation = RawObservation(state=state,
                                 action=action,
                                 reward=reward,
                                 next_state=next_state,
                                 done=done,
                                 entropy=pi.entropy(),
                                 log_prob_action=pi.log_prob(action)
                                 ).to_torch()
    state = next_state
    if render:
        environment.render()
    return observation, state, done


def rollout_agent(environment, agent, num_episodes=1, max_steps=1000, render=False,
                  print_frequency=0, milestones=None, plot_callbacks=None):
    """Conduct a rollout of an agent in an environment.

    Parameters
    ----------
    environment: AbstractEnvironment
        Environment with which the abstract interacts.
    agent: AbstractAgent
        Agent that interacts with the environment.
    num_episodes: int, optional (default=1)
        Number of episodes.
    max_steps: int.
        Maximum number of steps per episode.
    render: bool.
        Flag that indicates whether to render the environment or not.
    print_frequency: int, optional.
        Print agent stats every `print_episodes' if > 0.
    milestones: List[int], optional.
        List with episodes in which to save the agent.
    plot_callbacks: List[Callable[[AbstractAgent], None]], optional.
        List of functions for plotting the agent.
    """
    milestones = list() if milestones is None else milestones
    plot_callbacks = list() if plot_callbacks is None else plot_callbacks
    for episode in tqdm(range(num_episodes)):
        state = environment.reset()
        agent.start_episode()
        done = False
        while not done:
            action = agent.act(state)
            observation, state, done = step(environment, state, action, agent.pi,
                                             render)
            agent.observe(observation)
            if max_steps <= environment.time:
                break
        agent.end_episode()

        if print_frequency and episode % print_frequency == 0:
            print(agent)
            for plot_callback in plot_callbacks:
                plot_callback(agent)

        if episode in milestones:
            file_name = f"{environment.name}_{agent.name}_{episode}.pkl"
            with open(file_name, 'wb') as file:
                pickle.dump(agent, file)
    agent.end_interaction()


def rollout_policy(environment, policy, num_episodes=1, max_steps=1000, render=False):
    """Conduct a rollout of a policy in an environment.

    Parameters
    ----------
    environment: AbstractEnvironment
        Environment with which the policy interacts.
    policy: AbstractPolicy
        Policy that interacts with the environment.
    num_episodes: int, optional (default=1)
        Number of episodes.
    max_steps: int.
        Maximum number of steps per episode.
    render: bool.
        Flag that indicates whether to render the environment or not.

    Returns
    -------
    trajectories: List[Trajectory]=List[List[Observation]]
        A list of trajectories.

    """
    trajectories = []
    for _ in tqdm(range(num_episodes)):
        state = environment.reset()
        done = False
        trajectory = []
        with torch.no_grad():
            while not done:
                pi = tensor_to_distribution(policy(
                    torch.tensor(state, dtype=torch.get_default_dtype())))
                action = pi.sample().numpy()
                observation, state, done = step(environment, state, action, pi, render)
                trajectory.append(observation)
                if max_steps <= environment.time:
                    break
        trajectories.append(trajectory)
    return trajectories


def rollout_model(dynamical_model, reward_model, policy, initial_state,
                  termination=None, max_steps=1000):
    """Conduct a rollout of a policy interacting with a model.

    Parameters
    ----------
    dynamical_model: AbstractModel
        Dynamical Model with which the policy interacts.
    reward_model: AbstractReward, optional.
        Reward Model with which the policy interacts.
    policy: AbstractPolicy
        Policy that interacts with the environment.
    initial_state: State
        Starting states for the interaction.
    termination: Callable.
        Termination condition to finish the rollout.
    max_steps: int.
        Maximum number of steps per episode.

    Returns
    -------
    trajectory: Trajectory=List[Observation]
        A list of observations.

    Notes
    -----
    It will try to do the re-parametrization trick with the policy and models.
    """
    trajectory = list()
    state = initial_state
    assert max_steps > 0
    for _ in range(max_steps):
        # Sample actions
        pi = tensor_to_distribution(policy(state))
        if pi.has_rsample:
            action = pi.rsample()
        else:
            action = pi.sample()

        # % Sample a reward
        reward_distribution = tensor_to_distribution(reward_model(state, action))
        if reward_distribution.has_rsample:
            reward = reward_distribution.rsample()
        else:
            reward = reward_distribution.sample()

        # Sample next states
        next_state_out = dynamical_model(state, action)
        next_state_distribution = tensor_to_distribution(next_state_out)
        if len(next_state_out) == 3:
            next_state_tril = next_state_out[2]
        else:
            next_state_tril = next_state_out[1]

        if next_state_distribution.has_rsample:
            next_state = next_state_distribution.rsample()
        else:
            next_state = next_state_distribution.sample()

        # Check for termination.
        if termination is not None and termination(state, action):
            done = True
        else:
            done = False

        trajectory.append(
            RawObservation(state, action, reward, next_state, done,
                           log_prob_action=pi.log_prob(action),
                           entropy=pi.entropy(),
                           next_state_scale_tril=next_state_tril).to_torch()
        )

        # Update state
        state = next_state

        if done:
            break

    return trajectory


def rollout_actions(dynamical_model, reward_model, action_sequence, initial_state,
                    termination=None, max_steps=1000):
    """Conduct a rollout of an action sequence interacting with a model.

    Parameters
    ----------
    dynamical_model: AbstractModel
        Dynamical Model with which the policy interacts.
    reward_model: AbstractReward, optional.
        Reward Model with which the policy interacts.
    action_sequence: Action
        Action Sequence that interacts with the environment.
        The dimensions are [horizon x num_samples x dim_action].
    initial_state: State
        Starting states for the interaction.
        The dimensions are [1 x num_samples x dim_state].
    termination: Callable.
        Termination condition to finish the rollout.
    max_steps: int.
        Maximum number of steps per episode.

    Returns
    -------
    trajectory: Trajectory=List[Observation]
        A list of observations.

    Notes
    -----
    It will try to do the re-parametrization trick with the policy and models.
    """
    trajectory = list()
    state = initial_state
    assert max_steps > 0
    for t in range(max_steps):
        # Sample actions
        action = action_sequence[t]

        # % Sample a reward
        reward_distribution = tensor_to_distribution(reward_model(state, action))
        if reward_distribution.has_rsample:
            reward = reward_distribution.rsample()
        else:
            reward = reward_distribution.sample()

        # Sample next states
        next_state_out = dynamical_model(state, action)
        next_state_distribution = tensor_to_distribution(next_state_out)
        if len(next_state_out) == 3:
            next_state_tril = next_state_out[2]
        else:
            next_state_tril = next_state_out[1]

        if next_state_distribution.has_rsample:
            next_state = next_state_distribution.rsample()
        else:
            next_state = next_state_distribution.sample()

        # Check for termination.
        if termination is not None and termination(state, action):
            done = True
        else:
            done = False

        trajectory.append(
            RawObservation(state, action, reward, next_state, done,
                           next_state_scale_tril=next_state_tril).to_torch()
        )

        # Update state
        state = next_state

        if done:
            break

    return trajectory
