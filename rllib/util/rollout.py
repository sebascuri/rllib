"""Helper functions to conduct a rollout with policies or agents."""

import torch
from rllib.dataset.datatypes import Observation
import pickle


def _step(environment, state, action, render):
    try:
        next_state, reward, done, _ = environment.step(action)
    except TypeError:
        next_state, reward, done, _ = environment.step(action.item())
    observation = Observation(state=state,
                              action=action,
                              reward=reward,
                              next_state=next_state,
                              done=done)
    state = next_state
    if render:
        environment.render()
    return observation, state, done


def rollout_agent(environment, agent, num_episodes=1, max_steps=1000, render=False,
                  milestones=None):
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
    milestones: list.
        List with episodes in which to save the agent.

    """
    milestones = list() if milestones is None else milestones
    for episode in range(num_episodes):
        state = environment.reset()
        agent.start_episode()
        done = False
        while not done:
            action = agent.act(state)
            observation, state, done = _step(environment, state, action, render)
            agent.observe(observation)
            if max_steps <= environment.time:
                break
        agent.end_episode()

        if episode in milestones:
            file_name = '{}_{}_{}.pkl'.format(environment.name, agent.name, episode)
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
    for _ in range(num_episodes):
        state = environment.reset()
        done = False
        trajectory = []
        with torch.no_grad():
            while not done:
                action = policy(torch.tensor(state).float()).sample().numpy()
                observation, state, done = _step(environment, state, action, render)
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

    for _ in range(max_steps):
        # Sample actions
        action = policy(state)
        if action.has_rsample:
            action = action.rsample()
        else:
            action = action.sample()

        # % Sample a reward
        reward_distribution = reward_model(state, action)
        if reward_distribution.has_rsample:
            reward = reward_distribution.rsample()
        else:
            reward = reward_distribution.sample()

        # Sample next states
        next_state = dynamical_model(state, action)
        if next_state.has_rsample:
            next_state = next_state.rsample()
        else:
            next_state = next_state.sample()

        # Check for termination.
        if termination is not None and termination(state, action):
            trajectory.append(Observation(state, action, reward, next_state, True))
            break
        else:
            trajectory.append(Observation(state, action, reward, next_state, False))

        # Update state
        state = next_state

    return trajectory
