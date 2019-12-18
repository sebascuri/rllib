"""Helper functions to conduct a rollout with policies or agents."""

import torch
from rllib.dataset import Observation


__all__ = ['rollout_agent', 'rollout_policy']


def _step(environment, state, action, render):
    next_state, reward, done, _ = environment.step(action)
    observation = Observation(state=state,
                              action=action,
                              reward=reward,
                              next_state=next_state,
                              done=done)
    state = next_state
    if render:
        environment.render()
    return observation, state, done


def rollout_agent(environment, agent, num_episodes=1, max_steps=1000, render=False):
    """Conduct a single rollout of an agent in an environment.

    Parameters
    ----------
    environment: AbstractEnvironment
    agent: AbstractAgent
    num_episodes: int, optional (default=1)
    max_steps: int.
    render: bool.

    """
    for _ in range(num_episodes):
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
    agent.end_interaction()


def rollout_policy(environment, policy, num_episodes=1, max_steps=1000, render=False):
    """Conduct a single rollout of a policy in an environment.

    Parameters
    ----------
    environment : AbstractEnvironment
    policy : AbstractPolicy
    num_episodes: int, optional (default=1)
    max_steps: int
    render: bool

    Returns
    -------
    trajectories: list of list of Observation.

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
