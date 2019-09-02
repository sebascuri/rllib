"""Helper functions to conduct a rollout with policies or agents."""

import torch
from rllib.dataset import Observation


def rollout_agent(environment, agent, num_episodes=1):
    """Conduct a single rollout of an agent in an environment.

    Parameters
    ----------
    environment: AbstractEnvironment
    agent: AbstractAgent
    num_episodes: int, optional (default=1)

    """
    for _ in range(num_episodes):
        state = environment.reset()
        agent.start_episode()
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = environment.step(action)
            observation = Observation(state=state,
                                      action=action,
                                      reward=reward,
                                      next_state=next_state,
                                      done=done)
            agent.observe(observation)
            state = next_state

            if agent.episode_length <= environment.time:
                done = True
        agent.end_episode()
    agent.end_interaction()


def rollout_policy(environment, policy, num_episodes=1):
    """Conduct a single rollout of a policy in an environment.

    Parameters
    ----------
    environment : AbstractEnvironment
    policy : AbstractPolicy
    num_episodes: int, optional (default=1)

    Returns
    -------
    trajectory: list of Observation

    """
    trajectories = []
    for _ in range(num_episodes):
        state = environment.reset()
        done = False
        trajectory = []
        with torch.no_grad():
            while not done:
                state_torch = torch.tensor(state).float()
                action = policy(state_torch).sample().numpy()
                next_state, reward, done, _ = environment.step(action)
                observation = Observation(state=state,
                                          action=action,
                                          reward=reward,
                                          next_state=next_state,
                                          done=done)
                trajectory.append(observation)
                state = next_state
        trajectories.append(trajectory)
    return trajectories
