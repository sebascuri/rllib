"""Helper functions to conduct a rollout with policies or agents."""

import torch
from rllib.dataset import Observation
from rllib.agent import AbstractAgent
from rllib.environment import AbstractEnvironment
from rllib.policy import AbstractPolicy
from typing import List


__all__ = ['rollout_agent', 'rollout_policy']


def rollout_agent(environment: AbstractEnvironment, agent: AbstractAgent,
                  num_episodes: int = 1, max_steps: int = 1000, render: bool = False
                  ) -> None:
    """Conduct a single rollout of an agent in an environment.

    Parameters
    ----------
    environment: AbstractEnvironment
    agent: AbstractAgent
    num_episodes: int, optional (default=1)
    max_steps: int.
    render:bool.

    """
    for _ in range(num_episodes):
        state = environment.reset()
        agent.start_episode()
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = environment.step(action)
            max_time = max_steps <= environment.time
            if done:
                print(environment.time)
            observation = Observation(state=state,
                                      action=action,
                                      reward=reward,
                                      next_state=next_state,
                                      done=done)
            agent.observe(observation)
            state = next_state
            if render:
                environment.render()

            if max_time:
                break
        agent.end_episode()
    agent.end_interaction()


def rollout_policy(environment: AbstractEnvironment, policy: AbstractPolicy,
                   num_episodes: int = 1, max_steps: int = 1000, render: bool = False
                   ) -> List[List[Observation]]:
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
                max_time = max_steps <= environment.time
                observation = Observation(state=state,
                                          action=action,
                                          reward=reward,
                                          next_state=next_state,
                                          done=done)
                trajectory.append(observation)
                state = next_state

                if render:
                    environment.render()
                if max_time:
                    break
        trajectories.append(trajectory)
    return trajectories
