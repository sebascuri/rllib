import torch
from rllib.dataset import Observation


def rollout_agent(environment, agent, num_episodes=1):
    """Conduct a single rollout of an agent in an environment.

    Parameters
    ----------
    environment: rllib.environment.AbstractEnvironment
    agent: AbstractAgent
    num_episodes: int, optional

    Returns
    -------
    None

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
        agent.end_episode()
    agent.end_interaction()


def rollout_policy(environment, policy, num_episodes=1):
    """Conduct a single rollout of a policy in an environment.

    Parameters
    ----------
    environment : rllib.environment.AbstractEnvironment
    policy : AbstractPolicy
    num_episodes: int, optional

    Returns
    -------
    trajectory: list

    """
    trajectories = []
    for _ in range(num_episodes):
        state = environment.reset()
        done = False
        trajectory = []
        with torch.no_grad():
            while not done:
                state_torch = torch.from_numpy(state).float()
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
