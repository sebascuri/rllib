import torch
from rllib.dataset.observation import Observation


def rollout_agent(environment, agent):
    """Conduct a single rollout of an agent in an environment.

    Parameters
    ----------
    environment : gym.Env
    agent : AbstractAgent

    Returns
    -------
    None

    """

    state = environment.reset()
    agent.start_episode()
    episode_done = False
    while not episode_done:
        action = agent.act(state)
        next_state, reward, episode_done, _ = environment.step(action)
        agent.observe(state, action, reward, next_state)
        state = next_state
        agent.end_episode()


def rollout_policy(environment, policy):
    """Conduct a single rollout of a policy in an environment.

    Parameters
    ----------
    environment : gym.Env
    policy : AbstractPolicy

    Returns
    -------
    trajectory: list

    """

    state = environment.reset()
    episode_done = False
    trajectory = []
    with torch.no_grad():
        while not episode_done:
            state_torch = torch.from_numpy(state).float()
            action_torch = policy(state_torch)
            action = action_torch.numpy()
            next_state, reward, episode_done, _ = environment.step(action)
            observation = Observation(state=state,
                                      action=action,
                                      reward=reward,
                                      next_state=next_state)
            trajectory.append(observation)
            state = next_state

    return trajectory
