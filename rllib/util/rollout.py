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
    done = False
    trajectory = []
    with torch.no_grad():
        while not done:
            state_torch = torch.from_numpy(state).float()
            action_torch = policy.action(state_torch).sample()
            action = action_torch.numpy()
            next_state, reward, done, _ = environment.step(action)
            observation = Observation(state=state,
                                      action=action,
                                      reward=reward,
                                      next_state=next_state,
                                      done=done)
            trajectory.append(observation)
            state = next_state

    return trajectory
