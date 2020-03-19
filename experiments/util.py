import pickle

import matplotlib.pyplot as plt
import numpy as np

from rllib.util.rollout import rollout_agent


def train(agent, environment, num_episodes, max_steps):
    """Train an agent in an environment.

    Parameters
    ----------
    agent: AbstractAgent
    environment: AbstractEnvironment
    num_episodes: int
    max_steps: int
    """
    agent.train()
    rollout_agent(environment, agent, num_episodes=num_episodes, max_steps=max_steps)

    for key, log in agent.logs.items():
        plt.plot(log.episode_log)
        plt.xlabel('Episode')
        plt.ylabel(' '.join(key.split('_')).capitalize())
        plt.title('{} in {}'.format(agent.name, environment.name))
        plt.show()
    print(repr(agent))


def evaluate(agent, environment, num_episodes, max_steps):
    """Evaluate an agent in an environment.

    Parameters
    ----------
    agent: AbstractAgent
    environment: AbstractEnvironment
    num_episodes: int
    max_steps: int
    """
    agent.eval()
    rollout_agent(environment, agent, max_steps=max_steps, num_episodes=num_episodes,
                  render=True)
    print('Test Rewards:',
          np.array(agent.logs['rewards'].episode_log[-num_episodes]).mean()
          )


def save_agent(agent, file_name):
    """Save the agent as a pickled file.

    Parameters
    ----------
    agent: AbstractAgent.
    file_name: str.

    """
    with open(file_name, 'wb') as file:
        pickle.dump(agent, file)


def load_agent(file_name):
    """Load and return the agent at a given file location.

    Parameters
    ----------
    file_name: str.

    Returns
    -------
    agent: AbstractAgent.
    """
    with open(file_name, 'rb') as f:
        agent = pickle.load(f)

    return agent
