"""Utility functions for training models."""

import pickle

import numpy as np
import torch
import matplotlib.pyplot as plt

from .rollout import rollout_agent
from .logger import Logger


def _model_loss(model, state, action, next_state):
    mean, cov = model(state, action)
    y_pred = mean
    y = next_state
    if torch.all(cov == 0):
        loss = torch.mean((y_pred - y) ** 2)
    else:
        loss = ((mean - y) @ torch.inverse(cov) @ (mean - y).T).mean()
        loss += torch.mean(torch.logdet(cov))
    return loss


def train_model(model, train_loader, optimizer, max_iter=100, eps=1e-6,
                convergence_horizon=10, print_flag=False):
    """Train a Dynamical Model."""
    logger = Logger('mean')
    for i_epoch in range(max_iter):
        logger.start_episode()
        for obs in train_loader:
            optimizer.zero_grad()

            loss = _model_loss(model, obs.state, obs.action, obs.next_state)
            loss.backward()

            optimizer.step()
            logger.append(loss.item())

        logger.end_episode()

        episode_loss = logger.episode_log
        if print_flag:
            print(f"""Epoch {i_epoch}/{max_iter}
                  Train Loss: {episode_loss[-1]:.2f}.""")

        if i_epoch > 2 * convergence_horizon and np.abs(
                np.mean(episode_loss[-2 * convergence_horizon: -convergence_horizon]) -
                np.mean(episode_loss[-convergence_horizon:])
        ):
            break
    return logger


def train_agent(agent, environment, num_episodes, max_steps):
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


def evaluate_agent(agent, environment, num_episodes, max_steps):
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
