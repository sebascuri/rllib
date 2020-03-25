"""Utility functions for training models."""

import pickle

import numpy as np
import torch
import matplotlib.pyplot as plt

from .rollout import rollout_agent
from .logger import Logger


def _model_loss(model, state, action, next_state):
    mean, cov = model(state, action)
    y = next_state
    if torch.all(cov == 0):
        loss = ((mean - y) ** 2).sum(-1)
    else:
        delta = (mean - y).unsqueeze(-1)
        loss = (delta.transpose(-2, -1) @ torch.inverse(cov) @ delta).squeeze()
        loss += torch.logdet(cov)
    return loss


def train_model(model, train_loader, optimizer, max_iter=100, logger=None):
    """Train a Dynamical Model."""
    if logger is None:
        logger = Logger('model_training')
    for i_epoch in range(max_iter):
        for obs, mask in train_loader:
            ensemble_loss = 0
            for i in range(model.num_heads):
                optimizer.zero_grad()
                model.select_head(i)
                loss = (mask[:, i] * _model_loss(model, obs.state, obs.action,
                                                 obs.next_state)).mean()

                loss.backward()
                optimizer.step()
                ensemble_loss += loss.item()
                logger.update(**{f"model-{i}": loss.item()})
            logger.update(ensemble_loss=ensemble_loss)


def train_agent(agent, environment, num_episodes, max_steps, plot_flag=True):
    """Train an agent in an environment.

    Parameters
    ----------
    agent: AbstractAgent
    environment: AbstractEnvironment
    num_episodes: int
    max_steps: int
    plot_flag: bool
    """
    agent.train()
    rollout_agent(environment, agent, num_episodes=num_episodes, max_steps=max_steps)

    if plot_flag:
        for key in agent.logger.keys():
            plt.plot(agent.logger.get(key))
            plt.xlabel("Episode")
            plt.ylabel(" ".join(key.split('_')).title())
            plt.title(f"{agent.name} in {environment.name}")
            plt.show()
    print(agent)


def evaluate_agent(agent, environment, num_episodes, max_steps, render=True):
    """Evaluate an agent in an environment.

    Parameters
    ----------
    agent: AbstractAgent
    environment: AbstractEnvironment
    num_episodes: int
    max_steps: int
    render: bool
    """
    agent.eval()
    rollout_agent(environment, agent, max_steps=max_steps, num_episodes=num_episodes,
                  render=render)
    print(f"Test Rewards: {np.mean(agent.logger.get('rewards')[-num_episodes:])}")


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
