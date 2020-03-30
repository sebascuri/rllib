"""Utility functions for training models."""

import pickle

import numpy as np
import torch
import matplotlib.pyplot as plt

from .rollout import rollout_agent
from .logger import Logger
from .utilities import tensor_to_distribution
from rllib.util.gaussian_processes.mlls import exact_mll

from rllib.model.ensemble_model import EnsembleModel
from rllib.model.gp_model import ExactGPModel
from rllib.model.nn_model import NNModel


def _model_loss(model, state, action, next_state):
    mean, cov = model(state, action)
    y = next_state
    if torch.all(cov == 0):  # Deterministic Model
        loss = ((mean - y) ** 2).sum(-1)
    else:  # Probabilistic Model
        delta = (mean - y).unsqueeze(-1)
        loss = (delta.transpose(-2, -1) @ torch.inverse(cov) @ delta).squeeze()
        loss += torch.logdet(cov)
    return loss


def train_nn_step(model, observation, optimizer):
    """Train a Neural Network Model."""
    model.train()
    optimizer.zero_grad()
    loss = _model_loss(model, observation.state, observation.action,
                       observation.next_state).mean()
    loss.backward()
    optimizer.step()

    return loss.item()


def train_ensemble_step(model, observation, mask, optimizer, logger):
    """Train a model ensemble."""
    model.train()
    ensemble_loss = 0
    for i in range(model.num_heads):
        optimizer.zero_grad()
        model.select_head(i)
        loss = (mask[:, i] * _model_loss(
            model, observation.state, observation.action, observation.next_state)
                ).mean()

        loss.backward()
        optimizer.step()
        ensemble_loss += loss.item()
        logger.update(**{f"model-{i}": loss.item()})
    return ensemble_loss


def train_exact_gp_type2mll_step(model, observation, optimizer):
    """Train a GP using type-2 Marginal-Log-Likelihood optimization."""
    optimizer.zero_grad()
    model.training = True

    output = tensor_to_distribution(model(observation.state, observation.action))
    loss = exact_mll(output, observation.next_state.T, model.gp)
    loss.backward()
    optimizer.step()

    model.training = False
    return loss.item()


def train_model(model, train_loader, optimizer, max_iter=100, logger=None):
    """Train a Dynamical Model."""
    if logger is None:
        logger = Logger('model_training')
    for i_epoch in range(max_iter):
        for observation, mask in train_loader:
            if isinstance(model, EnsembleModel):
                model_loss = train_ensemble_step(model, observation, mask, optimizer,
                                                 logger)
            elif isinstance(model, NNModel):
                model_loss = train_nn_step(model, observation, optimizer)
            elif isinstance(model, ExactGPModel):
                model_loss = train_exact_gp_type2mll_step(model, observation, optimizer)
            else:
                raise TypeError("Only Implemented for Ensembles and GP Models.")
            logger.update(model_loss=model_loss)


def train_agent(agent, environment, num_episodes, max_steps, plot_flag=True,
                print_frequency=0, render=False):
    """Train an agent in an environment.

    Parameters
    ----------
    agent: AbstractAgent
    environment: AbstractEnvironment
    num_episodes: int
    max_steps: int
    plot_flag: bool, optional.
    print_frequency: int, optional.
    render: bool, optional.
    """
    agent.train()
    rollout_agent(environment, agent, num_episodes=num_episodes, max_steps=max_steps,
                  print_frequency=print_frequency, render=render)

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
    returns = np.mean(agent.logger.get('environment_return')[-num_episodes:])
    print(f"Test Cumulative Rewards: {returns}")


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
