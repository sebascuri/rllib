"""Utility functions for training models."""

import gpytorch.settings
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from rllib.model.ensemble_model import EnsembleModel
from rllib.model.gp_model import ExactGPModel
from rllib.model.nn_model import NNModel
from rllib.util.gaussian_processes.mlls import exact_mll

from .logger import Logger
from .rollout import rollout_agent
from .utilities import tensor_to_distribution


def _model_mse(model, state, action, next_state):
    mean = model(state, action)[0]
    y = next_state
    return ((mean - y) ** 2).sum(-1)


def _model_loss(model, state, action, next_state):
    pred_next_state = model(state, action)
    mean, scale_tril = pred_next_state[0], pred_next_state[1]
    y = next_state
    if torch.all(scale_tril == 0):  # Deterministic Model
        loss = ((mean - y) ** 2).sum(-1)
    else:  # Probabilistic Model

        scale_tril_inv = torch.inverse(scale_tril)
        delta = scale_tril_inv @ ((mean - y).unsqueeze(-1))
        loss = (delta.transpose(-2, -1) @ delta).squeeze()

        # log det \Sigma = 2 trace log (scale_tril)
        idx = torch.arange(mean.shape[-1])
        loss += 2 * torch.log(scale_tril[..., idx, idx]).sum(dim=-1).squeeze()
    return loss


def train_nn_step(model, observation, optimizer):
    """Train a Neural Network Model."""
    model.train()
    optimizer.zero_grad()
    loss = _model_loss(
        model, observation.state, observation.action, observation.next_state
    ).mean()
    loss.backward()
    optimizer.step()

    return loss.item()


def train_ensemble_step(model, observation, mask, optimizer, logger):
    """Train a model ensemble."""
    model.train()
    ensemble_loss = 0
    for i in range(model.num_heads):
        optimizer.zero_grad()
        model.set_head(i)
        model.set_prediction_strategy("set_head")
        loss = (
            mask[:, i]
            * _model_loss(
                model, observation.state, observation.action, observation.next_state
            )
        ).mean()
        with torch.no_grad():
            mse = _model_mse(
                model, observation.state, observation.action, observation.next_state
            ).mean()
        loss.backward()
        optimizer.step()
        ensemble_loss += loss.item()
        logger.update(**{f"model-{i}": loss.item()})
        logger.update(**{f"model-mse-{i}": mse.item()})

    return ensemble_loss


def train_exact_gp_type2mll_step(model, observation, optimizer):
    """Train a GP using type-2 Marginal-Log-Likelihood optimization."""
    optimizer.zero_grad()
    # model.training = True
    model.train()
    output = tensor_to_distribution(
        model(observation.state[:, 0], observation.action[:, 0])
    )
    with gpytorch.settings.fast_pred_var():
        val = torch.stack(tuple([gp.train_targets for gp in model.gp]), 0)
        loss = exact_mll(output, val, model.gp)
        # loss = exact_mll(output, observation.next_state[:, 0].T, model.gp)
    loss.backward()
    optimizer.step()

    # model.training = False
    model.eval()
    return loss.item()


def train_model(model, train_loader, optimizer, max_iter=100, logger=None):
    """Train a Dynamical Model."""
    if logger is None:
        logger = Logger(f"{model.name}_training")
    for i_epoch in tqdm(range(max_iter)):
        for observation, idx, mask in train_loader:
            if isinstance(model, EnsembleModel):
                current_head = model.get_head()
                current_pred = model.get_prediction_strategy()

                model_loss = train_ensemble_step(
                    model, observation, mask, optimizer, logger
                )
                model.set_head(current_head)
                model.set_prediction_strategy(current_pred)

            elif isinstance(model, NNModel):
                model_loss = train_nn_step(model, observation, optimizer)
            elif isinstance(model, ExactGPModel):
                model_loss = train_exact_gp_type2mll_step(model, observation, optimizer)
            else:
                raise TypeError("Only Implemented for Ensembles and GP Models.")
            logger.update(model_loss=model_loss)


def train_agent(
    agent,
    environment,
    num_episodes,
    max_steps,
    plot_flag=True,
    print_frequency=0,
    plot_frequency=1,
    save_milestones=None,
    render=False,
    plot_callbacks=None,
):
    """Train an agent in an environment.

    Parameters
    ----------
    agent: AbstractAgent
    environment: AbstractEnvironment
    num_episodes: int
    max_steps: int
    plot_flag: bool, optional.
    print_frequency: int, optional.
    plot_frequency: int
    save_milestones: List[int], optional.
        List with episodes in which to save the agent.
    render: bool, optional.
    plot_callbacks: list, optional.

    """
    agent.train()
    rollout_agent(
        environment,
        agent,
        num_episodes=num_episodes,
        max_steps=max_steps,
        print_frequency=print_frequency,
        plot_frequency=plot_frequency,
        save_milestones=save_milestones,
        render=render,
        plot_callbacks=plot_callbacks,
    )

    if plot_flag:
        for key in agent.logger.keys:
            plt.plot(agent.logger.get(key))
            plt.xlabel("Episode")
            plt.ylabel(" ".join(key.split("_")).title())
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
    rollout_agent(
        environment,
        agent,
        max_steps=max_steps,
        num_episodes=num_episodes,
        render=render,
    )
    returns = np.mean(agent.logger.get("environment_return")[-num_episodes:])
    print(f"Test Cumulative Rewards: {returns}")
