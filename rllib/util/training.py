"""Utility functions for training models."""

import gpytorch.settings
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from rllib.dataset.datatypes import Observation
from rllib.model import EnsembleModel, ExactGPModel, NNModel
from rllib.model.utilities import PredictionStrategy
from rllib.util.early_stopping import EarlyStopping
from rllib.util.gaussian_processes.mlls import exact_mll

from .logger import Logger
from .rollout import rollout_agent
from .utilities import tensor_to_distribution


def _get_target(model, observation):
    if model.model_kind == "dynamics":
        target = observation.next_state
    elif model.model_kind == "rewards":
        target = observation.reward.unsqueeze(-1)
    elif model.model_kind == "termination":
        target = observation.done
    else:
        raise NotImplementedError
    return target


def _model_mse(model, observation):
    state, action = observation.state, observation.action
    target = _get_target(model, observation)

    mean = model(state, action)[0]
    y = target

    return ((mean - y) ** 2).sum(-1).mean().item()


def _model_loss(model, observation):
    state, action = observation.state, observation.action
    target = _get_target(model, observation)

    prediction = model(state, action)
    if len(prediction) == 1:  # Cross entropy loss.
        return nn.CrossEntropyLoss(reduction="none")(prediction[0], target)

    mean, scale_tril = prediction[0], prediction[1]
    y = target
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


def train_nn_step(model, observation, optimizer, weight=1.0):
    """Train a Neural Network Model."""
    optimizer.zero_grad()
    loss = (weight * _model_loss(model, observation)).mean()
    loss.backward()
    optimizer.step()

    return loss.item()


def train_ensemble_step(model, observation, mask, optimizer, logger):
    """Train a model ensemble."""
    ensemble_loss = 0

    model_list = list(range(model.num_heads))
    np.random.shuffle(model_list)
    with PredictionStrategy(model, prediction_strategy="set_head"):
        for i in model_list:
            model.set_head(i)
            loss = train_nn_step(model, observation, optimizer, weight=mask[:, i])
            ensemble_loss += loss / model.num_heads
            logger.update(**{f"{model.model_kind} model-{i}": loss})

    return ensemble_loss


def train_exact_gp_type2mll_step(model, observation, optimizer):
    """Train a GP using type-2 Marginal-Log-Likelihood optimization."""
    optimizer.zero_grad()
    output = tensor_to_distribution(
        model(observation.state[:, 0], observation.action[:, 0])
    )
    with gpytorch.settings.fast_pred_var():
        val = torch.stack(tuple([gp.train_targets for gp in model.gp]), 0)
        loss = exact_mll(output, val, model.gp)
    loss.backward()
    optimizer.step()
    model.eval()
    return loss.item()


def train_model(
    model,
    train_set,
    optimizer,
    batch_size=100,
    max_iter=100,
    epsilon=0.1,
    logger=None,
    validation_set=None,
):
    """Train a Predictive Model.

    Parameters
    ----------
    model: AbstractModel.
        Predictive model to optimize.
    train_set: ExperienceReplay.
        Dataset to train with.
    optimizer: Optimizer.
        Optimizer to call for the model.
    batch_size: int (default=1000).
        Batch size to iterate through.
    max_iter: int (default = 100).
        Maximum number of epochs.
    epsilon: float.
        Early stopping parameter. If epoch loss is > (1 + epsilon) of minimum loss the
        optimization process stops.
    logger: Logger, optional.
        Progress logger.
    validation_set: ExperienceReplay, optional.
        Dataset to validate with.
    """
    if logger is None:
        logger = Logger(f"{model.name}_training")
    if validation_set is None:
        validation_set = train_set

    model.train()
    early_stopping = EarlyStopping(epsilon, non_decrease_iter=5)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    for _ in tqdm(range(max_iter)):
        for observation, idx, mask in train_loader:
            observation = Observation(**observation)
            if isinstance(model, EnsembleModel):
                loss = train_ensemble_step(model, observation, mask, optimizer, logger)
            elif isinstance(model, NNModel):
                loss = train_nn_step(model, observation, optimizer)
            elif isinstance(model, ExactGPModel):
                loss = train_exact_gp_type2mll_step(model, observation, optimizer)
            else:
                raise TypeError("Only Implemented for Ensembles and GP Models.")

            with torch.no_grad():
                observation, _, _ = validation_set.sample_batch(batch_size)
                mse = _model_mse(model, observation)

            logger.update(**{f"{model.model_kind} model-loss": loss})
            logger.update(**{f"{model.model_kind} model-validation-mse": mse})

            early_stopping.update(mse)

        if early_stopping.stop:
            return
        early_stopping.reset(hard=False)  # reset to zero the moving averages.


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
