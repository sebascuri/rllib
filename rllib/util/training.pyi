"""Utility functions for training models."""
from typing import Callable, List, Optional, Union

import torch.nn as nn
from torch import Tensor
from torch.optim.optimizer import Optimizer

from rllib.agent import AbstractAgent
from rllib.dataset.datatypes import Observation
from rllib.dataset.experience_replay import ExperienceReplay
from rllib.environment import AbstractEnvironment
from rllib.model.abstract_model import AbstractModel
from rllib.model.ensemble_model import EnsembleModel
from rllib.model.gp_model import ExactGPModel
from rllib.model.nn_model import NNModel

from .logger import Logger

def _model_mse(
    model: AbstractModel, state: Tensor, action: Tensor, next_state: Tensor
) -> Tensor: ...
def _model_loss(
    model: AbstractModel, state: Tensor, action: Tensor, next_state: Tensor
) -> Tensor: ...
def train_nn_step(
    model: NNModel,
    observation: Observation,
    optimizer: Optimizer,
    weight: Union[Tensor, float] = ...,
) -> float: ...
def train_ensemble_step(
    model: EnsembleModel,
    observation: Observation,
    mask: Tensor,
    optimizer: Optimizer,
    logger: Logger,
) -> float: ...
def train_exact_gp_type2mll_step(
    model: ExactGPModel, observation: Observation, optimizer: Optimizer
) -> float: ...
def train_model(
    model: AbstractModel,
    train_set: ExperienceReplay,
    optimizer: Optimizer,
    batch_size: int = ...,
    max_iter: int = ...,
    epsilon: float = ...,
    logger: Optional[Logger] = ...,
    validation_set: Optional[ExperienceReplay] = ...,
) -> None: ...
def train_agent(
    agent: AbstractAgent,
    environment: AbstractEnvironment,
    num_episodes: int,
    max_steps: int,
    plot_flag: bool = ...,
    print_frequency: int = ...,
    plot_frequency: int = ...,
    save_milestones: Optional[List[int]] = ...,
    render: bool = ...,
    plot_callbacks: Optional[List[Callable[[AbstractAgent, int], None]]] = ...,
) -> None: ...
def evaluate_agent(
    agent: AbstractAgent,
    environment: AbstractEnvironment,
    num_episodes: int,
    max_steps: int,
    render: bool = ...,
) -> None: ...
