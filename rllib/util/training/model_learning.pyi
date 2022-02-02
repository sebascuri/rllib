"""Utility functions for training models."""
from typing import Optional, Tuple, Union

import torch.nn as nn
from torch import Tensor
from torch.optim.optimizer import Optimizer

from rllib.dataset.datatypes import Observation
from rllib.dataset.experience_replay import ExperienceReplay
from rllib.model.abstract_model import AbstractModel
from rllib.model.ensemble_model import EnsembleModel
from rllib.model.independent_ensemble_model import IndependentEnsembleModel
from rllib.model.gp_model import ExactGPModel
from rllib.model.nn_model import NNModel
from rllib.util.logger import Logger

def train_nn_step(
    model: NNModel,
    observation: Observation,
    optimizer: Optimizer,
    weight: Union[Tensor, float] = ...,
    dynamical_model: Optional[AbstractModel] = ...,
) -> Tensor: ...
def train_ensemble_step(
    model: Union[EnsembleModel, IndependentEnsembleModel],
    observation: Observation,
    optimizer: Optimizer,
    mask: Tensor,
    dynamical_model: Optional[AbstractModel] = ...,
) -> Tensor: ...
def train_exact_gp_type2mll_step(
    model: ExactGPModel, observation: Observation, optimizer: Optimizer
) -> Tensor: ...
def train_model(
    model: AbstractModel,
    train_set: ExperienceReplay,
    optimizer: Optimizer,
    batch_size: int = ...,
    num_epochs: Optional[int] = ...,
    max_iter: int = ...,
    min_iter: int = ...,
    epsilon: float = ...,
    non_decrease_iter: int = ...,
    logger: Optional[Logger] = ...,
    validation_set: Optional[ExperienceReplay] = ...,
    dynamical_model: Optional[AbstractModel] = ...,
) -> None: ...
def calibrate_model(
    model: AbstractModel,
    train_set: ExperienceReplay,
    max_iter: int = ...,
    epsilon: float = ...,
    temperature_range: Tuple[float, float] = ...,
    logger: Optional[Logger] = ...,
    dynamical_model: Optional[AbstractModel] = ...,
) -> None: ...
def evaluate_model(
    model: AbstractModel,
    observation: Observation,
    logger: Optional[Logger] = ...,
    dynamical_model: Optional[AbstractModel] = ...,
) -> None: ...
