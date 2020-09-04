"""Model Learning Functions."""
import gpytorch.settings
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from rllib.dataset.datatypes import Observation
from rllib.model import EnsembleModel, ExactGPModel, NNModel
from rllib.model.utilities import PredictionStrategy
from rllib.util.early_stopping import EarlyStopping
from rllib.util.gaussian_processes.mlls import exact_mll
from rllib.util.logger import Logger
from rllib.util.utilities import tensor_to_distribution

from .utilities import calibration_score, model_loss, model_mse, sharpness


def train_nn_step(model, observation, optimizer, weight=1.0):
    """Train a Neural Network Model."""
    optimizer.zero_grad()
    loss = (weight * model_loss(model, observation)).mean()
    loss.backward()
    optimizer.step()

    return loss


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
    return loss


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
    validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)

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
            logger.update(**{f"{model.model_kind} model-loss": loss.item()})

        for observation, idx, mask in validation_loader:
            observation = Observation(**observation)
            with torch.no_grad():
                mse = model_mse(model, observation).item()
                sharpness_ = sharpness(model, observation).item()
                calibration_score_ = calibration_score(model, observation).item()

            logger.update(
                **{
                    f"{model.model_kind} model-validation-mse": mse,
                    f"{model.model_kind} model-sharpness": sharpness_,
                    f"{model.model_kind} model-calibration_score": calibration_score_,
                }
            )

            early_stopping.update(mse)

        if early_stopping.stop:
            return
        early_stopping.reset(hard=False)  # reset to zero the moving averages.
