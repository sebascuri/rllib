"""Model Learning Functions."""
import gpytorch.settings
import numpy as np
import torch
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


def train_ensemble_step(model, observation, optimizer, mask):
    """Train a model ensemble."""
    ensemble_loss = 0

    model_list = list(range(model.num_heads))
    np.random.shuffle(model_list)
    with PredictionStrategy(model, prediction_strategy="set_head"):
        for i in model_list:
            model.set_head(i)
            loss = train_nn_step(model, observation, optimizer, weight=mask[:, i])
            ensemble_loss += loss / model.num_heads

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


def _train_model_step(model, observation, optimizer, mask, logger):
    if not isinstance(observation, Observation):
        observation = Observation(**observation)
    observation.action = observation.action[..., : model.dim_action[0]]
    if isinstance(model, EnsembleModel):
        loss = train_ensemble_step(model, observation, optimizer, mask)
    elif isinstance(model, NNModel):
        loss = train_nn_step(model, observation, optimizer)
    elif isinstance(model, ExactGPModel):
        loss = train_exact_gp_type2mll_step(model, observation, optimizer)
    else:
        raise TypeError("Only Implemented for Ensembles and GP Models.")
    logger.update(**{f"{model.model_kind[:3]}-loss": loss.item()})


def _validate_model_step(model, observation, logger):
    if not isinstance(observation, Observation):
        observation = Observation(**observation)
    observation.action = observation.action[..., : model.dim_action[0]]

    mse = model_mse(model, observation).item()
    sharpness_ = sharpness(model, observation).item()
    calibration_score_ = calibration_score(model, observation).item()

    logger.update(
        **{
            f"{model.model_kind[:3]}-val-mse": mse,
            f"{model.model_kind[:3]}-sharp": sharpness_,
            f"{model.model_kind[:3]}-calib": calibration_score_,
        }
    )
    return mse


def train_model(
    model,
    train_set,
    optimizer,
    batch_size=100,
    num_epochs=None,
    max_iter=100,
    epsilon=0.1,
    non_decrease_iter=float("inf"),
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
    num_epochs: int, optional.
    max_iter: int (default = 100).
        Maximum number of epochs.
    epsilon: float.
        Early stopping parameter. If epoch loss is > (1 + epsilon) of minimum loss the
        optimization process stops.
    non_decrease_iter: int, optional.
        Early stopping parameter. If epoch loss does not decrease for consecutive
        non_decrease_iter, the optimization process stops.
    logger: Logger, optional.
        Progress logger.
    validation_set: ExperienceReplay, optional.
        Dataset to validate with.
    """
    if logger is None:
        logger = Logger(f"{model.name}_training", tensorboard=True)
    if validation_set is None:
        validation_set = train_set

    data_size = len(train_set)
    if num_epochs is not None:
        max_iter = data_size * num_epochs // batch_size
        non_decrease_iter = data_size * non_decrease_iter
    model.train()
    early_stopping = EarlyStopping(epsilon, non_decrease_iter=non_decrease_iter)

    for _ in tqdm(range(max_iter)):
        observation, idx, mask = train_set.sample_batch(batch_size)
        _train_model_step(model, observation, optimizer, mask, logger)

        observation, idx, mask = validation_set.sample_batch(batch_size)
        with torch.no_grad():
            mse = _validate_model_step(model, observation, logger)
        early_stopping.update(mse)

        if early_stopping.stop:
            return


def calibrate_model(
    model,
    calibration_set,
    max_iter=100,
    epsilon=0.0001,
    temperature_range=(0.1, 100.0),
    logger=None,
):
    """Calibrate a model by scaling the temperature.

    First, find a suitable temperature by logarithmic search (increasing or decreasing).
    Then, find a reasonable temperature by binary search.
    """
    if logger is None:
        logger = Logger(f"{model.name}_calibration")

    observation = calibration_set.all_data
    observation.action = observation.action[..., : model.dim_action[0]]

    with torch.no_grad():
        initial_score = calibration_score(model, observation).item()
    initial_temperature = model.temperature

    # Increase temperature.
    model.temperature = initial_temperature.clone()
    score, temperature = initial_score, initial_temperature.clone()
    for _ in range(max_iter):
        if model.temperature > 2 * temperature_range[1]:
            break
        model.temperature *= 2
        with torch.no_grad():
            new_score = calibration_score(model, observation).item()
        if new_score > score:
            break
        score, temperature = new_score, model.temperature.clone()
    max_score, max_temperature = score, temperature

    # Decrease temperature.
    model.temperature = initial_temperature.clone()
    score, temperature = initial_score, initial_temperature.clone()
    for _ in range(max_iter):
        if model.temperature < temperature_range[0] / 2:
            break
        model.temperature /= 2
        with torch.no_grad():
            new_score = calibration_score(model, observation).item()
        if new_score > score:
            break
        score, temperature = new_score, model.temperature.clone()
    min_score, min_temperature = score, temperature

    if max_score < min_score:
        score, temperature = max_score, max_temperature
    else:
        score, temperature = min_score, min_temperature

    # Binary search:
    min_temperature, max_temperature = temperature / 2, 2 * temperature
    with torch.no_grad():
        model.temperature = max_temperature
        max_score = calibration_score(model, observation).item()
        model.temperature = min_temperature
        min_score = calibration_score(model, observation).item()

    if min_score > max_score:
        max_score, min_score = min_score, max_score
        max_temperature, min_temperature = min_temperature, max_temperature

    for _ in range(max_iter):
        if max_score - min_score < epsilon:
            break

        if score < max_score:
            max_score, max_temperature = score, temperature.clone()
        else:
            min_score, min_temperature = score, temperature.clone()

        if min_score > max_score:
            max_score, min_score = min_score, max_score
            max_temperature, min_temperature = min_temperature, max_temperature

        temperature = torch.exp(
            0.5 * (torch.log(min_temperature) + torch.log(max_temperature))
        )
        model.temperature = temperature.clone().clamp(*temperature_range)
        with torch.no_grad():
            score = calibration_score(model, observation).item()
    sharpness_ = sharpness(model, observation).item()

    logger.update(
        **{
            f"{model.model_kind[:3]}-temperature": model.temperature.item(),
            f"{model.model_kind[:3]}-post-sharp": sharpness_,
            f"{model.model_kind[:3]}-post-calib": score,
        }
    )


def evaluate_model(model, observation, logger=None):
    """Train a Predictive Model.

    Parameters
    ----------
    model: AbstractModel.
        Predictive model to evaluate.
    observation: Observation.
        Observation to evaluate..
    logger: Logger, optional.
        Progress logger.
    """
    if logger is None:
        logger = Logger(f"{model.name}_evaluation")

    model.eval()

    with torch.no_grad():
        loss = model_loss(model, observation).mean().item()
        mse = model_mse(model, observation).item()
        sharpness_ = sharpness(model, observation).item()
        calibration_score_ = calibration_score(model, observation).item()

        logger.update(
            **{
                f"{model.model_kind[:3]}-eval-loss": loss,
                f"{model.model_kind[:3]}-eval-mse": mse,
                f"{model.model_kind[:3]}-eval-sharp": sharpness_,
                f"{model.model_kind[:3]}-eval-calib": calibration_score_,
            }
        )
