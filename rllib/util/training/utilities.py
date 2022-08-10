"""Training utility functions."""

import torch
import torch.nn as nn
from torch.distributions import Categorical

from rllib.util.neural_networks.utilities import one_hot_encode
from rllib.util.utilities import tensor_to_distribution


def get_target(model, observation):
    """Get target from observation."""
    if model.model_kind == "dynamics":
        target = observation.next_state
    elif model.model_kind == "rewards":
        target = observation.reward
    elif model.model_kind == "termination":
        target = observation.done
    else:
        raise NotImplementedError
    return target


def get_prediction(model, observation, dynamical_model=None):
    """Get prediction from a model."""
    state, action = observation.state, observation.action
    is_input_sequence = False
    if state.ndim >= 3:
        is_input_sequence = state.shape[1] > 1
    if (
        dynamical_model is None
        or state.shape[1] == 1
        or not is_input_sequence
        or model.is_rnn
    ):
        # no dynamical model is passed, i.e., just compute one step ahead predictions.
        # state.shape[1] == 1 means that the time coordinate is 1.
        # state.ndim < 3 means that there is no time coordinate.
        # model.is_rnn indicates that it is an rnn.
        prediction = model(state, action)
    else:
        prediction = rollout_predictions(
            dynamical_model=dynamical_model,
            model=model,
            initial_state=state[..., 0, :],
            action_sequence=action.transpose(0, 1),
        )
    return prediction


def gaussian_cdf(x, mean, chol_std):
    """Get cdf of multi-variate gaussian."""
    scale = torch.diagonal(chol_std, dim1=-1, dim2=-2)
    z = (x - mean) / (scale + 1e-6)
    return 0.5 * (1 + torch.erf(z / torch.sqrt(torch.tensor(2.0))))


def calibration_count(target, mean, chol_std, buckets):
    """Get the calibration count of a target for the given buckets."""
    p_hat = gaussian_cdf(target, mean, chol_std).reshape(-1)
    total = len(p_hat)

    count = []
    for p in buckets:
        count.append((p_hat <= p).sum().double() / total)
    count = torch.tensor(count)

    return count


def calibration_score(model, observation, bins=10, dynamical_model=None):
    """Get calibration score of a model.

    References
    ----------
    Gneiting, T., & Raftery, A. E. (2007).
    Strictly proper scoring rules, prediction, and estimation. JASA.

    Brier, G. W. (1950).
    Verification of forecasts expressed in terms of probability. Monthly weather review.

    Kuleshov, V., Fenner, N., & Ermon, S. (2018).
    Accurate uncertainties for deep learning using calibrated regression. ICML.
    Equation (9).
    """
    target = get_target(model, observation)
    prediction = get_prediction(model, observation, dynamical_model)
    return _calibration_score(prediction, target, bins=bins)


def _calibration_score(prediction, target, bins=10):
    if len(prediction) == 1:
        logits = prediction[0]
        probabilities = Categorical(logits=logits).probs
        labels = one_hot_encode(target, num_classes=logits.shape[-1])
        calibration_error = torch.mean((probabilities - labels) ** 2)
    else:
        mean, chol_std = prediction
        buckets = torch.linspace(0, 1, bins + 1)
        count = calibration_count(target, mean, chol_std, buckets)
        calibration_error = torch.sum((buckets - count) ** 2)
    return calibration_error


def sharpness(model, observation, dynamical_model=None):
    """Get prediction sharpness score.

    References
    ----------
    Kuleshov, V., Fenner, N., & Ermon, S. (2018).
    Accurate uncertainties for deep learning using calibrated regression. ICML.
    Equation (10).
    """
    prediction = get_prediction(model, observation, dynamical_model)
    return _sharpness(prediction)


def _sharpness(prediction):
    """TODO: Implement for discrete inputs as entropy."""
    _, chol_std = prediction
    scale = torch.diagonal(chol_std, dim1=-1, dim2=-2)
    return scale.square().mean()


def model_mse(model, observation, dynamical_model=None):
    """Get model MSE."""
    target = get_target(model, observation)
    prediction = get_prediction(model, observation, dynamical_model)
    return _mse(prediction, target)


def _mse(prediction, target):
    return ((prediction[0] - target) ** 2).mean(-1).mean()


def model_loss(model, observation, dynamical_model=None):
    """Get model loss."""
    target = get_target(model, observation)
    prediction = get_prediction(model, observation, dynamical_model)
    return _loss(prediction, target)


def _loss(prediction, target):
    if len(prediction) == 1:  # Cross entropy loss.
        return nn.CrossEntropyLoss(reduction="none")(prediction[0], target)

    mean, scale_tril = prediction[0], prediction[1]
    y = target
    if torch.all(scale_tril == 0):  # Deterministic Model
        loss = ((mean - y) ** 2).mean(-1)
    else:  # Probabilistic Model
        scale_tril_inv = torch.inverse(scale_tril)
        delta = scale_tril_inv @ ((mean - y).unsqueeze(-1))
        loss = (delta.transpose(-2, -1) @ delta).squeeze()

        # log det \Sigma = 2 trace log (scale_tril)
        idx = torch.arange(mean.shape[-1])
        loss += 2 * torch.log(scale_tril[..., idx, idx]).mean(dim=-1).squeeze()

    loss = loss.sum(dim=-1)  # add up time coordinates.
    return loss


def rollout_predictions(dynamical_model, model, initial_state, action_sequence):
    """Rollout a sequence of predictions using a dynamical model."""
    state = initial_state
    predictions = []

    for action in action_sequence:
        predictions.append(model(state, action))

        next_state_distribution = tensor_to_distribution(dynamical_model(state, action))
        if next_state_distribution.has_rsample:
            next_state = next_state_distribution.rsample()
        else:
            next_state = next_state_distribution.sample()

        state = next_state

    if len(predictions[0]) == 1:
        return (torch.stack([k[0] for k in predictions], dim=1),)
    else:
        return (
            torch.stack([k[0] for k in predictions], dim=1),
            torch.stack([k[1] for k in predictions], dim=1),
        )


def get_model_validation_score(model, observation, dynamical_model=None):
    """Get validation score."""
    target = get_target(model, observation)

    model.reset()
    prediction = get_prediction(model, observation, dynamical_model=dynamical_model)

    loss = _loss(prediction, target).mean().item()
    mse = _mse(prediction, target).item()
    sharpness_ = _sharpness(prediction).item()
    calibration = _calibration_score(prediction, target).item()
    return loss, mse, sharpness_, calibration


class Evaluate(object):
    """Context manager for evaluating an agent."""

    def __init__(self, agent):
        self.agent = agent

    def __enter__(self):
        """Set the agent into eval mode."""
        self.agent.eval()

    def __exit__(self, *args):
        """Set the agent into training mode."""
        self.agent.train()
