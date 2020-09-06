"""Training utility functions."""

import torch
import torch.nn as nn
from torch.distributions import Categorical

from rllib.util.neural_networks.utilities import one_hot_encode


def get_target(model, observation):
    """Get target from observation."""
    if model.model_kind == "dynamics":
        target = observation.next_state
    elif model.model_kind == "rewards":
        target = observation.reward.unsqueeze(-1)
    elif model.model_kind == "termination":
        target = observation.done
    else:
        raise NotImplementedError
    return target


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


def calibration_score(model, observation, bins=10):
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
    state, action = observation.state, observation.action
    target = get_target(model, observation)
    prediction = model(state, action)
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


def sharpness(model, observation):
    """Get prediction sharpness score.

    References
    ----------
    Kuleshov, V., Fenner, N., & Ermon, S. (2018).
    Accurate uncertainties for deep learning using calibrated regression. ICML.
    Equation (10).
    """
    mean, chol_std = model(observation.state, observation.action)
    scale = torch.diagonal(chol_std, dim1=-1, dim2=-2)
    return scale.square().mean()


def model_mse(model, observation):
    """Get model MSE."""
    state, action = observation.state, observation.action
    target = get_target(model, observation)

    mean = model(state, action)[0]
    y = target

    return ((mean - y) ** 2).mean(-1).mean()


def model_loss(model, observation):
    """Get model loss."""
    state, action = observation.state, observation.action
    target = get_target(model, observation)

    prediction = model(state, action)
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
    return loss


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
