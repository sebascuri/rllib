"""Utilities for GP models."""

import gpytorch
import torch
from .exact_gp import SparseGP
from torch.distributions import Bernoulli


def _add_data_to_gp(gp_model, new_inputs, new_targets):
    """Add new data points to an existing GP model.

    Once available, gp_model.get_fantasy_model should be preferred over this.
    """
    inputs = torch.cat((gp_model.train_inputs[0], new_inputs), dim=0)
    targets = torch.cat((gp_model.train_targets, new_targets), dim=-1)
    gp_model.set_train_data(inputs, targets, strict=False)
    # return gp_model.get_fantasy_model(inputs, targets)


def add_data_to_gp(gp_model, new_inputs=None, new_targets=None,
                   max_num: int = None, weight_function=None):
    """Summarize the GP model with a fixed number of data points inplace.

    Parameters
    ----------
    gp_model : gpytorch.models.ExactGPModel
    new_inputs : torch.Tensor
        New data points to add to the GP. Ignored if new_targets is None.
    new_targets : torch.Tensor
        New target data points to add to the GP. Ignored if new_inputs is None.
    max_num : int
        The maximum number of data points to use.
    weight_function: Callable[[torch.Tensor], torch.Tensor]
        weighing_function that computes the weight of each input.
    """
    old_inputs = gp_model.train_inputs[0]
    old_targets = gp_model.train_targets

    # For convenience, define new data as empty arrays
    if new_inputs is None or new_targets is None:
        new_inputs = torch.empty((0, old_inputs.shape[-1]))
        new_targets = torch.empty((old_targets.shape[0], 0))

    # Can add all data points directly
    if max_num is None or len(old_inputs) + len(new_inputs) <= max_num:
        return _add_data_to_gp(gp_model, new_inputs, new_targets)

    # Remove all data points but one
    gp_model.set_train_data(old_inputs[0].unsqueeze(0), old_targets[0].unsqueeze(-1),
                            strict=False)
    inputs = torch.cat((old_inputs[1:], new_inputs), dim=0)
    targets = torch.cat((old_targets[1:], new_targets), dim=-1)

    gp_model.eval()
    for _ in range(max_num - 1):
        with gpytorch.settings.fast_pred_var():
            # The set function to maximize is f_s = log det (I + \lambda^2 K_s).
            # Greedy selection resorts to sequentially selecting the index that solves
            # i^\star = \arg max_i log (1 + \lambda^2 K_(i|s))
            # This is equivalent to doing \arg max_i (1 + \lambda^2 K_(i|s)) and to
            # i^\star = \arg max_i K_(i|s).
            # Hence, the point with greater predictive variance is selected.
            # torch.log(1 + pred.variance)
            pred_var = gp_model(inputs).variance
            if weight_function is not None:
                pred_var = torch.log(1 + pred_var) * weight_function(inputs)
            index = torch.argmax(pred_var)

        new_input = inputs[index].unsqueeze(0)
        new_target = targets[index].unsqueeze(-1)

        # Once enabled use this
        # gp_model = gp_model.get_fantasy_model(new_input, new_target)
        _add_data_to_gp(gp_model, new_input, new_target)

        # Remove data from input space
        idx = int(index.item())
        inputs = torch.cat((inputs[:idx], inputs[idx + 1:]), dim=0)
        targets = torch.cat((targets[:idx], targets[idx + 1:]), dim=-1)


def bkb(gp_model, new_inputs, new_targets, q_bar=1):
    """Update the GP model using BKB algorithm.

    Parameters
    ----------
    gp_model: ExactGP
        model to update
    new_inputs: torch.Tensor
        Tensor of dimension [N x d_x]
    new_targets: torch.Tensor
        Tensor of dimesnion [N]
    q_bar: float
        float with algorithm parameter.
    """
    gp_model.eval()
    inputs = torch.cat((gp_model.train_inputs[0], new_inputs), dim=0)
    targets = torch.cat((gp_model.train_targets, new_targets), dim=-1)

    # Previous arms
    if isinstance(gp_model, SparseGP):
        ip = torch.cat((gp_model.xu, new_inputs), dim=0)
    else:
        ip = inputs

    # Scaled predictive variance of arms under current model.
    p = q_bar * gp_model(ip).variance / gp_model.likelihood.noise
    q = Bernoulli(probs=p.clamp_(0, 1)).sample()
    idx = torch.where(q == 1)[0]
    if len(idx) == 0:  # the GP has to have at least one point.
        idx = [0]

    if isinstance(gp_model, SparseGP):
        gp_model.set_train_data(inputs, targets, strict=False)
        gp_model.set_inducing_points(ip[idx])
    else:
        gp_model.set_train_data(inputs[idx], targets[idx], strict=False)
