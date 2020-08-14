"""Utilities for GP models."""

import gpytorch
import torch
from torch.distributions import Bernoulli


def add_data_to_gp(gp_model, new_inputs, new_targets):
    """Add new data points to an existing GP model.

    Once available, gp_model.get_fantasy_model should be preferred over this.
    """
    inputs = torch.cat((gp_model.train_inputs[0], new_inputs), dim=0)
    targets = torch.cat((gp_model.train_targets, new_targets), dim=-1)
    gp_model.set_train_data(inputs, targets, strict=False)

    # TODO: return gp_model.get_fantasy_model(inputs, targets)


def summarize_gp(gp_model, max_num_points=None, weight_function=None):
    """Summarize the GP model with a fixed number of data points inplace.

    Parameters
    ----------
    gp_model : gpytorch.models.ExactGPModel
    max_num_points : int
        The maximum number of data points to use.
    weight_function: Callable[[torch.Tensor], torch.Tensor]
        weighing_function that computes the weight of each input.
    """
    inputs = gp_model.train_inputs[0]
    targets = gp_model.train_targets

    # Can add all data points directly
    if max_num_points is None or len(inputs) <= max_num_points:
        return

    # Remove all data points but one
    gp_model.set_train_data(
        inputs[0].unsqueeze(0), targets[0].unsqueeze(-1), strict=False
    )
    gp_model.eval()
    for _ in range(max_num_points - 1):
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
        add_data_to_gp(gp_model, new_input, new_target)

        # Remove data from input space
        idx = int(index.item())
        inputs = torch.cat((inputs[:idx], inputs[idx + 1 :]), dim=0)
        targets = torch.cat((targets[:idx], targets[idx + 1 :]), dim=-1)


def bkb(gp_model, inducing_points, q_bar=1):
    """Update the GP model using BKB algorithm.

    Parameters
    ----------
    gp_model: ExactGP
        model to update
    inducing_points: torch.Tensor
        Tensor of dimension [N x d_x]
    q_bar: float
        float with algorithm parameter.
    """
    gp_model.eval()

    # Scaled predictive variance of arms under current model.
    p = q_bar * gp_model(inducing_points).variance / (gp_model.likelihood.noise)
    q = Bernoulli(probs=p.clamp_(0, 1)).sample()
    idx = torch.where(q == 1)[0]
    if len(idx) == 0:  # the GP has to have at least one point.
        idx = [0]

    return inducing_points[idx]
