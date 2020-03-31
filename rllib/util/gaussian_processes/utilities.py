"""Utilities for GP models."""

import gpytorch
import torch


def _add_data_to_gp(gp_model: gpytorch.models.ExactGP, new_inputs, new_targets):
    """Add new data points to an existing GP model.

    Once available, gp_model.get_fantasy_model should be preferred over this.
    """
    inputs = torch.cat((gp_model.train_inputs[0], new_inputs), dim=0)
    targets = torch.cat((gp_model.train_targets, new_targets), dim=-1)
    gp_model.set_train_data(inputs, targets, strict=False)
    # return gp_model.get_fantasy_model(inputs, targets)


def add_data_to_gp(gp_model, new_inputs=None, new_targets=None,
                   max_num: int = int(1e9)):
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
    """
    old_inputs = gp_model.train_inputs[0]
    old_targets = gp_model.train_targets

    # For convenience, define new data as empty arrays
    if new_inputs is None or new_targets is None:
        new_inputs = torch.empty((0, old_inputs.shape[-1]))
        new_targets = torch.empty((old_targets.shape[0], 0))

    # Can add all data points directly
    if len(old_inputs) + len(new_inputs) <= max_num:
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
            index = torch.argmax(gp_model(inputs).variance)

        new_input = inputs[index].unsqueeze(0)
        new_target = targets[index].unsqueeze(-1)

        # Once enabled use this
        # gp_model = gp_model.get_fantasy_model(new_input, new_target)
        _add_data_to_gp(gp_model, new_input, new_target)

        # Remove data from input space
        idx = int(index.item())
        inputs = torch.cat((inputs[:idx], inputs[idx + 1:]), dim=0)
        targets = torch.cat((targets[:idx], targets[idx + 1:]), dim=-1)
