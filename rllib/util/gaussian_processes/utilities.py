"""Utilities for GP models."""

import gpytorch
import matplotlib.pyplot as plt
import torch


def plot_gp_inputs(model: gpytorch.models.ExactGP, axes=None):
    """Plot the inputs of the GP model."""
    if axes is None:
        axes = plt.gca()
    inputs = model.train_inputs[0]
    return axes.scatter(torch.atan2(inputs[:, 1], inputs[:, 0]) * 180 / 3.141,
                        inputs[:, 2], c=inputs[:, 3])


def plot_gp(x: torch.Tensor, model: gpytorch.models.GP, num_samples: int) -> None:
    """Plot 1-D GP.

    Parameters
    ----------
    x: points to plot.
    model: GP model.
    num_samples: number of random samples from gp.
    """
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = model(x)
        mean = pred.mean.numpy()
        error = 2 * pred.stddev.numpy()

    # Plot GP
    plt.fill_between(x, mean - error, mean + error, lw=0, alpha=0.4, color='C0')

    # Plot mean
    plt.plot(x, mean, color='C0')

    # Plot samples.
    for _ in range(num_samples):
        plt.plot(x.numpy(), pred.sample().numpy())


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
            pred = gp_model(inputs)
            noise_var = gp_model.likelihood.noise_covar(inputs).diag()
            index = torch.argmax(torch.log(1 + pred.variance / noise_var))

        new_input = inputs[index].unsqueeze(0)
        new_target = targets[index].unsqueeze(-1)

        # Once enabled use this
        # gp_model = gp_model.get_fantasy_model(new_input, new_target)
        _add_data_to_gp(gp_model, new_input, new_target)

        # Remove data from input space
        idx = int(index.item())
        inputs = torch.cat((inputs[:idx], inputs[idx + 1:]), dim=0)
        targets = torch.cat((targets[:idx], targets[idx + 1:]), dim=-1)
