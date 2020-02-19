"""Utilities for GP models."""

import torch
import gpytorch
import matplotlib.pyplot as plt


def plot_gp(x, model, num_samples):
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
