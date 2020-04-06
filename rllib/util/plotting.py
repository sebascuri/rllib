"""Python Script Template."""
import itertools

import matplotlib.pyplot as plt
import numpy as np
import gpytorch
import torch
from rllib.dataset.utilities import stack_list_of_tuples
from rllib.model.gp_model import ExactGPModel


def moving_average_filter(x, y, horizon):
    """Apply a moving average filter to data x and y.

    This function truncates the data to match the horizon.

    Parameters
    ----------
    x : ndarray
        The time stamps of the data.
    y : ndarray
        The values of the data.
    horizon : int
        The horizon over which to apply the filter.

    Returns
    -------
    x_smooth : ndarray
        A shorter array of x positions for smoothed values.
    y_smooth : ndarray
        The corresponding smoothed values
    """
    horizon = min(horizon, len(y))

    smoothing_weights = np.ones(horizon) / horizon
    x_smooth = x[horizon // 2: -horizon // 2 + 1]
    y_smooth = np.convolve(y, smoothing_weights, 'valid')
    return x_smooth, y_smooth


def combinations(arrays):
    """Return a single array with combinations of parameters.

    Parameters
    ----------
    arrays : list of np.array

    Returns
    -------
    array : np.array
        An array that contains all combinations of the input arrays
    """
    return np.array(np.meshgrid(*arrays)).T.reshape(-1, len(arrays))


def linearly_spaced_combinations(bounds, num_entries):
    """
    Return 2-D array with all linearly spaced combinations with the bounds.

    Parameters
    ----------
    bounds : sequence of tuples
        The bounds for the variables, [(x1_min, x1_max), (x2_min, x2_max), ...]
    num_entries : integer or array_like
        Number of samples to use for every dimension. Can be a constant if
        the same number should be used for all, or an array to fine-tune
        precision. Total number of data points is num_samples ** len(bounds).

    Returns
    -------
    combinations : 2-d array
        A 2-d arrray. If d = len(bounds) and l = prod(num_samples) then it
        is of size l x d, that is, every row contains one combination of
        inputs.
    """
    bounds = np.atleast_2d(bounds)
    num_vars = len(bounds)
    num_entries = np.broadcast_to(num_entries, num_vars)

    # Create linearly spaced test inputs
    inputs = [np.linspace(b[0], b[1], n) for b, n in zip(bounds,
                                                         num_entries)]

    # Convert to 2-D array
    return combinations(inputs)


def plot_combinations_as_grid(axis, values, num_entries, bounds=None, **kwargs):
    """Take values from a grid and plot them as an image.

    Takes values generated from `linearly_spaced_combinations`.

    Parameters
    ----------
    axis : matplotlib.axis
    values : ndarray
    num_entries : array_like
        Number of samples to use for every dimension.
        Used for reshaping
    bounds : sequence of tuples
        The bounds for the variables, [(x1_min, x1_max), (x2_min, x2_max), ...]
    kwargs : dict
        Passed to axis.imshow
    """
    kwargs['origin'] = 'lower'
    if bounds is not None:
        kwargs['extent'] = list(itertools.chain(*bounds))
    return axis.imshow(values.reshape(*num_entries).T, **kwargs)


def plot_learning_losses(policy_losses, value_losses, horizon):
    """Plot the losses encountnered during learning.

    Parameters
    ----------
    policy_losses : list or ndarray
    value_losses : list or ndarray
    horizon : int
        Horizon used for smoothing
    """
    t = np.arange(len(policy_losses))

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(15, 5))

    plt.sca(ax1)
    plt.plot(t, policy_losses)
    plt.plot(*moving_average_filter(t, policy_losses, horizon),
             label='smoothed')
    plt.xlabel('Iteration')
    plt.ylabel('Policy loss')
    plt.legend()

    plt.sca(ax2)
    plt.plot(t, value_losses)
    plt.plot(*moving_average_filter(t, value_losses, horizon),
             label='smoothed')
    plt.xlabel('Iteration')
    plt.ylabel('Value loss')
    plt.legend()


def plot_on_grid(function, bounds, num_entries):
    """Plot function values on a grid.

    Parameters
    ----------
    function : callable
    bounds : list
    num_entries : list

    Returns
    -------
    axis
    """
    axis = plt.gca()
    states = linearly_spaced_combinations(bounds, num_entries)
    values = function(torch.tensor(states, dtype=torch.get_default_dtype()))
    values = values.detach().numpy()

    img = plot_combinations_as_grid(axis, values, num_entries, bounds)
    plt.colorbar(img)
    axis.set_xlim(bounds[0])
    axis.set_ylim(bounds[1])
    return axis


def plot_values_and_policy(value_function, policy, bounds, num_entries):
    """Plot the value and policy function over a grid.

    Parameters
    ----------
    value_function : torch.nn.Module
    policy : torch.nn.Module
    bounds : list
    num_entries : list

    Returns
    -------
    ax_value
    ax_policy
    """
    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(15, 10))

    plt.sca(ax1)
    plot_on_grid(value_function, bounds=bounds, num_entries=num_entries)
    plt.title('Learned value function')
    plt.xlabel('Angle[rad]')
    plt.ylabel('Angular velocity [rad/s]')
    plt.axis('tight')

    plt.sca(ax2)
    plot_on_grid(lambda x: policy(x)[0], bounds=bounds, num_entries=num_entries)
    plt.title('Learned policy')
    plt.xlabel('Angle [rad]')
    plt.ylabel('Angular velocity [rad/s]')
    plt.axis('tight')

    return ax1, ax2


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


def plot_pendulum_trajectories(agent):
    """Plot GP inputs and trajectory in a Pendulum environment."""
    model = agent.mppo.dynamical_model.base_model
    trajectory = stack_list_of_tuples(agent.trajectory)
    sim_obs = agent.sim_trajectory

    for transformation in agent.dataset.transformations:
        trajectory = transformation(trajectory)
        sim_obs = transformation(sim_obs)
    if isinstance(model, ExactGPModel):
        fig, axes = plt.subplots(model.dim_state, 3, sharex='col', sharey='row')
    else:
        fig, axes = plt.subplots(model.dim_state, 2, sharex='col', sharey='row')

    for i in range(model.dim_state):
        sin, cos = torch.sin(trajectory.state[:, 0]), torch.cos(trajectory.state[:, 0])
        axes[i, 0].scatter(torch.atan2(sin, cos) * 180 / np.pi, trajectory.state[:, 1],
                           c=trajectory.action[:, 0], cmap='jet', vmin=-1, vmax=1)

        sin = torch.sin(sim_obs.state[:, 0, :, 0])
        cos = torch.cos(sim_obs.state[:, 0, :, 0])
        axes[i, 1].scatter(torch.atan2(sin, cos) * 180 / np.pi,
                           sim_obs.state[:, 0, :, 1],
                           c=sim_obs.action[:, 0, :, 0], cmap='jet', vmin=-1, vmax=1)

        if isinstance(model, ExactGPModel):
            inputs = model.gp[i].train_inputs[0]
            sin, cos = inputs[:, 1], inputs[:, 0]
            axes[i, 2].scatter(torch.atan2(sin, cos) * 180 / np.pi, inputs[:, 2],
                               c=inputs[:, 3], cmap='jet', vmin=-1, vmax=1)

        for j in range(axes.shape[-1]):
            axes[i, j].set_xlim([-180, 180])
            axes[i, j].set_ylim([-15, 15])

    for i in range(2):
        axes[i, 0].set_ylabel('Angular Velocity')

    axes[0, 0].set_title('Last real trajectory.')
    axes[0, 1].set_title('Last sim trajectories.')

    if isinstance(model, ExactGPModel):
        axes[0, 2].set_title('Input Data of GP.')

    for j in range(axes.shape[-1]):
        axes[-1, j].set_xlabel('Angle')
    plt.show()
