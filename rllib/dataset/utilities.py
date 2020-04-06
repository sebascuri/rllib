"""Utilities for dataset submodule."""
import numpy as np
import torch

from .datatypes import Observation


def _cast_to_iter_class(generator, class_):
    if class_ in (tuple, list):
        return class_(generator)
    else:
        return class_(*generator)


def map_and_cast(fun, iter_):
    """Map a function on a iterable and recast the resulting generator.

    Parameters
    ----------
    fun : callable
    iter_ : iterable
    """
    generator = map(fun, zip(*iter_))
    return _cast_to_iter_class(generator, iter_[0].__class__)


def stack_list_of_tuples(iter_):
    """Convert a list of observation tuples to a list of numpy arrays.

    Parameters
    ----------
    iter_: list
        Each entry represents one row in the resulting vectors.

    Returns
    -------
    *arrays
        One stacked array for each entry in the tuple.
    """
    try:
        generator = map(torch.stack, zip(*iter_))
        return _cast_to_iter_class(generator, iter_[0].__class__)
    except TypeError:
        generator = map(np.stack, zip(*iter_))
        return _cast_to_iter_class(generator, iter_[0].__class__)


def bootstrap_trajectory(trajectory, bootstraps):
    """Bootstrap a trajectory into `bootstrap' different i.i.d. trajectories."""
    num_points = len(trajectory)
    new_trajectories = []
    for _ in range(bootstraps):
        idx = np.random.choice(num_points, num_points, replace=True)
        t = []
        for i in idx:
            t.append(trajectory[i])
        new_trajectories.append(t)

    return new_trajectories


def batch_trajectory_to_single_trajectory(trajectory):
    """Convert a batch trajectory into a single trajectory.

    A batch trajectory contains a list of batch observations, e.g., Observations with
    states that have b x h x dim_states dimensions.

    Return a Trajectory that have just 1 x dim_states.
    """
    batch_shape = trajectory[0].state.shape[:-1]
    out = []
    for batch_obs in trajectory:
        expanded_obs = Observation(
            *[k.repeat(batch_shape) if k.dim() < 1 else k for k in batch_obs])
        squeezed_obs = Observation(
            *[k.reshape(-1, *k.shape[len(batch_shape):]) for k in expanded_obs]
        )
        out += [Observation(*k) for k in zip(*squeezed_obs)]

    return out


def concatenate_observations(observation, new_observation):
    """Concatenate observations and return a new observation."""
    return Observation(*[torch.cat((a, b.unsqueeze(0)), dim=0)
                         for a, b in zip(observation, new_observation)])
