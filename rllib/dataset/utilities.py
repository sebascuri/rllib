"""Utilities for dataset submodule."""
from itertools import product

import numpy as np
import torch

from .datatypes import Observation


def _cast_to_iter_class(generator, class_):
    if class_ in (tuple, list):
        return class_(generator)
    else:
        return class_(*generator)


def map_observation(func, observation):
    """Map observations through the function func."""
    return Observation(*map(lambda x: func(x), observation))


def map_and_cast(fun, iter_):
    """Map a function on a iterable and recast the resulting generator.

    Parameters
    ----------
    fun : callable
    iter_ : iterable
    """
    generator = map(fun, zip(*iter_))
    return _cast_to_iter_class(generator, iter_[0].__class__)


def average_named_tuple(named_tuple_):
    """Return an averaged named-tuple."""
    return type(named_tuple_)(*map(lambda x: x.mean().item(), named_tuple_))


def average_dataclass(dataclass_):
    """Return an averaged data-class."""
    d = []
    for val in dataclass_:
        d.append(val.mean().item())

    return type(dataclass_)(*d)


def stack_list_of_tuples(iter_, dim=None):
    """Convert a list of observation tuples to a list of numpy arrays.

    Parameters
    ----------
    iter_: list
        Each entry represents one row in the resulting vectors.
    dim: int, optional (default=0).

    Returns
    -------
    *arrays
        One stacked array for each entry in the tuple.
    """
    try:
        if dim is None:
            generator = map(torch.stack, zip(*iter_))
        else:
            generator = map(
                lambda x: torch.stack(
                    x, dim=(dim if x[0].ndim > max(dim, -dim - 1) else -1)
                ),
                zip(*iter_),
            )
        return _cast_to_iter_class(generator, iter_[0].__class__)
    except (TypeError, AttributeError):
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
            *[k.repeat(batch_shape) if k.dim() < 1 else k for k in batch_obs]
        )
        squeezed_obs = Observation(
            *[k.reshape(-1, *k.shape[len(batch_shape) :]) for k in expanded_obs]
        )
        out += [Observation(*k) for k in zip(*squeezed_obs)]

    return out


def concatenate_observations(observation, new_observation):
    """Concatenate observations and return a new observation."""
    return Observation(
        *[
            torch.cat((a, b.unsqueeze(0)), dim=0)
            for a, b in zip(observation, new_observation)
        ]
    )


def gather_trajectories(trajectories, gather_dim=1):
    """Gather parallel trajectories.

    Parameters
    ----------
    trajectories: List[Trajectory].
    gather_dim: int, optional. (default=1).
    """
    batch_trajectories = [stack_list_of_tuples(traj) for traj in trajectories]
    trajectory = Observation(
        *map(
            lambda args: torch.cat(args, dim=gather_dim)
            if args[0].dim() > 1
            else torch.stack(args, -1),
            zip(*batch_trajectories),
        )
    )
    return trajectory


def unstack_observations(observation):
    """Unstack observations in a list."""
    in_dim = observation.reward.shape
    observations = []
    for indexes in product(*map(range, in_dim)):

        def _extract_index(tensor):
            try:
                return tensor[indexes]
            except IndexError:
                return tensor

        observations.append(map_observation(_extract_index, observation))
    return observations


def chunk(array, num_steps):
    """Chunk an array into size of N steps.

    The array of size N x k_1 x ... k_n will be reshaped to be of size
    Batch x num_steps x k_1 x ... k_n.

    Parameters
    ----------
    array: Array.
        Array to reshape.
    num_steps: int.
        Number of steps to chunk the batch.

    Returns
    -------
    array: Array.
        Chunked Array.
    """
    batch_size = array.shape[0] // num_steps
    return array.reshape(batch_size, num_steps, *array.shape[1:])


def d4rl_to_observation(dataset):
    """Transform a d4rl dataset into an observation dataset.

    Parameters
    ----------
    dataset: Dict.
        Dict with dataset.

    Returns
    -------
    observation: Observation
        Dataset in observation format..
    """
    num_points = dataset["observations"].shape[0]
    dim_state = dataset["observations"].shape[1]
    dim_actions = dataset["actions"].shape[1]
    dataset = Observation(
        state=dataset["observations"].reshape(num_points,1,dim_state),
        action=dataset["actions"].reshape(num_points,1,dim_actions),
        reward=dataset["rewards"].reshape(num_points,1,1),
        next_state=dataset["next_observations"].reshape(num_points,1,dim_state),
        done=dataset["terminals"].reshape(num_points,1,1),
        log_prob_action=dataset["infos/action_log_probs"].reshape(num_points,1,1),
    ).to_torch()
    return dataset


def split_observations_by_done(observation):
    """Split an observation into a list of observations."""
    end_indexes = torch.where(observation.done)[0]
    start_indexes = torch.cat((torch.tensor([0]), end_indexes + 1))[:-1]
    observations = []

    def _extract_index(tensor, start_index_, end_index_):
        try:
            return tensor[start_index_ : end_index_ + 1]
        except IndexError:
            return tensor

    for start_index, end_index in zip(start_indexes, end_indexes):
        observations.append(
            map_observation(
                lambda x: _extract_index(x, start_index, end_index), observation
            )
        )

    return observations


def drop_last(observation, k):
    """Drop last k indexes from observation."""
    #

    def _extract_index(tensor):
        try:
            return tensor[:-k]
        except IndexError:
            return tensor

    return map_observation(_extract_index, observation)


def _observation_to_num_steps_with_repeat(observation, num_steps):
    """Do something."""
    #

    def _safe_repeat(tensor):
        try:
            shape = torch.tensor(tensor.shape)
            shape[0] = (shape[0] - num_steps) + 1
            shape[1] = num_steps
            out = torch.zeros(*shape)
            for i in range(num_steps):
                first_idx = i
                last_idx = first_idx + shape[0]
                out[:, i, :] = tensor[first_idx:last_idx, 0, :]
            return out
        except IndexError:
            return tensor

    return map_observation(_safe_repeat, observation)


def _observation_to_num_steps(observation, num_steps, repeat=False):
    """Get an observation and chunk it into batches of num_steps."""
    if repeat:
        return _observation_to_num_steps_with_repeat(observation, num_steps)
    num_transitions = observation.state.shape[0]
    drop_k = num_transitions % num_steps
    if drop_k > 0:
        # drop last k transitions.
        observation = drop_last(observation, drop_k)

    def _safe_chunk(tensor):
        try:
            return chunk(tensor.squeeze(), num_steps)
        except IndexError:
            return tensor

    return map_observation(_safe_chunk, observation)


def observation_to_num_steps(observation, num_steps, repeat=False):
    """Convert an observation to num_steps."""
    # split into trajectories
    trajectory = split_observations_by_done(observation)

    # convert each trajectory to num step chunks
    chunked_trajectories = trajectory_to_num_steps(trajectory, num_steps, repeat=repeat)

    # gather back trajectories into an observation.
    return merge_observations(chunked_trajectories)


def trajectory_to_num_steps(trajectory, num_steps, repeat=False):
    """Trajectory to num_steps."""
    chunked_observations = []
    for observation in trajectory:
        chunked_observations.append(
            _observation_to_num_steps(observation, num_steps, repeat=repeat)
        )
    return chunked_observations


def merge_observations(trajectory, dim=0):
    """Concatenate observations and return a new observation."""
    observation = trajectory[0]
    for new_observation in trajectory[1:]:
        observation = Observation(
            *[
                torch.cat((a, b), dim=dim) if a.dim() > 0 else a
                for a, b in zip(observation, new_observation)
            ]
        )
    return observation


def flatten_observation(observation):
    """Flatten an observation by reshaping the time coordinates."""
    #

    def _flatten(tensor):
        try:
            return tensor.reshape(-1, tensor.shape[-1])
        except IndexError:
            return tensor

    return map_observation(func=_flatten, observation=observation)
