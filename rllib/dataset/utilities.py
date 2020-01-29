"""Utilities for dataset submodule."""

import numpy as np
import torch
from .datatypes import Observation

__all__ = ['init_er_from_environment', 'init_er_from_er', 'init_er_from_rollout',
           'stack_list_of_tuples']


def stack_list_of_tuples(iter_, dtype=None, backend=np):
    """Convert a list of observation tuples to a list of numpy arrays.

    Parameters
    ----------
    iter_: list
        Each entry represents one row in the resulting vectors.
    dtype: type
    backend: Module
        A library that implements `backend.stack`. E.g., numpy or torch.

    Returns
    -------
    *arrays
        One stacked array for each entry in the tuple.
    """
    stacked_generator = (
        backend.stack(tuple(map(lambda x: torch.tensor(np.array(x)), x)))
        for x in zip(*iter_)
    )
    if dtype is not None:
        stacked_generator = (x.astype(dtype) for x in stacked_generator)

    entry_class = iter_[0].__class__
    if entry_class in (tuple, list):
        return entry_class(stacked_generator)
    else:
        return entry_class(*stacked_generator)


def init_er_from_er(target_er, source_er):
    """Initialize an Experience Replay from an Experience Replay.

    Copy all the transitions in the source ER to the target ER.

    Parameters
    ----------
    target_er: Experience Replay
        Experience replay to be filled.
    source_er: Experience Replay
        Experience replay to be used.
    """
    for i in range(len(source_er)):
        observation = source_er[i]
        target_er.append(observation)


def init_er_from_environment(target_er, environment):
    """Initialize an Experience Replay from an Experience Replay.

    Initialize an observation per state in the environment.
    The environment must have discrete states.

    Parameters
    ----------
    target_er: Experience Replay.
        Experience replay to be filled.
    environment: Environment.
        Discrete environment.
    """
    assert environment.num_states is not None

    for state in range(environment.num_states):
        observation = Observation(state, torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0))
        target_er.append(observation)


def init_er_from_rollout(target_er, agent, environment, max_steps=1000):
    """Initialize an Experience Replay from an Experience Replay.

    Initialize an observation per state in the environment.

    Parameters
    ----------
    target_er: Experience Replay.
        Experience replay to be filled.
    agent: Agent.
        Agent to act in environment.
    environment: Environment.
        Discrete environment.
    max_steps: int.
        Maximum number of steps in the environment.
    """
    while not target_er.is_full:
        state = environment.reset()
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = environment.step(action)
            observation = Observation(state=state,
                                      action=action,
                                      reward=reward,
                                      next_state=next_state,
                                      done=done)
            state = next_state

            target_er.append(observation)
            if max_steps <= environment.time:
                break
