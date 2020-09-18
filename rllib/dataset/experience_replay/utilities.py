"""Utilities for experience replay submodule."""

import torch

from rllib.dataset.datatypes import Observation


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
        observation, idx, weight = source_er[i]
        target_er.append(Observation(**observation))


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
        observation = Observation(state, torch.empty(0)).to_torch()
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
            observation = Observation(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
            ).to_torch()
            state = next_state

            target_er.append(observation)
            if max_steps <= environment.time:
                break


class MakeRaw(object):
    """Make an experience replay get raw observations."""

    def __init__(self, experience_replay):
        self.experience_replay = experience_replay

    def __enter__(self):
        """Enter into make-raw context."""
        self.experience_replay.raw = True

    def __exit__(self, *args):
        """Exit make-raw context."""
        self.experience_replay.raw = False
