"""Utilities for environment module."""

from typing import Dict, List, Tuple

import numpy as np
from gym.spaces import Space

from rllib.policy import AbstractPolicy

from .gym_environment import GymEnvironment
from .mdp import MDP

def parse_space(space: Space) -> Tuple[Tuple, int]: ...
def gym2mdp(environment: GymEnvironment) -> MDP: ...
def mdp2mrp(environment: MDP, policy: AbstractPolicy) -> MDP: ...
def kernelreward2transitions(
    kernel: np.ndarray, reward: np.ndarray
) -> Dict[Tuple[int, int], List]: ...
def transitions2kernelreward(
    transitions: Dict[Tuple[int, int], List], num_states: int, num_actions: int
) -> Tuple[np.ndarray, np.ndarray]: ...
