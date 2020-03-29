"""Utilities for environment module."""

from typing import Dict, Tuple, List
import numpy as np

from rllib.policy import AbstractPolicy
from .gym_environment import GymEnvironment
from .mdp import MDP


def gym2mdp(environment: GymEnvironment) -> MDP: ...


def mdp2mrp(environment: MDP, policy: AbstractPolicy) -> MDP: ...

def kernelreward2transitions(kernel: np.ndarray, reward: np.ndarray
                             ) -> Dict[Tuple[int, int], List]: ...

def transitions2kernelreward(transitions: Dict[Tuple[int, int], List], num_states: int, num_actions: int
                             ) -> Tuple[np.ndarray, np.ndarray]: ...
