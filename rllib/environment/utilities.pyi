"""Utilities for environment module."""

from .mdp import MDP
from .gym_environment import GymEnvironment
from rllib.policy import AbstractPolicy

def gym2mdp(environment: GymEnvironment) -> MDP: ...


def mdp2mrp(environment: MDP, policy: AbstractPolicy) -> MDP: ...
