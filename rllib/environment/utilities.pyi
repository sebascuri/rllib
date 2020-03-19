"""Utilities for environment module."""

from rllib.policy import AbstractPolicy
from .gym_environment import GymEnvironment
from .mdp import MDP


def gym2mdp(environment: GymEnvironment) -> MDP: ...


def mdp2mrp(environment: MDP, policy: AbstractPolicy) -> MDP: ...
