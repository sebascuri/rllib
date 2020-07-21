"""Baird Star Environment."""
import numpy as np

from rllib.environment.mdp import MDP, Transition

class BairdStar(MDP):
    def __init__(self, num_states: int = ...) -> None: ...
    @staticmethod
    def _build_mdp(num_states: int) -> Transition: ...
    @staticmethod
    def _build_feature_matrix(num_states: int) -> np.ndarray: ...
