from .mdp import MDP
from typing import List, Tuple
import numpy as np
from rllib.dataset.datatypes import State, Action



class EasyGridWorld(MDP):
    width: int
    height: int

    def __init__(self, width: int = 5, height: int = 5, num_actions: int = 4,
                 terminal_states: List[State] = None) -> None: ...

    def _build_mdp(self, num_actions: int, terminal_states: List[State] = None
                   ) -> Tuple[np.ndarray, np.ndarray]: ...

    def _state_to_grid(self, state: State) -> np.ndarray: ...

    def _grid_to_state(self, grid_state: np.ndarray) -> State: ...

    @staticmethod
    def _action_to_grid(self, action: Action, num_actions: int) -> np.ndarray: ...

    def _is_valid(self, grid_state: np.ndarray) -> bool: ...
