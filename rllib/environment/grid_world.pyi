from typing import Dict, List, Optional, Tuple

import numpy as np

from rllib.dataset.datatypes import Action, State

from .mdp import MDP

class EasyGridWorld(MDP):
    width: int
    height: int
    def __init__(
        self,
        width: int = ...,
        height: int = ...,
        num_actions: int = ...,
        terminal_states: Optional[List[State]] = ...,
    ) -> None: ...
    def _build_mdp(
        self, terminal_states: Optional[List[State]] = ...
    ) -> Dict[Tuple[int, int], List]: ...
    def _state_to_grid(self, state: State) -> np.ndarray: ...
    def _grid_to_state(self, grid_state: np.ndarray) -> State: ...
    def _action_to_grid(self, action: Action) -> np.ndarray: ...
    def _is_valid(self, grid_state: np.ndarray) -> bool: ...
