from ..abstract_agent import State, Action, Reward, Done
from .dpg_agent import DPGAgent
from typing import Tuple
from torch import Tensor


class GDPGAgent(DPGAgent):

    def _td(self, state: State, action: Action, reward: Reward, next_state: State,
            done: Done) -> Tuple[Tensor, Tensor]: ...
