from rllib.dataset.datatypes import State, Action, Reward, Done
from .dpg_agent import DPGAgent
from typing import Tuple, Any
from torch import Tensor


class GDPGAgent(DPGAgent):

    def _td(self, state: State, action: Action, reward: Reward, next_state: State,
            done: Done, *args: Any, **kwargs: Any) -> Tuple[Tensor, Tensor]: ...
