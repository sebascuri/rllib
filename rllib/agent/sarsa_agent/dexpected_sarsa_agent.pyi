from .abstract_sarsa_agent import AbstractSARSAAgent
from rllib.dataset.datatypes import State, Action, Reward, Done
from torch import Tensor
from typing import Tuple, Any

class DExpectedSARSAAgent(AbstractSARSAAgent):
    def _td(self, state: State, action: Action, reward: Reward, next_state: State,
            done: Done, next_action: Action, *args: Any, **kwargs: Any) -> Tuple[Tensor, Tensor]: ...
