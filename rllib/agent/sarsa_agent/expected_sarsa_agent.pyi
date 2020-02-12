from .abstract_sarsa_agent import AbstractSARSAAgent
from ..abstract_agent import State, Action, Reward, Done
from torch import Tensor
from typing import Tuple

class ExpectedSARSAAgent(AbstractSARSAAgent):
    def _td(self, state: State, action: Action, reward: Reward, next_state: State,
            done: Done, next_action: Action) -> Tuple[Tensor, Tensor]: ...
