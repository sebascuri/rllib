from .q_learning_agent import QLearningAgent
from rllib.dataset.datatypes import State, Action, Reward, Done
from torch import Tensor
from typing import Tuple, Any


class GQLearningAgent(QLearningAgent):

    def _td(self, state: State, action: Action, reward: Reward, next_state: State,
            done: Done, *args: Any, **kwargs: Any) -> Tuple[Tensor, Tensor]: ...

