from .abstract_agent import AbstractAgent, State, Action
from rllib.dataset import TrajectoryDataset, Observation
from rllib.policy import RandomPolicy
from typing import List

class RandomAgent(AbstractAgent):
    """Agent that interacts randomly in an environment."""
    _policy: RandomPolicy
    trajectory: List[Observation]
    dataset: TrajectoryDataset

    def __init__(self, dim_state: int, dim_action: int,
                 num_states: int = None, num_actions: int = None,
                 gamma: float = None,
                 exploration_steps: int = 0, exploration_episodes: int = 0) -> None: ...

    def act(self, state: State) -> Action: ...

    def observe(self, observation: Observation) -> None: ...

    def start_episode(self) -> None: ...

    def end_episode(self) -> None: ...

