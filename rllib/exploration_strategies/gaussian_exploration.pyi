from .abstract_exploration_strategy import AbstractExplorationStrategy
from .utilities import Distribution, Action


class GaussianExploration(AbstractExplorationStrategy):
    def __call__(self, action_distribution: Distribution, steps: int = None
                 ) -> Action: ...
