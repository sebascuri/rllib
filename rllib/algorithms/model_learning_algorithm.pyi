from typing import Any, Optional

from torch.optim.optimizer import Optimizer

from rllib.dataset.datatypes import Trajectory
from rllib.dataset.experience_replay import BootstrapExperienceReplay
from rllib.util.logger import Logger

from .abstract_mb_algorithm import AbstractMBAlgorithm

class ModelLearningAlgorithm(AbstractMBAlgorithm):
    model_optimizer: Optional[Optimizer]
    num_epochs: int = ...
    batch_size: int = ...
    dataset: BootstrapExperienceReplay
    def __init__(
        self,
        model_optimizer: Optional[Optimizer] = ...,
        num_epochs: int = ...,
        batch_size: int = ...,
        bootstrap: bool = ...,
        max_memory: int = ...,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
    def update_model_posterior(
        self, last_trajectory: Trajectory, logger: Logger
    ) -> None: ...
    def learn(self, last_trajectory: Trajectory, logger: Logger) -> None: ...
