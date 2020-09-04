from typing import Any, Optional

from torch.optim.optimizer import Optimizer

from rllib.dataset.datatypes import Trajectory
from rllib.dataset.experience_replay import BootstrapExperienceReplay
from rllib.util.logger import Logger

from .abstract_mb_algorithm import AbstractMBAlgorithm

class ModelLearningAlgorithm(AbstractMBAlgorithm):
    model_optimizer: Optional[Optimizer]
    num_epochs: int
    batch_size: int
    epsilon: float
    validation_ratio: float
    train_set: BootstrapExperienceReplay
    validation_set: BootstrapExperienceReplay
    def __init__(
        self,
        model_optimizer: Optional[Optimizer] = ...,
        num_epochs: int = ...,
        batch_size: int = ...,
        bootstrap: bool = ...,
        max_memory: int = ...,
        validation_ratio: float = ...,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
    def update_model_posterior(
        self, last_trajectory: Trajectory, logger: Logger
    ) -> None: ...
    def learn(self, last_trajectory: Trajectory, logger: Logger) -> None: ...
