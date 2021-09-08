from typing import Any, Optional

from torch.optim.optimizer import Optimizer

from rllib.dataset.datatypes import Trajectory
from rllib.dataset.experience_replay import BootstrapExperienceReplay
from rllib.model import AbstractModel
from rllib.util.logger import Logger

from .abstract_mb_algorithm import AbstractMBAlgorithm

class ModelLearningAlgorithm(AbstractMBAlgorithm):
    model_optimizer: Optional[Optimizer]
    num_epochs: int
    batch_size: int
    epsilon: float
    non_decrease_iter: int
    validation_ratio: float
    calibrate: bool
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
        epsilon: float = ...,
        non_decrease_iter: int = ...,
        calibrate: bool = ...,
        num_steps: int = ...,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
    def _update_model_posterior(self, last_trajectory: Trajectory) -> None: ...
    def add_last_trajectory(self, last_trajectory: Trajectory) -> None: ...
    def _learn(
        self, model: AbstractModel, logger: Logger, calibrate: bool = ...
    ) -> None: ...
    def learn(self, logger: Logger) -> None: ...
