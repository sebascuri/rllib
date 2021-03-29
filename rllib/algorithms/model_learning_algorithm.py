"""Python Script Template."""
import numpy as np
from gym.utils import colorize

from rllib.dataset.experience_replay import BootstrapExperienceReplay
from rllib.dataset.utilities import stack_list_of_tuples
from rllib.model import ExactGPModel
from rllib.util.gaussian_processes import SparseGP
from rllib.util.training.model_learning import (
    calibrate_model,
    evaluate_model,
    train_model,
)

from .abstract_mb_algorithm import AbstractMBAlgorithm


class ModelLearningAlgorithm(AbstractMBAlgorithm):
    """An algorithm for model learning.

    Parameters
    ----------
    model_optimizer: Optimizer, optional.
        Optimizer to learn parameters of model.
    num_epochs: int.
        Number of epochs to iterate through the dataset.
    batch_size: int.
        Batch size of optimization algorithm.
    bootstrap: bool.
        Flag that indicates whether or not to add bootstrapping to dataset.
    max_memory: int.
        Maximum size of dataset.
    validation_ratio: float.
        Validation set ratio.

    Other Parameters
    ----------------
    See AbstractMBAlgorithm.

    Methods
    -------
    update_model_posterior(self, last_trajectory: Trajectory, logger: Logger) -> None:
        Update model posterior of GP models.
    learn(self, last_trajectory: Trajectory, logger: Logger) -> None: ...
        Learn using stochastic gradient descent on marginal maximum likelihood.
    """

    def __init__(
        self,
        model_optimizer=None,
        num_epochs=1,
        batch_size=100,
        bootstrap=True,
        max_memory=10000,
        epsilon=0.1,
        non_decrease_iter=5,
        validation_ratio=0.1,
        calibrate=True,
        num_steps=0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model_optimizer = model_optimizer

        if hasattr(self.dynamical_model.base_model, "num_heads"):
            num_heads = self.dynamical_model.base_model.num_heads
        else:
            num_heads = 1

        # Note: The transformations are shared by both data sets.
        self.train_set = BootstrapExperienceReplay(
            max_len=max_memory,
            transformations=self.dynamical_model.forward_transformations,
            num_bootstraps=num_heads,
            bootstrap=bootstrap,
            num_steps=num_steps,
        )
        self.validation_set = BootstrapExperienceReplay(
            max_len=max_memory,
            transformations=self.dynamical_model.forward_transformations,
            num_bootstraps=num_heads,
            bootstrap=bootstrap,
            num_steps=num_steps,
        )

        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.non_decrease_iter = non_decrease_iter
        self.validation_ratio = validation_ratio
        self.calibrate = calibrate

        if self.num_epochs > 0:
            assert self.model_optimizer is not None

    def _update_model_posterior(self, last_trajectory):
        """Update model posterior of GP-models with new data."""
        if isinstance(self.dynamical_model.base_model, ExactGPModel):
            observation = stack_list_of_tuples(last_trajectory)  # Parallelize.
            if observation.action.shape[-1] > self.dynamical_model.dim_action[0]:
                observation.action = observation.action[
                    ..., : self.dynamical_model.dim_action[0]
                ]
            for transform in self.train_set.transformations:
                observation = transform(observation)
            print(colorize("Add data to GP Model", "yellow"))
            self.dynamical_model.base_model.add_data(
                observation.state, observation.action, observation.next_state
            )

            print(colorize("Summarize GP Model", "yellow"))
            self.dynamical_model.base_model.summarize_gp()

    def add_last_trajectory(self, last_trajectory):
        """Add last trajectory to learning algorithm."""
        self._update_model_posterior(last_trajectory)
        for observation in last_trajectory:
            observation = observation.clone()
            if observation.action.shape[-1] > self.dynamical_model.dim_action[0]:
                observation.action = observation.action[
                    ..., : self.dynamical_model.dim_action[0]
                ]  # Only get real actions.
            if np.random.rand() < self.validation_ratio:
                self.validation_set.append(observation)
            else:
                self.train_set.append(observation)

    def _learn(self, model, logger, calibrate=False):
        """Learn a model."""
        print(colorize(f"Training {model.model_kind} model", "yellow"))

        train_model(
            model=model,
            train_set=self.train_set,
            validation_set=self.validation_set,
            batch_size=self.batch_size,
            num_epochs=self.num_epochs,
            optimizer=self.model_optimizer,
            logger=logger,
            epsilon=self.epsilon,
            non_decrease_iter=self.non_decrease_iter,
        )
        if (
            calibrate
            and not model.deterministic
            and len(self.validation_set) > self.batch_size
        ):
            calibrate_model(model, self.validation_set, self.num_epochs, logger=logger)

    def learn(self, logger):
        """Learn using stochastic gradient descent on marginal maximum likelihood."""
        self._learn(self.dynamical_model.base_model, logger, calibrate=self.calibrate)
        if len(self.validation_set) > self.batch_size:
            validation_data = self.validation_set.all_raw
            evaluate_model(self.dynamical_model, validation_data, logger)

        if any(p.requires_grad for p in self.reward_model.parameters()):
            self._learn(self.reward_model.base_model, logger, calibrate=self.calibrate)
            if len(self.validation_set) > self.batch_size:
                evaluate_model(self.reward_model, validation_data, logger)

        if self.termination_model is not None and any(
            p.requires_grad for p in self.termination_model.parameters()
        ):
            self._learn(self.termination_model, logger, calibrate=False)
            if len(self.validation_set) > self.batch_size:
                evaluate_model(self.termination_model, validation_data, logger)

        if isinstance(self.dynamical_model.base_model, ExactGPModel):
            for i, gp in enumerate(self.dynamical_model.base_model.gp):
                logger.update(**{f"gp{i} num inputs": len(gp.train_targets)})

                if isinstance(gp, SparseGP):
                    logger.update(**{f"gp{i} num inducing inputs": gp.xu.shape[0]})
