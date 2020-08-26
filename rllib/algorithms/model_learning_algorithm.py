"""Python Script Template."""
from gym.utils import colorize
from torch.utils.data import DataLoader

from rllib.dataset.experience_replay import (
    BootstrapExperienceReplay,
    StateExperienceReplay,
)
from rllib.dataset.utilities import stack_list_of_tuples
from rllib.model import ExactGPModel
from rllib.util.gaussian_processes import SparseGP
from rllib.util.training import train_model

from .abstract_mb_algorithm import AbstractMBAlgorithm


class ModelLearningAlgorithm(AbstractMBAlgorithm):
    """Base class for model learning algorithm."""

    def __init__(
        self,
        model_optimizer,
        num_epochs=1,
        batch_size=64,
        bootstrap=True,
        max_memory=10000,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model_optimizer = model_optimizer

        if hasattr(self.dynamical_model.base_model, "num_heads"):
            num_heads = self.dynamical_model.base_model.num_heads
        else:
            num_heads = 1

        self.dataset = BootstrapExperienceReplay(
            max_len=max_memory,
            transformations=self.dynamical_model.forward_transformations,
            num_bootstraps=num_heads,
            bootstrap=bootstrap,
        )
        self.initial_states_dataset = StateExperienceReplay(
            max_len=max_memory, dim_state=self.dynamical_model.dim_state
        )

        self.num_epochs = num_epochs
        self.batch_size = batch_size

    def update_model_posterior(self, last_trajectory, logger):
        """Update model posterior of GP-models with new data."""
        if isinstance(self.dynamical_model.base_model, ExactGPModel):
            observation = stack_list_of_tuples(last_trajectory)  # Parallelize.
            for transform in self.dataset.transformations:
                observation = transform(observation)
            print(colorize("Add data to GP Model", "yellow"))
            self.dynamical_model.base_model.add_data(
                observation.state, observation.action, observation.next_state
            )

            print(colorize("Summarize GP Model", "yellow"))
            self.dynamical_model.base_model.summarize_gp()

            for i, gp in enumerate(self.dynamical_model.base_model.gp):
                logger.update(**{f"gp{i} num inputs": len(gp.train_targets)})

                if isinstance(gp, SparseGP):
                    logger.update(**{f"gp{i} num inducing inputs": gp.xu.shape[0]})

        print(colorize("Training Model", "yellow"))

    def learn(self, last_trajectory, logger):
        """Learn model with the last trajectory.

        Step 1: Train dynamical model.
        Step 2: TODO Train the reward model.
        Step 3: TODO Train the initial distribution model.

        """
        self.initial_states_dataset.append(last_trajectory[0].state.unsqueeze(0))

        self.update_model_posterior(last_trajectory, logger)
        for observation in last_trajectory:
            self.dataset.append(observation)

        loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        train_model(
            self.dynamical_model.base_model,
            train_loader=loader,
            max_iter=self.num_epochs,
            optimizer=self.model_optimizer,
            logger=logger,
        )
