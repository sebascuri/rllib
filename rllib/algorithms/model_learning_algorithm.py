"""Python Script Template."""
from gym.utils import colorize
from torch.utils.data import DataLoader

from rllib.dataset.experience_replay import BootstrapExperienceReplay
from rllib.dataset.utilities import stack_list_of_tuples
from rllib.model import ExactGPModel
from rllib.util.gaussian_processes import SparseGP
from rllib.util.training import train_model

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

        self.num_epochs = num_epochs
        self.batch_size = batch_size
        if self.num_epochs > 0:
            assert self.model_optimizer is not None

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
        """Learn using stochastic gradient descent on marginal maximum likelihood.

        The last trajectory is added to the data set.

        Step 1: Train dynamical model.

        """
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

        if any(p.requires_grad for p in self.reward_model.parameters()):
            train_model(
                self.reward_model,
                train_loader=loader,
                max_iter=self.num_epochs,
                optimizer=self.model_optimizer,
                logger=logger,
            )

        if self.termination_model is not None and any(
            p.requires_grad for p in self.termination_model.parameters()
        ):
            train_model(
                self.termination_model,
                train_loader=loader,
                max_iter=self.num_epochs,
                optimizer=self.model_optimizer,
                logger=logger,
            )
