"""Derived Agent."""
from importlib import import_module

from .model_based_agent import ModelBasedAgent


class DataAugmentationAgent(ModelBasedAgent):
    """Data Augmentation simulates data with the model and trains with such data.

    References
    ----------
    Venkatraman, A., et al. (2016)
    Improved learning of dynamics models for control.
    International Symposium on Experimental Robotics.

    Kalweit, G., & Boedecker, J. (2017).
    Uncertainty-driven imagination for continuous deep reinforcement learning. CoRL.

    Racanière, S., et al. (2017).
    Imagination-augmented agents for deep reinforcement learning. NeuRIPS.
    """

    def __init__(self, base_algorithm, *args, **kwargs):
        super().__init__(
            policy_learning_algorithm=base_algorithm,
            augment_dataset_with_sim=True,
            *args,
            **kwargs,
        )

    @classmethod
    def default(cls, environment, base_agent="SAC", *args, **kwargs):
        """See `AbstractAgent.default'."""
        base_agent = getattr(
            import_module("rllib.agent"), f"{base_agent}Agent"
        ).default(environment, *args, **kwargs)
        base_agent.logger.delete_directory()
        base_algorithm = base_agent.algorithm
        kwargs.update(
            optimizer=base_agent.optimizer,
            num_iter=base_agent.num_iter,
            batch_size=base_agent.batch_size,
            train_frequency=base_agent.train_frequency,
            num_rollouts=base_agent.num_rollouts,
        )
        return super().default(
            environment=environment, base_algorithm=base_algorithm, *args, **kwargs
        )
