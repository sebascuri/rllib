"""Derived Agent."""
from rllib.algorithms.data_augmentation import DataAugmentation

from .derived_model_based_agent import DerivedMBAgent


class DataAugmentationAgent(DerivedMBAgent):
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

    def __init__(
        self,
        num_initial_distribution_samples=0,
        num_memory_samples=16,
        num_initial_state_samples=0,
        refresh_interval=2,
        initial_distribution=None,
        *args,
        **kwargs,
    ):
        super().__init__(
            derived_algorithm_=DataAugmentation,
            num_initial_distribution_samples=num_initial_distribution_samples,
            num_memory_samples=num_memory_samples,
            num_initial_state_samples=num_initial_state_samples,
            refresh_interval=refresh_interval,
            initial_distribution=initial_distribution,
            *args,
            **kwargs,
        )
