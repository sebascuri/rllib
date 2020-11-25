"""Derived Agent."""
from importlib import import_module

import torch

from rllib.algorithms.simulation_algorithm import SimulationAlgorithm
from rllib.dataset.experience_replay import ExperienceReplay
from rllib.dataset.utilities import unstack_observations

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

    def __init__(
        self,
        base_agent,
        num_steps=1,
        num_samples=2,
        num_initial_distribution_samples=0,
        num_memory_samples=32,
        num_initial_state_samples=0,
        refresh_interval=2,
        initial_distribution=None,
        only_sim=False,
        *args,
        **kwargs,
    ):
        super().__init__(
            policy_learning_algorithm=base_agent.algorithm, *args, **kwargs
        )
        self.simulation_algorithm = SimulationAlgorithm(
            dynamical_model=self.dynamical_model,
            reward_model=self.reward_model,
            termination_model=self.termination_model,
            num_initial_distribution_samples=num_initial_distribution_samples,
            num_memory_samples=num_memory_samples,
            num_initial_state_samples=num_initial_state_samples,
            num_steps=num_steps,
            num_samples=num_samples,
            initial_distribution=initial_distribution,
        )
        self.sim_memory = ExperienceReplay(
            max_len=self.memory.max_len,
            num_steps=self.memory.num_steps,
            transformations=self.memory.transformations,
        )
        self.refresh_interval = refresh_interval
        self.only_sim = only_sim

    def simulate(self):
        """Simulate model and generate transitions. Append to the data set."""
        with torch.no_grad():
            self.policy.reset()  # TODO: Add goal distribution.
            initial_states = self.simulation_algorithm.get_initial_states(
                self.initial_states_dataset, self.memory
            )

            trajectory = self.simulation_algorithm.simulate(
                initial_states, self.policy, logger=self.logger, stack_obs=False
            )
            for observations in trajectory:
                observation_samples = unstack_observations(observations)
                for observation in observation_samples:
                    self.sim_memory.append(observation)

    def learn(self):
        """Simulate the model and optimize the policy with the learned data.

        This consists of two steps:
            Step 1: Simulate trajectories with the model.
            Step 2: Implement a model free RL method that optimizes the policy.
        """
        self.dynamical_model.eval()

        # Step 1: Simulate the state distribution
        if (
            self.refresh_interval > 0
            and self.train_steps % (self.refresh_interval * self.num_iter) == 0
        ):
            self.sim_memory.reset()
        self.simulate()

        # Learn base algorithm with real data set.
        if not self.only_sim:
            super().learn()
        super().learn(memory=self.sim_memory)

    @property
    def name(self) -> str:
        """See `AbstractAgent.name'."""
        base_name = self.algorithm.__class__.__name__
        return f"DataAugmentation+{base_name}Agent"

    @classmethod
    def default(cls, environment, base_agent="SAC", *args, **kwargs):
        """See `AbstractAgent.default'."""
        base_agent = getattr(
            import_module("rllib.agent"), f"{base_agent}Agent"
        ).default(environment, *args, **kwargs)
        base_agent.logger.delete_directory()
        kwargs.update(
            optimizer=base_agent.optimizer,
            num_iter=base_agent.num_iter,
            batch_size=base_agent.batch_size,
            train_frequency=base_agent.train_frequency,
            num_rollouts=base_agent.num_rollouts,
        )

        return super().default(
            environment=environment, base_agent=base_agent, *args, **kwargs
        )
