"""Derived Agent."""
from importlib import import_module

from .model_based_agent import ModelBasedAgent


class DerivedMBAgent(ModelBasedAgent):
    """Implementation of a Derived Agent.

    A Derived Agent gets a model-free algorithm and uses the model to derive an
    algorithm.
    """

    def __init__(
        self,
        base_algorithm,
        derived_algorithm_,
        dynamical_model,
        reward_model,
        num_samples=15,
        num_steps=1,
        termination_model=None,
        *args,
        **kwargs,
    ):
        algorithm = derived_algorithm_(
            base_algorithm=base_algorithm,
            dynamical_model=dynamical_model,
            reward_model=reward_model,
            termination_model=termination_model,
            num_steps=num_steps,
            num_samples=num_samples,
            *args,
            **kwargs,
        )
        algorithm.criterion = type(algorithm.criterion)(reduction="mean")

        super().__init__(
            policy_learning_algorithm=algorithm,
            dynamical_model=dynamical_model,
            reward_model=reward_model,
            termination_model=termination_model,
            *args,
            **kwargs,
        )

        self.optimizer = type(self.optimizer)(
            [
                p
                for name, p in self.algorithm.named_parameters()
                if ("model" not in name and "target" not in name and p.requires_grad)
            ],
            **self.optimizer.defaults,
        )

    @property
    def name(self) -> str:
        """See `AbstractAgent.name'."""
        derived_name = self.__class__.__name__[:-5]
        base_name = self.algorithm.base_algorithm_name
        return f"{derived_name}+{base_name}Agent"

    @classmethod
    def default(cls, environment, base_agent_name="SAC", *args, **kwargs):
        """See `AbstractAgent.default'."""
        base_agent = getattr(
            import_module("rllib.agent"), f"{base_agent_name}Agent"
        ).default(environment, *args, **kwargs)
        base_agent.logger.delete_directory()
        base_algorithm = base_agent.algorithm
        return super().default(
            environment=environment,
            base_algorithm=base_algorithm,
            optimizer=base_agent.optimizer,
            num_iter=base_agent.num_iter,
            batch_size=base_agent.batch_size,
            train_frequency=base_agent.train_frequency,
            num_rollouts=base_agent.num_rollouts,
            *args,
            **kwargs,
        )
