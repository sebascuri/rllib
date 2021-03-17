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
        num_samples=8,
        num_steps=1,
        only_sim=False,
        *args,
        **kwargs,
    ):
        super().__init__(policy_learning_algorithm=base_algorithm, *args, **kwargs)
        self.algorithm = derived_algorithm_(
            base_algorithm=self.algorithm,
            memory=self.memory,
            initial_state_dataset=self.initial_states_dataset,
            criterion=type(self.algorithm.criterion)(reduction="mean"),
            num_steps=num_steps,
            num_samples=num_samples,
            only_sim=only_sim,
            *args,
            **kwargs,
        )

        self.optimizer = type(self.optimizer)(
            [
                p
                for name, p in self.algorithm.named_parameters()
                if (
                    "model" not in name
                    and "target" not in name
                    and "old_policy" not in name
                    and p.requires_grad
                )
            ],
            **self.optimizer.defaults,
        )

    @property
    def name(self) -> str:
        """See `AbstractAgent.name'."""
        derived_name = self.__class__.__name__[:-5]
        try:
            base_name = self.algorithm.base_algorithm.__class__.__name__
        except AttributeError:
            base_name = self.algorithm.__class__.__name__
        return f"{derived_name}+{base_name}Agent"

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
