"""Derived Agent."""
from itertools import chain

from torch.optim import Adam

from rllib.algorithms.model_learning_algorithm import ModelLearningAlgorithm
from rllib.algorithms.mpc.policy_shooting import PolicyShooting
from rllib.algorithms.simulation_algorithm import SimulationAlgorithm
from rllib.dataset.transforms import DeltaState, MeanFunction
from rllib.model import EnsembleModel, NNModel, TransformedModel

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

        super().__init__(policy_learning_algorithm=algorithm, *args, **kwargs)

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
        test = kwargs.get("test", False)

        from importlib import import_module

        base_agent = getattr(
            import_module("rllib.agent"), f"{base_agent_name}Agent"
        ).default(environment, *args, **kwargs)
        base_algorithm = base_agent.algorithm

        model = EnsembleModel.default(environment, deterministic=True)
        dynamical_model = TransformedModel(model, [MeanFunction(DeltaState())])

        reward_model = kwargs.pop(
            "rewards", NNModel.default(environment, model_kind="rewards")
        )

        model_optimizer = Adam(
            chain(dynamical_model.parameters(), reward_model.parameters()), lr=5e-4
        )

        model_learning_algorithm = ModelLearningAlgorithm(
            dynamical_model=dynamical_model,
            reward_model=reward_model,
            num_epochs=4 if kwargs.get("test", False) else 50,
            model_optimizer=model_optimizer,
        )
        simulation_algorithm = SimulationAlgorithm(
            dynamical_model=dynamical_model,
            reward_model=reward_model,
            initial_distribution=None,
            max_memory=100000,
            num_subsample=2,
            num_steps=4 if test else 200,
            num_initial_state_samples=8,
            num_initial_distribution_samples=0,
            num_memory_samples=4,
            refresh_interval=2,
        )
        planning_algorithm = PolicyShooting(
            policy=base_agent.policy,
            dynamical_model=dynamical_model,
            reward_model=reward_model,
            horizon=1,
            gamma=base_agent.gamma,
            num_iter=1,
            num_samples=8,
            num_elites=1,
            action_scale=base_agent.policy.action_scale,
            num_cpu=1,
        )

        return cls(
            base_algorithm=base_algorithm,
            dynamical_model=dynamical_model,
            reward_model=reward_model,
            model_learning_algorithm=model_learning_algorithm,
            simulation_algorithm=simulation_algorithm,
            planning_algorithm=planning_algorithm,
            optimizer=base_agent.optimizer,
            num_iter=5 if test else kwargs.pop("num_iter", base_agent.num_iter),
            batch_size=base_agent.batch_size,
            num_samples=15,
            num_steps=kwargs.pop("num_steps", 1),
            num_simulation_iterations=0,
            thompson_sampling=False,
            learn_from_real=True,
            gamma=base_algorithm.gamma,
            *args,
            **kwargs,
        )
