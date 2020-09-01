"""Model-Based BPTT Agent."""
from itertools import chain

import torch.nn.modules.loss as loss
from torch.optim import Adam

from rllib.algorithms.bptt import BPTT
from rllib.algorithms.model_learning_algorithm import ModelLearningAlgorithm
from rllib.model import EnsembleModel, TransformedModel
from rllib.policy import NNPolicy
from rllib.value_function import NNValueFunction

from .model_based_agent import ModelBasedAgent


class BPTTAgent(ModelBasedAgent):
    """Implementation of a Back-Propagation Through Time Agent."""

    def __init__(
        self,
        policy,
        critic,
        dynamical_model,
        reward_model,
        criterion,
        termination_model=None,
        num_steps=1,
        num_samples=15,
        algorithm=BPTT,
        *args,
        **kwargs,
    ):
        algorithm = algorithm(
            policy=policy,
            critic=critic,
            dynamical_model=dynamical_model,
            reward_model=reward_model,
            termination_model=termination_model,
            criterion=criterion(reduction="mean"),
            num_steps=num_steps,
            num_samples=num_samples,
            *args,
            **kwargs,
        )

        super().__init__(policy_learning_algorithm=algorithm, *args, **kwargs)

        self.optimizer = type(self.optimizer)(
            [
                p
                for name, p in self.algorithm.named_parameters()
                if ("model" not in name and "target" not in name and p.requires_grad)
            ],
            **self.optimizer.defaults,
        )

    @classmethod
    def default(cls, environment, *args, **kwargs):
        """See `AbstractAgent.default'."""
        test = kwargs.get("test", False)

        critic = NNValueFunction.default(environment)
        policy = NNPolicy.default(environment)

        optimizer = Adam(chain(policy.parameters(), critic.parameters()), lr=1e-3)
        criterion = loss.MSELoss

        dynamical_model = TransformedModel(
            EnsembleModel.default(environment), kwargs.get("transformations", list())
        )

        reward_model = kwargs.pop(
            "rewards", EnsembleModel.default(environment, model_kind="rewards")
        )
        model_optimizer = Adam(
            chain(dynamical_model.parameters(), reward_model.parameters()), lr=5e-4
        )

        model_learning_algorithm = ModelLearningAlgorithm(
            dynamical_model=dynamical_model,
            reward_model=reward_model,
            num_epochs=2 if kwargs.get("test", False) else 30,
            batch_size=64,
            bootstrap=True,
            model_optimizer=model_optimizer,
        )

        return cls(
            policy=policy,
            critic=critic,
            dynamical_model=dynamical_model,
            reward_model=reward_model,
            criterion=criterion,
            termination_model=None,
            learn_from_real=True,
            model_learning_algorithm=model_learning_algorithm,
            optimizer=optimizer,
            num_iter=5 if test else 50,
            batch_size=64,
            thompson_sampling=False,
            comment=environment.name,
            gamma=0.99,
            *args,
            **kwargs,
        )
