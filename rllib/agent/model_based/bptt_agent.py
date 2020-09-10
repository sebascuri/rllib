"""Model-Based BPTT Agent."""
from itertools import chain

import torch.nn.modules.loss as loss
from torch.optim import Adam

from rllib.algorithms.bptt import BPTT
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
        criterion=loss.MSELoss,
        termination_model=None,
        epsilon_mean=0.0,
        epsilon_var=0.0,
        regularization=True,
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
            epsilon_mean=epsilon_mean,
            epsilon_var=epsilon_var,
            regularization=regularization,
            *args,
            **kwargs,
        )

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

    @classmethod
    def default(cls, environment, *args, **kwargs):
        """See `AbstractAgent.default'."""
        test = kwargs.get("test", False)

        critic = NNValueFunction.default(environment)
        policy = NNPolicy.default(environment)

        optimizer = Adam(chain(policy.parameters(), critic.parameters()), lr=1e-3)

        return super().default(
            environment=environment,
            policy=policy,
            critic=critic,
            learn_from_real=True,
            optimizer=optimizer,
            num_iter=5 if test else 50,
            batch_size=100,
            *args,
            **kwargs,
        )
