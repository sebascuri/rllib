"""Model-Based SVG Agent."""
from itertools import chain

import torch.nn.modules.loss as loss
from torch.optim import Adam

from rllib.algorithms.svg import SVG
from rllib.policy import NNPolicy
from rllib.value_function import NNValueFunction

from .model_based_agent import ModelBasedAgent


class SVGAgent(ModelBasedAgent):
    """Implementation of a SVG-Agent."""

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
        kl_regularization=True,
        eta_mean=0.0,
        entropy_regularization=True,
        *args,
        **kwargs,
    ):
        algorithm = SVG(
            policy=policy,
            critic=critic,
            dynamical_model=dynamical_model,
            reward_model=reward_model,
            termination_model=termination_model,
            criterion=criterion(reduction="mean"),
            epsilon_mean=epsilon_mean,
            epsilon_var=epsilon_var,
            kl_regularization=kl_regularization,
            eta=eta_mean,
            entropy_regularization=entropy_regularization,
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
    def default(cls, environment, num_iter=50, lr=3e-4, *args, **kwargs):
        """See `AbstractAgent.default'."""
        critic = NNValueFunction.default(environment)
        policy = NNPolicy.default(environment)
        optimizer = Adam(chain(policy.parameters(), critic.parameters()), lr=lr)

        return super().default(
            environment=environment,
            policy=policy,
            critic=critic,
            learn_from_real=True,
            optimizer=optimizer,
            num_iter=num_iter,
            batch_size=100,
            *args,
            **kwargs,
        )
