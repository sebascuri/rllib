"""MPO Agent Implementation."""
from itertools import chain

import torch.nn.modules.loss as loss
from torch.optim import Adam

from rllib.algorithms.mpo import MPO
from rllib.policy import NNPolicy
from rllib.value_function import NNQFunction

from .off_policy_agent import OffPolicyAgent


class MPOAgent(OffPolicyAgent):
    """Implementation of an agent that runs MPO."""

    def __init__(
        self,
        policy,
        critic,
        criterion=loss.MSELoss,
        num_action_samples=15,
        epsilon=0.1,
        epsilon_mean=0.1,
        epsilon_var=0.001,
        regularization=False,
        train_frequency=0,
        num_rollouts=2,
        *args,
        **kwargs,
    ):
        super().__init__(
            train_frequency=train_frequency, num_rollouts=num_rollouts, *args, **kwargs
        )

        self.algorithm = MPO(
            policy=policy,
            critic=critic,
            num_action_samples=num_action_samples,
            criterion=criterion(reduction="none"),
            epsilon=epsilon,
            epsilon_mean=epsilon_mean,
            epsilon_var=epsilon_var,
            regularization=regularization,
            gamma=self.gamma,
        )
        # Over-write optimizer.
        self.optimizer = type(self.optimizer)(
            [p for n, p in self.algorithm.named_parameters() if "target" not in n],
            **self.optimizer.defaults,
        )
        self.policy = self.algorithm.policy

    @classmethod
    def default(cls, environment, *args, **kwargs):
        """See `AbstractAgent.default'."""
        critic = kwargs.pop("critic", NNQFunction.default(environment))
        policy = NNPolicy.default(environment, layers=[100, 100])

        optimizer = Adam(chain(policy.parameters(), critic.parameters()), lr=5e-4)

        return super().default(
            environment,
            policy=policy,
            critic=critic,
            optimizer=optimizer,
            num_iter=5 if kwargs.get("test", False) else 1000,
            *args,
            **kwargs,
        )
