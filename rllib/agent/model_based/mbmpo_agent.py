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
        kl_regularization=False,
        train_frequency=0,
        num_iter=1000,
        num_rollouts=2,
        *args,
        **kwargs,
    ):
        super().__init__(
            train_frequency=train_frequency,
            num_iter=num_iter,
            num_rollouts=num_rollouts,
            *args,
            **kwargs,
        )

        self.algorithm = MPO(
            policy=policy,
            critic=critic,
            num_action_samples=num_action_samples,
            criterion=criterion(reduction="mean"),
            epsilon=epsilon,
            epsilon_mean=epsilon_mean,
            epsilon_var=epsilon_var,
            kl_regularization=kl_regularization,
            *args,
            **kwargs,
        )
        # Over-write optimizer.
        self.optimizer = type(self.optimizer)(
            [
                p
                for n, p in self.algorithm.named_parameters()
                if "target" not in n and "old_policy" not in n
            ],
            **self.optimizer.defaults,
        )
        self.policy = self.algorithm.policy

    @classmethod
    def default(cls, environment, policy=None, critic=None, lr=5e-4, *args, **kwargs):
        """See `AbstractAgent.default'."""
        if critic is None:
            critic = NNQFunction.default(environment)
        if policy is None:
            policy = NNPolicy.default(environment, layers=[100, 100])

        optimizer = Adam(chain(policy.parameters(), critic.parameters()), lr=lr)

        return super().default(
            environment,
            policy=policy,
            critic=critic,
            optimizer=optimizer,
            *args,
            **kwargs,
        )
