"""Model-Based MPO Agent Implementation."""
from itertools import chain

import torch.nn.modules.loss as loss
from torch.optim import Adam

from rllib.agent.off_policy.mpo_agent import MPOAgent
from rllib.algorithms.mb_mpo import MBMPO

from .model_based_agent import ModelBasedAgent


class MBMPOAgent(ModelBasedAgent):
    """Implementation of an agent that runs MB-MPO."""

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

        self.algorithm = MBMPO(
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
                parameter
                for name, parameter in self.algorithm.named_parameters()
                if "target" not in name
                and "old_policy" not in name
                and "model" not in name
                and parameter.requires_grad
            ],
            **self.optimizer.defaults,
        )
        self.policy = self.algorithm.policy

    @classmethod
    def default(cls, environment, policy=None, critic=None, lr=5e-4, *args, **kwargs):
        """See `AbstractAgent.default'."""
        critic = MPOAgent.default_critic(environment) if critic is None else critic
        policy = MPOAgent.default_policy(environment) if policy is None else policy

        optimizer = Adam(chain(policy.parameters(), critic.parameters()), lr=lr)

        return super().default(
            environment,
            policy=policy,
            critic=critic,
            optimizer=optimizer,
            *args,
            **kwargs,
        )
