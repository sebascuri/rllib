"""Model-Based BPTT Agent."""
from itertools import chain

import torch.nn.modules.loss as loss
from torch.optim import Adam

from rllib.algorithms.bptt import BPTT
from rllib.policy import NNPolicy
from rllib.value_function import NNEnsembleQFunction

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
        num_steps=1,
        num_samples=15,
        *args,
        **kwargs,
    ):
        algorithm = BPTT(
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
                if (
                    "model" not in name
                    and "target" not in name
                    and "old_policy" not in name
                    and p.requires_grad
                )
            ],
            **self.optimizer.defaults,
        )

    @classmethod
    def default(cls, environment, critic=None, policy=None, lr=3e-4, *args, **kwargs):
        """See `AbstractAgent.default'."""
        if critic is None:
            critic = NNEnsembleQFunction.default(environment)
        if policy is None:
            policy = NNPolicy.default(environment)
        optimizer = Adam(chain(policy.parameters(), critic.parameters()), lr=lr)

        return super().default(
            environment=environment,
            policy=policy,
            critic=critic,
            optimizer=optimizer,
            *args,
            **kwargs,
        )
