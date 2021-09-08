"""Implementation of DQNAgent Algorithms."""
from itertools import chain

import torch.nn.modules.loss as loss
from torch.optim import Adam

from rllib.algorithms.sac import SAC
from rllib.policy import NNPolicy
from rllib.value_function import NNEnsembleQFunction

from .off_policy_agent import OffPolicyAgent


class SACAgent(OffPolicyAgent):
    """Implementation of a SAC agent.

    Parameters
    ----------
    critic: AbstractQFunction
        critic that is learned.
    criterion: nn.Module
        Criterion to minimize the TD-error.

    References
    ----------
    Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018).
    Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a
    stochastic actor. ICML.

    """

    def __init__(
        self,
        critic,
        policy,
        criterion=loss.MSELoss,
        eta=0.2,
        entropy_regularization=False,
        num_iter=50,
        train_frequency=50,
        *args,
        **kwargs,
    ):
        super().__init__(
            num_iter=num_iter, train_frequency=train_frequency, *args, **kwargs
        )
        self.algorithm = SAC(
            policy=policy,
            critic=critic,
            criterion=criterion(reduction="none"),
            eta=eta,
            entropy_regularization=entropy_regularization,
            *args,
            **kwargs,
        )

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
    def default(cls, environment, policy=None, critic=None, lr=3e-4, *args, **kwargs):
        """See `AbstractAgent.default'."""
        if critic is None:
            critic = NNEnsembleQFunction.default(environment, jit_compile=False)
        if policy is None:
            policy = NNPolicy.default(environment, jit_compile=False)

        optimizer = Adam(chain(policy.parameters(), critic.parameters()), lr=lr)

        return super().default(
            environment,
            critic=critic,
            policy=policy,
            optimizer=optimizer,
            *args,
            **kwargs,
        )
