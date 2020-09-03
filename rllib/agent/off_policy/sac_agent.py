"""Implementation of DQNAgent Algorithms."""
from itertools import chain

import torch.nn.modules.loss as loss
from torch.optim import Adam

from rllib.algorithms.sac import SoftActorCritic
from rllib.policy import NNPolicy
from rllib.value_function import NNEnsembleQFunction, NNQFunction

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
        regularization=False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        critic = NNEnsembleQFunction.from_q_function(q_function=critic, num_heads=2)
        self.algorithm = SoftActorCritic(
            policy=policy,
            critic=critic,
            criterion=criterion(reduction="none"),
            gamma=self.gamma,
            eta=eta,
            regularization=regularization,
        )

        self.optimizer = type(self.optimizer)(
            [p for n, p in self.algorithm.named_parameters() if "target" not in n],
            **self.optimizer.defaults,
        )
        self.policy = self.algorithm.policy

    @classmethod
    def default(cls, environment, *args, **kwargs):
        """See `AbstractAgent.default'."""
        critic = NNQFunction.default(environment, non_linearity="ReLU")
        policy = NNPolicy.default(environment, non_linearity="ReLU")

        optimizer = Adam(chain(policy.parameters(), critic.parameters()), lr=1e-3)

        return super().default(
            environment,
            critic=critic,
            policy=policy,
            optimizer=optimizer,
            num_iter=kwargs.pop("num_iter", 50),
            train_frequency=kwargs.pop("train_frequency", 50),
            *args,
            **kwargs,
        )
