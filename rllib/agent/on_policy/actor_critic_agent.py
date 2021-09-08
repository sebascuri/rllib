"""Implementation of Model-Free Policy Gradient Algorithms."""

import torch.nn.modules.loss as loss
from torch.optim import Adam

from rllib.algorithms.ac import ActorCritic
from rllib.policy import NNPolicy
from rllib.value_function import NNQFunction

from .on_policy_agent import OnPolicyAgent


class ActorCriticAgent(OnPolicyAgent):
    """Abstract Implementation of the Actor-Critic Agent.

    The AbstractEpisodicPolicyGradient algorithm implements the Actor-Critic algorithms.

    TODO: build compatible function approximation.

    References
    ----------
    Sutton, R. S., McAllester, D. A., Singh, S. P., & Mansour, Y. (2000).
    Policy gradient methods for reinforcement learning with function approximation.NIPS.

    Konda, V. R., & Tsitsiklis, J. N. (2000).
    Actor-critic algorithms. NIPS.
    """

    eps = 1e-12

    def __init__(
        self,
        policy,
        critic,
        algorithm_=ActorCritic,
        criterion=loss.MSELoss,
        eta=0.001,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.algorithm = algorithm_(
            policy=policy,
            critic=critic,
            criterion=criterion(reduction="mean"),
            eta=eta,
            *args,
            **kwargs,
        )
        self.policy = self.algorithm.policy
        self.optimizer = type(self.optimizer)(
            [
                p
                for n, p in self.algorithm.named_parameters()
                if "target" not in n and "old_policy" not in n
            ],
            **self.optimizer.defaults,
        )

    @classmethod
    def default(
        cls,
        environment,
        policy=None,
        critic=None,
        critic_lr=1e-3,
        actor_lr=3e-4,
        *args,
        **kwargs,
    ):
        """See `AbstractAgent.default'."""
        if policy is None:
            policy = NNPolicy.default(environment)
        if critic is None:
            critic = NNQFunction.default(environment)

        optimizer = Adam(
            [
                {"params": policy.parameters(), "lr": actor_lr},
                {"params": critic.parameters(), "lr": critic_lr},
            ]
        )

        return super().default(
            environment=environment,
            policy=policy,
            critic=critic,
            optimizer=optimizer,
            *args,
            **kwargs,
        )
