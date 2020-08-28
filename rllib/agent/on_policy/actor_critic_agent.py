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

    def __init__(self, policy, critic, criterion, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.algorithm = ActorCritic(
            policy=policy,
            critic=critic,
            criterion=criterion(reduction="mean"),
            gamma=self.gamma,
        )
        self.policy = self.algorithm.policy

    @classmethod
    def default(cls, environment, *args, **kwargs):
        """See `AbstractAgent.default'."""
        policy = NNPolicy.default(environment)
        critic = NNQFunction.default(environment)

        optimizer = Adam(
            [
                {"params": policy.parameters(), "lr": 1e-4},
                {"params": critic.parameters(), "lr": 1e-3},
            ]
        )
        criterion = loss.MSELoss

        return cls(
            policy=policy,
            critic=critic,
            optimizer=optimizer,
            criterion=criterion,
            num_iter=8,
            target_update_frequency=1,
            train_frequency=0,
            num_rollouts=8,
            comment=environment.name,
            *args,
            **kwargs,
        )
