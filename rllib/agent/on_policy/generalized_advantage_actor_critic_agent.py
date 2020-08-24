"""Implementation of Advantage-Actor Critic Agent."""

import torch.nn.modules.loss as loss
from torch.optim import Adam

from rllib.algorithms.gaac import GAAC
from rllib.policy import NNPolicy
from rllib.value_function import NNValueFunction

from .actor_critic_agent import ActorCriticAgent


class GAACAgent(ActorCriticAgent):
    """Implementation of the Advantage-Actor Critic.

    TODO: build compatible function approximation.

    References
    ----------
    Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2015).
    High-dimensional continuous control using generalized advantage estimation. ICLR.
    """

    def __init__(
        self, policy, critic, optimizer, criterion, lambda_=0.97, *args, **kwargs
    ):
        super().__init__(
            policy=policy,
            critic=critic,
            optimizer=optimizer,
            criterion=criterion,
            *args,
            **kwargs,
        )
        self.algorithm = GAAC(
            policy=policy,
            critic=critic,
            criterion=criterion(reduction="mean"),
            lambda_=lambda_,
            gamma=self.gamma,
        )
        self.policy = self.algorithm.policy

    @classmethod
    def default(cls, environment, *args, **kwargs):
        """See `AbstractAgent.default'."""
        policy = NNPolicy(
            dim_state=environment.dim_state,
            dim_action=environment.dim_action,
            num_states=environment.num_states,
            num_actions=environment.num_actions,
            layers=[200, 200],
            biased_head=True,
            non_linearity="Tanh",
            squashed_output=True,
            action_scale=environment.action_scale,
            tau=5e-3,
            initial_scale=0.5,
            deterministic=False,
            goal=environment.goal,
            input_transform=None,
        )
        critic = NNValueFunction(
            dim_state=environment.dim_state,
            num_states=environment.num_states,
            layers=[200, 200],
            biased_head=True,
            non_linearity="Tanh",
            tau=5e-3,
            input_transform=None,
        )

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
