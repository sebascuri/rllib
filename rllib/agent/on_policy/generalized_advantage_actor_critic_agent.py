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
        self,
        policy,
        critic,
        optimizer,
        criterion,
        num_iter=1,
        target_update_frequency=1,
        lambda_=0.97,
        train_frequency=0,
        num_rollouts=1,
        gamma=0.99,
        exploration_steps=0,
        exploration_episodes=0,
        tensorboard=False,
        comment="",
    ):
        super().__init__(
            policy=policy,
            critic=critic,
            optimizer=optimizer,
            criterion=criterion,
            num_iter=num_iter,
            target_update_frequency=target_update_frequency,
            train_frequency=train_frequency,
            num_rollouts=num_rollouts,
            gamma=gamma,
            exploration_steps=exploration_steps,
            exploration_episodes=exploration_episodes,
            tensorboard=tensorboard,
            comment=comment,
        )
        self.algorithm = GAAC(
            policy, critic, criterion(reduction="none"), lambda_, gamma
        )
        self.policy = self.algorithm.policy

    @classmethod
    def default(
        cls,
        environment,
        gamma=0.99,
        exploration_steps=0,
        exploration_episodes=0,
        tensorboard=False,
        test=False,
    ):
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

        # optimizer = Adam(chain(policy.parameters(), critic.parameters()), 3e-4)
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
            num_iter=1,
            target_update_frequency=1,
            train_frequency=0,
            num_rollouts=1,
            gamma=gamma,
            exploration_steps=exploration_steps,
            exploration_episodes=exploration_episodes,
            tensorboard=tensorboard,
            comment=environment.name,
        )
