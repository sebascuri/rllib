"""Implementation of REINFORCE Algorithms."""

import torch.nn as nn
from torch.optim import Adam

from rllib.algorithms.reinforce import REINFORCE
from rllib.policy import NNPolicy
from rllib.value_function import NNValueFunction

from .on_policy_agent import OnPolicyAgent


class REINFORCEAgent(OnPolicyAgent):
    """Implementation of the REINFORCE algorithm.

    The REINFORCE algorithm computes the policy gradient using MC
    approximation for the returns (sum of discounted rewards).

    References
    ----------
    Williams, Ronald J. "Simple statistical gradient-following algorithms for
    connectionist reinforcement learning." Machine learning 8.3-4 (1992): 229-256.
    """

    def __init__(
        self,
        policy,
        optimizer,
        baseline=None,
        criterion=nn.MSELoss,
        num_iter=1,
        target_update_frequency=1,
        train_frequency=0,
        num_rollouts=1,
        gamma=0.99,
        exploration_steps=0,
        exploration_episodes=0,
        tensorboard=False,
        comment="",
    ):
        super().__init__(
            optimizer=optimizer,
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
        self.algorithm = REINFORCE(policy, baseline, criterion(reduction="mean"), gamma)
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
        baseline = NNValueFunction(
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
                {"params": baseline.parameters(), "lr": 1e-3},
            ]
        )
        criterion = nn.MSELoss

        return cls(
            policy=policy,
            baseline=baseline,
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
