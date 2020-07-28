"""MPPO Agent Implementation."""
from itertools import chain

import torch.nn.modules.loss as loss
from torch.optim import Adam

from rllib.algorithms.mppo import MPPO
from rllib.dataset.experience_replay import ExperienceReplay
from rllib.policy import NNPolicy
from rllib.value_function import NNQFunction

from .off_policy_agent import OffPolicyAgent


class MPPOAgent(OffPolicyAgent):
    """Implementation of an agent that runs MPPO."""

    def __init__(
        self,
        policy,
        q_function,
        optimizer,
        memory,
        criterion,
        num_action_samples=15,
        epsilon=0.1,
        epsilon_mean=0.1,
        epsilon_var=0.001,
        regularization=False,
        num_iter=100,
        batch_size=64,
        target_update_frequency=4,
        train_frequency=0,
        num_rollouts=1,
        policy_update_frequency=1,
        gamma=0.99,
        exploration_steps=0,
        exploration_episodes=0,
        tensorboard=False,
        comment="",
    ):
        self.algorithm = MPPO(
            policy=policy,
            q_function=q_function,
            num_action_samples=num_action_samples,
            criterion=criterion,
            epsilon=epsilon,
            epsilon_mean=epsilon_mean,
            epsilon_var=epsilon_var,
            regularization=regularization,
            gamma=gamma,
        )

        self.policy = self.algorithm.policy
        optimizer = type(optimizer)(
            [
                p
                for name, p in self.algorithm.named_parameters()
                if "target" not in name
            ],
            **optimizer.defaults,
        )
        super().__init__(
            memory=memory,
            optimizer=optimizer,
            num_iter=num_iter,
            target_update_frequency=target_update_frequency,
            batch_size=batch_size,
            train_frequency=train_frequency,
            num_rollouts=num_rollouts,
            policy_update_frequency=policy_update_frequency,
            gamma=gamma,
            exploration_steps=exploration_steps,
            exploration_episodes=exploration_episodes,
            tensorboard=tensorboard,
            comment=comment,
        )

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
        q_function = NNQFunction(
            dim_state=environment.dim_state,
            dim_action=environment.dim_action,
            num_states=environment.num_states,
            num_actions=environment.num_actions,
            layers=[200, 200],
            biased_head=True,
            non_linearity="ReLU",
            tau=5e-3,
            input_transform=None,
        )
        policy = NNPolicy(
            dim_state=environment.dim_state,
            dim_action=environment.dim_action,
            num_states=environment.num_states,
            num_actions=environment.num_actions,
            action_scale=environment.action_scale,
            goal=environment.goal,
            layers=[100, 100],
            biased_head=True,
            non_linearity="ReLU",
            tau=5e-3,
            input_transform=None,
            deterministic=False,
        )

        optimizer = Adam(chain(policy.parameters(), q_function.parameters()), lr=5e-4)
        criterion = loss.MSELoss
        memory = ExperienceReplay(max_len=50000, num_steps=0)

        if environment.num_actions > 0:
            epsilon = 0.1
            epsilon_mean = 0.5
            epsilon_var = None
        else:
            epsilon = 0.1
            epsilon_mean = 0.1
            epsilon_var = 1e-4

        return cls(
            policy,
            q_function,
            optimizer,
            memory,
            criterion,
            num_action_samples=15,
            epsilon=epsilon,
            epsilon_mean=epsilon_mean,
            epsilon_var=epsilon_var,
            regularization=False,
            num_iter=5 if test else 1000,
            batch_size=100,
            target_update_frequency=1,
            train_frequency=0,
            num_rollouts=2,
            gamma=gamma,
            exploration_steps=exploration_steps,
            exploration_episodes=exploration_episodes,
            tensorboard=tensorboard,
            comment=environment.name,
        )
