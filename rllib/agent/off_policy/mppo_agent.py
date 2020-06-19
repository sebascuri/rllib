"""MPPO Agent Implementation."""
from rllib.algorithms.mppo import MPPO

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
        gamma=1.0,
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
            gamma=gamma,
            exploration_steps=exploration_steps,
            exploration_episodes=exploration_episodes,
            tensorboard=tensorboard,
            comment=comment,
        )
