"""Implementation of SARSA Algorithms."""

import torch
import torch.nn.modules.loss as loss
from torch.optim import Adam

from rllib.algorithms.sarsa import SARSA
from rllib.policy import EpsGreedy
from rllib.util.parameter_decay import ExponentialDecay
from rllib.value_function import NNQFunction

from .on_policy_agent import OnPolicyAgent


class SARSAAgent(OnPolicyAgent):
    """Implementation of a SARSA agent.

    The SARSA agent implements the SARSA algorithm except for the
    computation of the TD-Error, which leads to different algorithms.

    Parameters
    ----------
    critic: AbstractQFunction
        critic that is learned.
    policy: QFunctionPolicy.
        Q-function derived policy.
    criterion: nn.Module
        Criterion to minimize the TD-error.
    batch_size: int
        Number of trajectory batches before performing a TD-pdate.
    optimizer: nn.optim
        Optimization algorithm for q_function.
    target_update_frequency: int
        How often to update the q_function target.
    gamma: float, optional
        Discount factor.
    exploration_steps: int, optional
        Number of random exploration steps.
    exploration_episodes: int, optional
        Number of random exploration steps.

    References
    ----------
    Watkins, C. J., & Dayan, P. (1992). Q-learning. Machine learning, 8(3-4), 279-292.
    """

    def __init__(
        self,
        critic,
        policy,
        algorithm_=SARSA,
        criterion=loss.MSELoss,
        batch_size=1,
        num_rollouts=0,
        train_frequency=1,
        *args,
        **kwargs,
    ):
        super().__init__(
            batch_size=batch_size + 1,
            num_rollouts=num_rollouts,
            train_frequency=train_frequency,
            *args,
            **kwargs,
        )
        # batch_size + 1 as it will always remove the last observation before training.

        self.algorithm = algorithm_(
            policy=policy,
            critic=critic,
            criterion=criterion(reduction="mean"),
            *args,
            **kwargs,
        )
        self.policy = policy

    def act(self, state):
        """See `AbstractAgent.act'."""
        action = super().act(state)
        if len(self.trajectories[-1]):
            self.trajectories[-1][-1].next_action = torch.tensor(action)

        return action

    def learn(self):
        """Remove the last observation."""
        self.trajectories[-1].pop(-1)
        super().learn()

    @classmethod
    def default(
        cls,
        environment,
        critic=None,
        policy=None,
        epsilon=None,
        lr=3e-4,
        *args,
        **kwargs,
    ):
        """See `AbstractAgent.default'."""
        if critic is None:
            critic = NNQFunction.default(environment, tau=0)
        if policy is None:
            if epsilon is None:
                epsilon = ExponentialDecay(start=1.0, end=0.01, decay=500)
            policy = EpsGreedy(critic, epsilon)
        optimizer = Adam(critic.parameters(), lr=lr)

        return super().default(
            environment,
            critic=critic,
            policy=policy,
            optimizer=optimizer,
            *args,
            **kwargs,
        )
