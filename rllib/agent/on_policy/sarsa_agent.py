"""Implementation of SARSA Algorithms."""

import torch

from rllib.algorithms.sarsa import SARSA

from .on_policy_agent import OnPolicyAgent


class SARSAAgent(OnPolicyAgent):
    """Implementation of a SARSA agent.

    The SARSA agent implements the SARSA algorithm except for the
    computation of the TD-Error, which leads to different algorithms.

    Parameters
    ----------
    q_function: AbstractQFunction
        q_function that is learned.
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

    def __init__(self, q_function, policy, criterion, optimizer,
                 num_iter=1, batch_size=1, target_update_frequency=1,
                 train_frequency=1, num_rollouts=0,
                 gamma=1.0, exploration_steps=0, exploration_episodes=0,
                 tensorboard=False, comment=''):
        super().__init__(optimizer=optimizer, num_iter=num_iter,
                         batch_size=batch_size + 1,
                         target_update_frequency=target_update_frequency,
                         train_frequency=train_frequency, num_rollouts=num_rollouts,
                         gamma=gamma, exploration_steps=exploration_steps,
                         exploration_episodes=exploration_episodes,
                         tensorboard=tensorboard, comment=comment)
        # batch_size + 1 as it will always remove the last observation before training.

        self.algorithm = SARSA(q_function, criterion(reduction='mean'), gamma)
        self.policy = policy

    def act(self, state):
        """See `AbstractAgent.act'."""
        action = super().act(state)
        if len(self.trajectories[-1]):
            last_observation = self.trajectories[-1].pop(-1)
            self.trajectories[-1].append(
                last_observation._replace(next_action=torch.tensor(action))
            )

        return action

    def _train(self):
        """Remove the last observation."""
        self.trajectories[-1].pop(-1)
        super()._train()
