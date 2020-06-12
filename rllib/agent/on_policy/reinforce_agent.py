"""Implementation of REINFORCE Algorithms."""

import torch.nn as nn

from rllib.algorithms.reinforce import REINFORCE

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

    def __init__(self, policy, optimizer, baseline=None, criterion=nn.MSELoss,
                 num_iter=1,
                 target_update_frequency=1, train_frequency=0, num_rollouts=1,
                 gamma=1.0, exploration_steps=0, exploration_episodes=0,
                 tensorboard=False, comment=''):
        super().__init__(optimizer=optimizer, num_iter=num_iter,
                         target_update_frequency=target_update_frequency,
                         train_frequency=train_frequency, num_rollouts=num_rollouts,
                         gamma=gamma, exploration_steps=exploration_steps,
                         exploration_episodes=exploration_episodes,
                         tensorboard=tensorboard, comment=comment)
        self.algorithm = REINFORCE(policy, baseline, criterion(reduction='mean'), gamma)
        self.policy = self.algorithm.policy
