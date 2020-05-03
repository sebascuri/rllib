"""N-Step TD Learning Algorithm."""
import copy

import torch
import torch.nn as nn

from rllib.util.neural_networks import update_parameters
from rllib.util import discount_sum, mb_return
from .q_learning import QLearningLoss


class TDLearning(nn.Module):
    r"""Implementation of TD-Learning algorithm.

    TD is a policy-evaluation method algorithm.

    The TD algorithm attempts to find the fixed point of:
    .. math:: V(s) = r(s, a) + \gamma V(s')
    where a is sampled from the current policy and s' \sim P(s, a).

    Usually the loss is computed as:
    .. math:: V_{target} = \sum_{n=0}^{N-1} r(s_n, a_n) + \gamma^N V(s_N)
    .. math:: \mathcal{L}(V(s), V_{target})

    Parameters
    ----------
    value_function: AbstractValueFunction
        Value Function to optimize.
    criterion: _Loss
        Criterion to optimize.
    gamma: float
        Discount factor.
    n_steps: int, optional.
        Number of steps to optimize.

    References
    ----------
    Sutton, R. S. (1988).
    Learning to predict by the methods of temporal differences. Machine learning.
    """

    def __init__(self, value_function, criterion, gamma):
        super().__init__()
        self.value_function = value_function
        self.value_target = copy.deepcopy(value_function)
        self.criterion = criterion
        self.gamma = gamma
        # self.n_steps = n_steps

    def forward(self, state, action, reward, next_state, done):
        """Compute the loss and the td-error."""
        n_steps = state.shape[1]
        pred_v = self.value_function(state[:, 0])

        with torch.no_grad():
            returns = discount_sum(reward.transpose(0, 1), self.gamma)
            final_state = next_state[:, -1]
            next_v = self.value_target(final_state)

            target_v = returns + self.gamma ** n_steps * next_v * (1 - done[:, -1])

        return self._build_return(pred_v, target_v)

    def _build_return(self, pred_v, target_v):
        return QLearningLoss(loss=self.criterion(pred_v, target_v),
                             td_error=(pred_v - target_v).detach())

    def update(self):
        """Update the target network."""
        update_parameters(self.value_target, self.value_function,
                          tau=self.value_function.tau)


class ModelBasedTDLearning(TDLearning):
    r"""Implementation of Model-Based TD-Learning algorithm.

    TD is a policy-evaluation method algorithm.

    The TD algorithm attempts to find the fixed point of:
    .. math:: V(s) = r(s, a) + \gamma V(s')
    where a is sampled from the current policy and s' \sim Model(s, a).

    Usually the loss is computed as:
    .. math:: V_{target} = \sum_{n=0}^{N-1} r(s_n, a_n) + \gamma^N V(s_N)
    .. math:: \mathcal{L}(V(s), V_{target})

    Parameters
    ----------
    value_function: AbstractValueFunction
        Value Function to optimize.
    criterion: _Loss
        Criterion to optimize.
    gamma: float
        Discount factor.
    num_steps: int, optional.
        Number of steps to optimize.

    References
    ----------
    Sutton, R. S. (1988).
    Learning to predict by the methods of temporal differences. Machine learning.

    Lowrey, K., Rajeswaran, A., Kakade, S., Todorov, E., & Mordatch, I. (2018).
    Plan online, learn offline: Efficient learning and exploration via model-based
    control. ICLR.

    """

    def __init__(self, value_function, criterion, policy, dynamical_model, reward_model,
                 termination=None, num_steps=1, gamma=0.99):
        super().__init__(value_function, gamma=gamma, criterion=criterion)
        self.policy = policy
        self.dynamical_model = dynamical_model
        self.reward_model = reward_model
        self.num_steps = num_steps
        self.termination = termination

    def forward(self, state, *args):
        """Compute the loss and the td-error."""
        state = state[:, 0]
        pred_v = self.value_function(state)

        with torch.no_grad():
            value_estimate, trajectory = mb_return(
                state, dynamical_model=self.dynamical_model, policy=self.policy,
                reward_model=self.reward_model, num_steps=self.num_steps,
                value_function=self.value_target, gamma=self.gamma,
                termination=self.termination)

        return self._build_return(pred_v, value_estimate)
