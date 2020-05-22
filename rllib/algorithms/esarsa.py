"""Expected SARSA Algorithm."""

import torch

from rllib.util.utilities import integrate, tensor_to_distribution
from rllib.util.neural_networks import deep_copy_module, update_parameters
from .abstract_algorithm import TDLoss, AbstractAlgorithm


class ESARSA(AbstractAlgorithm):
    r"""Implementation of Expected SARSA algorithm.

    SARSA is an on-policy model-free control algorithm
    The expected SARSA algorithm computes the target by integrating the next action:

    .. math:: Q_{target} = r(s, a) + \gamma \sum_{a'} \pi(a')  Q(s', a')

    Parameters
    ----------
    q_function: AbstractQFunction
        Q_function to optimize.
    criterion: _Loss
        Criterion to optimize.
    gamma: float
        Discount factor.

    References
    ----------
    Van Seijen, H., Van Hasselt, H., Whiteson, S., & Wiering, M. (2009).
    A theoretical and empirical analysis of Expected Sarsa. IEEE.

    Van Hasselt, H. P. (2011).
    Insights in reinforcement learning: formal analysis and empirical evaluation of
    temporal-difference learning algorithms. Utrecht University.
    """

    def __init__(self, q_function, criterion, policy, gamma):
        super().__init__()
        self.q_function = q_function
        self.q_target = deep_copy_module(q_function)
        self.policy = policy
        self.criterion = criterion
        self.gamma = gamma

    def _build_return(self, pred_q, target_q):
        return TDLoss(loss=self.criterion(pred_q, target_q).squeeze(-1),
                      td_error=(pred_q - target_q).detach().squeeze(-1))

    def forward(self, trajectories):
        """Compute the loss and the td-error."""
        trajectory = trajectories[0]
        state, action = trajectory.state, trajectory.action
        reward, done = trajectory.reward, trajectory.done
        next_state = trajectory.next_state

        pred_q = self.q_function(state, action)
        with torch.no_grad():
            pi = tensor_to_distribution(self.policy(state))
            next_v = integrate(lambda a: self.q_target(next_state, a), pi)
            target_q = reward + self.gamma * next_v * (1 - done)

        return self._build_return(pred_q, target_q)

    def update(self):
        """Update Q target."""
        update_parameters(self.q_target, self.q_function, tau=self.q_function.tau)


class GradientESARSA(ESARSA):
    r"""Implementation of Gradient-Expected SARSA algorithm.

    The semi-gradient expected SARSA algorithm computes the target by integrating the
    next action and detaching the gradient.

    .. math:: Q_{target} = (r(s, a) + \gamma \sum_{a'} \pi(a')  Q(s', a')).detach()

    References
    ----------
    TODO: Find.
    """

    def forward(self, trajectories):
        """Compute the loss and the td-error."""
        trajectory = trajectories[0]
        state, action = trajectory.state, trajectory.action
        reward, done = trajectory.reward, trajectory.done
        next_state = trajectory.next_state

        pred_q = self.q_function(state, action)
        pi = tensor_to_distribution(self.policy(state))
        next_v = integrate(lambda a: self.q_function(next_state, a), pi)
        target_q = reward + self.gamma * next_v * (1 - done)

        return self._build_return(pred_q, target_q)
