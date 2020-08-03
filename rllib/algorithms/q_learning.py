"""Q Learning Algorithm."""

import torch

from rllib.util.neural_networks import deep_copy_module, update_parameters

from .abstract_algorithm import AbstractAlgorithm, TDLoss


class QLearning(AbstractAlgorithm):
    r"""Implementation of Q-Learning algorithm.

    Q-Learning is an off-policy model-free control algorithm.

    The Q-Learning algorithm attempts to find the fixed point of:
    .. math:: Q(s, a) = r(s, a) + \gamma \max_a Q(s', a)

    Usually the loss is computed as:
    .. math:: Q_{target} = r(s, a) + \gamma \max_a Q(s', a)
    .. math:: \mathcal{L}(Q(s, a), Q_{target})

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
    Watkins, C. J. C. H. (1989).
    Learning from delayed rewards.

    Watkins, C. J., & Dayan, P. (1992).
    Q-learning. Machine learning.

    Jaakkola, T., Jordan, M. I., & Singh, S. P. (1994).
    Convergence of stochastic iterative dynamic programming algorithms. NIPS.

    Tsitsiklis, J. N. (1994). Asynchronous stochastic approximation and Q-learning.
    Machine learning.

    Mnih, V., et. al. (2013).
    Playing atari with deep reinforcement learning. NIPS.
    """

    def __init__(self, q_function, criterion, gamma):
        super().__init__()
        self.q_function = q_function
        self.q_target = deep_copy_module(q_function)
        self.criterion = criterion
        self.gamma = gamma

    def get_q_target(self, reward, next_state, done):
        """Get q function target."""
        n_step = reward.shape[-1]
        gamma = torch.pow(
            torch.tensor(self.gamma, dtype=torch.get_default_dtype()),
            torch.arange(n_step),
        )
        n_step_return = (reward * gamma).sum(-1)

        if not self.q_function.discrete_state:
            final_state = next_state[..., -1, :]
        else:
            final_state = next_state[..., -1]

        final_v = self.q_function(final_state).max(dim=-1)[0]
        target_q = n_step_return + self.gamma ** n_step * final_v * (1 - done[..., -1])
        return target_q

    def forward(self, observation):
        """Compute the loss and the td-error."""
        state, action, reward, next_state, done, *r = observation

        if not self.q_function.discrete_state:
            state = state[..., 0, :]
        else:
            state = state[..., 0]
        action = action[..., 0]

        pred_q = self.q_function(state, action)
        with torch.no_grad():
            target_q = self.get_q_target(reward, next_state, done)

        return self._build_return(pred_q, target_q)

    def _build_return(self, pred_q, target_q):
        return TDLoss(
            loss=self.criterion(pred_q, target_q).squeeze(-1),
            td_error=(pred_q - target_q).detach().squeeze(-1),
        )

    def update(self):
        """Update the target network."""
        update_parameters(self.q_target, self.q_function, tau=self.q_function.tau)


class GradientQLearning(QLearning):
    r"""Implementation of Gradient Q Learning algorithm.

    The gradient q-learning algorithm propagates the gradient on both prediction and
    target estimates.

    .. math:: Q_{target} = (r(s, a) + \gamma \max_a Q(s', a))

    References
    ----------
    Watkins, C. J., & Dayan, P. (1992).
    Q-learning. Machine learning.
    """

    def get_target(self, reward, next_state, done):
        """Get q function target."""
        with torch.enable_grad():  # Require gradient after it's been disabled.
            next_v = self.q_function(next_state).max(dim=-1)[0]
            target_q = reward + self.gamma * next_v * (1 - done)
            return target_q
