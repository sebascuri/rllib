"""Q Learning Algorithm."""

import torch

from rllib.policy.q_function_policy import SoftMax
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

    def forward(self, observation):
        """Compute the loss and the td-error."""
        state, action, reward, next_state, done, *r = observation

        pred_q = self.q_function(state, action)

        with torch.no_grad():
            next_v = self.q_function(next_state).max(dim=-1)[0]
            target_q = reward + self.gamma * next_v * (1 - done)

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

    def forward(self, observation):
        """Compute the loss and the td-error."""
        state, action, reward, next_state, done, *r = observation

        pred_q = self.q_function(state, action)

        # target = r + gamma * max Q(x', a) and stop gradient.
        target_q = self.q_function(next_state).max(dim=-1)[0]
        target_q = reward + self.gamma * target_q * (1 - done)

        return self._build_return(pred_q, target_q)


class DQN(QLearning):
    r"""Implementation of Delayed Q Learning algorithm.

    The deep q-learning algorithm has a separate target network for the target value.

    Q_{target} = (r(s, a) + \gamma \max_a Q_{target}(s', a)).detach()

    References
    ----------
    Watkins, C. J., & Dayan, P. (1992). Q-learning. Machine learning, 8(3-4), 279-292.

    Mnih, Volodymyr, et al. "Human-level control through deep reinforcement learning."
    Nature 518.7540 (2015): 529-533.
    """

    def forward(self, observation):
        """Compute the loss and the td-error."""
        state, action, reward, next_state, done, *r = observation

        pred_q = self.q_function(state, action)

        with torch.no_grad():
            next_v = self.q_target(next_state).max(dim=-1)[0]
            target_q = reward + self.gamma * next_v * (1 - done)

        return self._build_return(pred_q, target_q)


class DDQN(QLearning):
    r"""Implementation of Double Delayed Q Learning algorithm.

    The double q-learning algorithm calculates the target value with the action that
    maximizes the primal function to mitigate over-estimation bias.

    a_{target} = \arg max_a Q(s', a)
    Q_{target} = (r(s, a) + \gamma \max_a Q_{target}(s', a_{target})).detach()

    References
    ----------
    Hasselt, H. V. (2010).
    Double Q-learning. NIPS.

    Van Hasselt, Hado, Arthur Guez, and David Silver. (2016)
    Deep reinforcement learning with double q-learning. AAAI.
    """

    def forward(self, observation):
        """Compute the loss and the td-error."""
        state, action, reward, next_state, done, *r = observation

        pred_q = self.q_function(state, action)

        with torch.no_grad():
            next_action = self.q_function(next_state).argmax(dim=-1)
            next_v = self.q_target(next_state, next_action)
            target_q = reward + self.gamma * next_v * (1 - done)

        return self._build_return(pred_q, target_q)


class SoftQLearning(QLearning):
    r"""Implementation of SoftQ-Learning algorithm.

    SoftQ-Learning is an off-policy model-free control algorithm.

    The SoftQ-Learning-Learning algorithm attempts to find the fixed point of:
    .. math:: Q(s, a) = r(s, a) + \gamma \tau \log \sum_a' \rho(a') \exp Q(s, a') / tau
    for some prior policy rho(a).

    Usually the loss is computed as:
    .. math:: Q_{target} = r + \gamma \tau \log \sum_a' \rho(a') \exp Q(s, a') / tau
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
    Fox, R., Pakman, A., & Tishby, N. (2015).
    Taming the noise in reinforcement learning via soft updates. UAI.

    Schulman, J., Chen, X., & Abbeel, P. (2017).
    Equivalence between policy gradients and soft q-learning.

    Haarnoja, T., Tang, H., Abbeel, P., & Levine, S. (2017).
    Reinforcement learning with deep energy-based policies. ICML.

    O'Donoghue, B., Munos, R., Kavukcuoglu, K., & Mnih, V. (2016).
    Combining policy gradient and Q-learning. ICLR.
    """

    def __init__(self, q_function, criterion, temperature, gamma):
        super().__init__(q_function, criterion, gamma)

        self.policy = SoftMax(self.q_function, temperature)
        self.policy_target = SoftMax(self.q_target, temperature)
        self.policy_target.param = self.policy.param

    def forward(self, observation):
        """Compute the loss and the td-error."""
        state, action, reward, next_state, done, *r = observation
        pred_q = self.q_function(state, action)

        # target = r + gamma \ tau * logsumexp(Q)
        with torch.no_grad():
            tau = self.policy.param()

            target_v = tau * torch.logsumexp(self.q_target(next_state) / tau, dim=-1)

            target_q = reward + self.gamma * target_v * (1 - done)

        return self._build_return(pred_q, target_q)

    def update(self):
        """Update the target network."""
        super().update()
        self.policy_target.param = self.policy.param
