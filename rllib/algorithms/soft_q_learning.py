"""Soft Q-Learning Algorithm."""

import torch

from rllib.policy.q_function_policy import SoftMax

from .q_learning import QLearning


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

    def get_q_target(self, reward, next_state, done):
        """Get q function target."""
        tau = self.policy.param()
        target_v = tau * torch.logsumexp(self.q_target(next_state) / tau, dim=-1)
        target_q = reward + self.gamma * target_v * (1 - done)
        return target_q

    def update(self):
        """Update the target network."""
        super().update()
        self.policy_target.param = self.policy.param
