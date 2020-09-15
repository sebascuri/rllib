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
    critic: AbstractQFunction
        Q_function to optimize.
    criterion: _Loss
        Criterion to optimize.
    gamma: float
        Discount factor.

    References
    ----------
    Fox, R., Pakman, A., & Tishby, N. (2015).
    Taming the noise in reinforcement learning via soft updates. UAI.

    Schulman, J., Chen, X., ,

    Haarnoja, T., Tang, H., Abbeel, P., & Levine, S. (2017).
    Reinforcement learning with deep energy-based policies. ICML.

    O'Donoghue, B., Munos, R., Kavukcuoglu, K., & Mnih, V. (2016).
    Combining policy gradient and Q-learning. ICLR.
    """

    def __init__(self, critic, temperature, *args, **kwargs):
        super().__init__(
            policy=SoftMax(critic, temperature), critic=critic, *args, **kwargs
        )
        self.policy_target.param = self.policy.param  # Hard copy the parameter.

    def get_value_target(self, observation):
        """Get q function target."""
        temperature = self.policy.param()
        next_v = temperature * torch.logsumexp(
            self.critic_target(observation.next_state) / temperature, dim=-1
        )
        next_v = next_v * (1 - observation.done)
        return self.get_reward(observation) + self.gamma * next_v
