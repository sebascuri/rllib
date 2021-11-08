"""Q Learning Algorithm."""

import torch

from rllib.util.neural_networks.utilities import broadcast_to_tensor

from .abstract_algorithm import AbstractAlgorithm


class QLearning(AbstractAlgorithm):
    r"""Implementation of Q-Learning algorithm.

    Q-Learning is an off-policy model-free control algorithm.

    The Q-Learning algorithm attempts to find the fixed point of:
    .. math:: Q(s, a) = r(s, a) + \gamma \max_a Q(s', a)

    Usually the loss is computed as:
    .. math:: Q_{target} = r(s, a) + \gamma \max_a Q(s', a)
    .. math:: \mathcal{L}(Q(s, a), Q_{target})

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

    def compute_optimal_target(self, q_function, observation):
        """Compute q target with the q function."""
        q_value = q_function(observation.next_state)
        q_value = self.multi_objective_reduction(q_value)
        next_v = q_value.max(dim=-1)[0]
        reward = self.get_reward(observation)
        next_v = broadcast_to_tensor(next_v, target_tensor=reward)
        not_done = broadcast_to_tensor(1.0 - observation.done, target_tensor=reward)
        next_v = next_v * not_done
        return reward + self.gamma * next_v

    def get_value_target(self, observation):
        """Get q function target."""
        return self.compute_optimal_target(
            q_function=self.critic, observation=observation
        )


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

    def get_value_target(self, observation):
        """Get q function target."""
        with torch.enable_grad():  # Require gradient after it's been disabled.
            return super().get_value_target(observation)
