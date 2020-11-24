"""MPC Algorithms."""

import torch
from torch.distributions import MultivariateNormal

from rllib.util.utilities import sample_mean_and_cov

from .abstract_solver import MPCSolver


class CEMShooting(MPCSolver):
    r"""Cross Entropy Method solves the MPC problem by adaptively sampling.

    The sampling distribution is adapted by fitting a Multivariate Gaussian to the
    best `num_elites' samples (action sequences) for `num_iter' times.

    Parameters
    ----------
    dynamical_model: state transition model.
    reward_model: reward model.
    horizon: int.
        Horizon to solve planning problem.
    gamma: float, optional.
        Discount factor.
    num_iter: int, optional.
        Number of iterations of CEM method.
    num_samples: int, optional.
        Number of samples for shooting method.
    num_elites: int, optional.
        Number of elite samples to keep between iterations.
    alpha: float, optional. (default = 0.)
        Low pass filter of mean and covariance update.
    termination: Callable, optional.
        Termination condition.
    terminal_reward: terminal reward model, optional.
    warm_start: bool, optional.
        Whether or not to start the optimization with a warm start.
    default_action: str, optional.
         Default action behavior.

    References
    ----------
    Chua, K., Calandra, R., McAllister, R., & Levine, S. (2018).
    Deep reinforcement learning in a handful of trials using probabilistic dynamics
    models. NeuRIPS.

    Botev, Z. I., Kroese, D. P., Rubinstein, R. Y., & L’Ecuyer, P. (2013).
    The cross-entropy method for optimization. In Handbook of statistics
    """

    def __init__(self, alpha=0.0, num_iter=5, num_elites=None, *args, **kwargs):
        super().__init__(num_iter=num_iter, *args, **kwargs)
        self.num_elites = (
            max(1, self.num_samples // 10) if not num_elites else num_elites
        )
        self.alpha = alpha

    def get_candidate_action_sequence(self):
        """Get candidate actions by sampling from a multivariate normal."""
        action_distribution = MultivariateNormal(self.mean, self.covariance)
        action_sequence = action_distribution.sample((self.num_samples,))
        action_sequence = action_sequence.permute(
            tuple(torch.arange(1, action_sequence.dim() - 1)) + (0, -1)
        )
        if self.clamp:
            return action_sequence.clamp(-1.0, 1.0)
        return action_sequence

    def get_best_action(self, action_sequence, returns):
        """Get best action by averaging the num_elites samples."""
        idx = torch.topk(returns, k=self.num_elites, largest=True, dim=-1)[1]
        idx = idx.unsqueeze(0).unsqueeze(-1)  # Expand dims to action_sequence.
        idx = idx.repeat_interleave(self.horizon, 0).repeat_interleave(
            self.dim_action, -1
        )
        return torch.gather(action_sequence, -2, idx)

    def update_sequence_generation(self, elite_actions):
        """Update distribution by the empirical mean and covariance of best actions."""
        new_mean, new_cov = sample_mean_and_cov(elite_actions.transpose(-1, -2))
        self.mean = self.alpha * self.mean + (1 - self.alpha) * new_mean
        self.covariance = self.alpha * self.covariance + (1 - self.alpha) * new_cov
