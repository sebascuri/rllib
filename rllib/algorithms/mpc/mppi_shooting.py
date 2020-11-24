"""MPC Algorithms."""
import torch
from torch.distributions import MultivariateNormal

from rllib.util.parameter_decay import Constant, ParameterDecay

from .abstract_solver import MPCSolver


class MPPIShooting(MPCSolver):
    """Solve MPC using Model Predictive Path Integral control.

    References
    ----------
    Williams, G., Drews, P., Goldfain, B., Rehg, J. M., & Theodorou, E. A. (2016).
    Aggressive driving with model predictive path integral control. ICRA.

    Williams, G., Aldrich, A., & Theodorou, E. (2015).
    Model predictive path integral control using covariance variable importance
    sampling. arXiv.

    Nagabandi, A., Konoglie, K., Levine, S., & Kumar, V. (2019).
    Deep Dynamics Models for Learning Dexterous Manipulation. arXiv.

    """

    def __init__(self, kappa=1.0, filter_coefficients=(0.25, 0.8, 0), *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not isinstance(kappa, ParameterDecay):
            kappa = Constant(kappa)
        self.kappa = kappa
        self.filter_coefficients = torch.tensor(filter_coefficients)
        self.filter_coefficients /= torch.sum(self.filter_coefficients)

    def get_candidate_action_sequence(self):
        """Get candidate actions by sampling from a multivariate normal."""
        noise_dist = MultivariateNormal(torch.zeros_like(self.mean), self.covariance)
        noise = noise_dist.sample((self.num_samples,))

        lag = len(self.filter_coefficients)
        for i in range(self.horizon):
            weights = self.filter_coefficients[: min(i + 1, lag)]
            aux = torch.einsum(
                "i, ki...j-> k...j",
                weights.flip(0),
                noise[:, max(0, i - lag + 1) : i + 1, ..., :],
            )
            noise[:, i, ..., :] = aux / torch.sum(weights)

        action_sequence = self.mean.unsqueeze(0).repeat_interleave(self.num_samples, 0)
        action_sequence += noise
        action_sequence = action_sequence.permute(
            tuple(torch.arange(1, action_sequence.dim() - 1)) + (0, -1)
        )
        if self.clamp:
            return action_sequence.clamp(-1.0, 1.0)
        return action_sequence

    def get_best_action(self, action_sequence, returns):
        """Get best action by a weighted average of e^kappa returns."""
        returns = self.kappa() * returns
        weights = torch.exp(returns - torch.max(returns))
        normalization = weights.sum()

        weights = weights.unsqueeze(0).unsqueeze(-1)
        weights = weights.repeat_interleave(self.horizon, 0).repeat_interleave(
            self.dim_action, -1
        )
        return (weights * action_sequence).sum(dim=-2) / normalization

    def update_sequence_generation(self, elite_actions):
        """Update distribution by the fitting the elite_actions to the mean."""
        self.mean = elite_actions
        self.kappa.update()
