"""KL-Constraint Helper Module."""
import torch
import torch.distributions
import torch.nn as nn

from rllib.dataset.datatypes import Loss
from rllib.util.parameter_decay import Constant, Learnable, ParameterDecay


class KLLoss(nn.Module):
    r"""KL-Constraint Loss.

    The method tries to approximate a KL-constraint KL(p, q) <= \epsilon.

    In regularization mode, it returns a loss given by:
    ..math :: Loss = \epsilon KL(p, q).

    In trust-region mode (regularization=False), it returns a loss given by:
    ..math :: Loss = \eta() (\epsilon - KL(p, q).detach()).

    In order to have a tighter control on the mean and variance of Normal distribution,
    the KL-Divergence associated with the mean and the variance of a distribution are
    computed separately. See `rllib.util.utilities.separated_kl`.

    Parameters
    ----------
    epsilon_mean: Union[ParameterDecay, float].
        The KL-divergence for the mean component in the M-step (fitting the policy).
        See `rllib.util.utilities.separated_kl`.
    epsilon_var: Union[ParameterDecay, float], optional.
        The KL-divergence for the variance component in the M-step (fitting the policy).
        See `rllib.util.utilities.separated_kl`.
    regularization: bool
        Flag that indicates if the algorithm is in regularization or trust-region mode.

    References
    ----------
    Kakade, S., & Langford, J. (2002, July).
    Approximately optimal approximate reinforcement learning. ICML.

    Abdolmaleki, A., et al. (2018).
    Maximum a posteriori policy optimisation. ICLR.

    Schulman, J., Levine, S., Abbeel, P., Jordan, M., & Moritz, P. (2015).
    Trust region policy optimization. ICML.
    """

    def __init__(self, epsilon_mean=0.0, epsilon_var=0.0, regularization=False):
        super().__init__()
        if epsilon_var is None:
            self.separated_kl = False
            epsilon_var = 0.0
        else:
            self.separated_kl = True

        self.regularization = regularization
        if self.regularization:
            eta_mean = epsilon_mean
            eta_var = epsilon_var
            if not isinstance(eta_mean, ParameterDecay):
                eta_mean = Constant(eta_mean)
            if not isinstance(eta_var, ParameterDecay):
                eta_var = Constant(eta_var)

            self._eta_mean = eta_mean
            self._eta_var = eta_var

            self.epsilon_mean = torch.tensor(0.0)
            self.epsilon_var = torch.tensor(0.0)

        else:  # Trust-Region: || KL(q || \pi_old) || < \epsilon
            self._eta_mean = Learnable(1.0, positive=True)
            self._eta_var = Learnable(1.0, positive=True)

            self.epsilon_mean = torch.tensor(epsilon_mean)
            self.epsilon_var = torch.tensor(epsilon_var)

    @property
    def eta_mean(self):
        """Get eta parameter."""
        return self._eta_mean().detach()

    @property
    def eta_var(self):
        """Get eta parameter."""
        return self._eta_var().detach()

    def forward(self, kl_mean, kl_var=None):
        """Return primal and dual loss terms from MMPO.

        Parameters
        ----------
        kl_mean : torch.Tensor
            A float corresponding to the KL divergence.
        kl_var : torch.Tensor
            A float corresponding to the KL divergence.
        """
        if self.epsilon_mean == 0.0 and not self.regularization:
            return Loss()
        if kl_var is None:
            kl_var = torch.zeros_like(kl_mean)

        kl_mean, kl_var = kl_mean.mean(), kl_var.mean()
        reg_loss = self.eta_mean * kl_mean + self.eta_var * kl_var
        if self.regularization:
            return Loss(reg_loss=reg_loss)
        else:
            if self.separated_kl:
                mean_loss = self._eta_mean() * (self.epsilon_mean - kl_mean).detach()
                var_loss = self._eta_var() * (self.epsilon_var - kl_var).detach()
                dual_loss = mean_loss + var_loss
            else:
                dual_loss = self._eta_mean() * (self.epsilon_mean - kl_mean).detach()

            return Loss(dual_loss=dual_loss, reg_loss=reg_loss)
