"""Entropy Loss Helper Module."""
import torch.nn as nn

from rllib.dataset.datatypes import Loss
from rllib.util.parameter_decay import Constant, Learnable, ParameterDecay


class EntropyLoss(nn.Module):
    r"""Entropy Loss.

    The method implements an entropy loss.

    In regularization mode, it returns a loss given by:
    ..math :: Loss = -\eta entropy.

    In trust-region mode (regularization=False), it returns a loss given by:
    ..math :: Loss = \eta (entropy - target_entropy).


    Parameters
    ----------
    eta: float.
    target_entropy: ParameterDecay.
    regularization: bool
        Flag that indicates if the algorithm is in regularization or trust-region mode.

    References
    ----------
    Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018).
    Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a
    Stochastic Actor. ICML.

    Haarnoja, T., Zhou, A., ... & Levine, S. (2018).
    Soft actor-critic algorithms and applications. arXiv.
    """

    def __init__(self, eta=0.0, target_entropy=0.0, regularization=True):
        super().__init__()
        # Actor
        self.target_entropy = target_entropy  # -self.policy.dim_action[0]
        self.regularization = regularization

        if regularization:  # Regularization: \eta KL(\pi || Uniform)
            if not isinstance(eta, ParameterDecay):
                eta = Constant(eta)
            self._eta = eta
        else:  # Trust-Region: || KL(\pi || Uniform) - target|| < \epsilon
            if isinstance(eta, ParameterDecay):
                eta = eta()
            self._eta = Learnable(eta, positive=True)

    @property
    def eta(self):
        """Get regularization parameter."""
        return self._eta().detach()

    def forward(self, entropy):
        """Return primal and dual loss terms from entropy loss.

        Parameters
        ----------
        entropy: torch.tensor.
        """
        if self.target_entropy == 0.0 and not self.regularization:
            return Loss()
        dual_loss = self._eta() * (entropy - self.target_entropy).detach()
        reg_loss = -self.eta * entropy
        return Loss(dual_loss=dual_loss, reg_loss=reg_loss)
