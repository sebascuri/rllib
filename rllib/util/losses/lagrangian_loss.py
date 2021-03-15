"""Lagrangian Loss Helper Module."""
import torch.nn as nn

from rllib.dataset.datatypes import Loss
from rllib.util.parameter_decay import Constant, Learnable, ParameterDecay


class LagrangianLoss(nn.Module):
    r"""Lagrangian Loss Module.

    The method implements a lagrangian loss for an inequality of type
    g(x) - c < \epsilon, or a regularization of type \epsilon g(x)

    In regularization mode, it returns a loss given by:
    ..math :: Loss = \epsilon g(x)

    In constraint mode (regularization=False), it returns a loss given by:
    ..math :: Loss = \epsilon (c - g(x)),
    where \epsilon is a learnable lagrangian parameter.

    To propagate gradients in constraint mode, it separates the loss as:
    regularization_loss = \epsilon.detach() g(x)
    dual_loss = \epsilon (c - g(x)).detach()

    Parameters
    ----------
    eta: float.
    entropy regularization parameter.
    regularization: bool
        Flag that indicates if the algorithm is in regularization or trust-region mode.

    """

    def __init__(self, dual=0.0, inequality_zero=0.0, regularization=False):
        super().__init__()
        # Actor
        self.inequality_zero = inequality_zero
        self.regularization = regularization

        if regularization:  # Regularization: \dual g(x)
            if not isinstance(dual, ParameterDecay):
                dual = Constant(dual)
            self._dual = dual
        else:  # Constraint: g(x) < epsilon
            if isinstance(dual, ParameterDecay):
                dual = dual()
            self._dual = Learnable(dual, positive=True)

    @property
    def dual(self):
        """Get dual parameter, detached."""
        return self._dual().detach()

    def forward(self, inequality_value):
        """Return primal and dual loss terms from entropy loss.

        Parameters
        ----------
        inequality_value: torch.tensor.
        """
        if self.inequality_zero == 0.0 and not self.regularization:
            return Loss()
        dual_loss = self._dual() * (self.inequality_zero - inequality_value).detach()
        reg_loss = self._dual().detach() * inequality_value
        return Loss(dual_loss=dual_loss, reg_loss=reg_loss)
