"""Useful distributions for the library."""

import math

import gpytorch
import torch
from torch.distributions import constraints
from torch.distributions.transforms import Transform
from torch.nn.functional import softplus


class Delta(gpytorch.distributions.Delta):
    """Delta Distribution."""

    def __init__(self, validate_args=False, *args, **kwargs):
        super().__init__(validate_args=validate_args, *args, **kwargs)

    def __str__(self):
        """Get string of Delta distribution."""
        return f"Delta loc: {self.v}"

    def entropy(self):
        """Return entropy of distribution."""
        return torch.zeros(self.batch_shape)


class TanhTransform(Transform):
    r"""Transform via the mapping :math:`y = \tanh(x)`.

    It is equivalent to
    ```
    ComposeTransform([AffineTransform(0., 2.),
                      SigmoidTransform(),
                      AffineTransform(-1., 2.)
                      ])
    ```
    However this might not be numerically stable, thus it is recommended to use
    `TanhTransform` instead.
    Note that one should use `cache_size=1` when it comes to `NaN/Inf` values.

    Notes
    -----
    This class should be released in the next version of pytorch.
    """

    domain = constraints.real
    codomain = constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    @staticmethod
    def atanh(x):
        """Compute arctanh."""
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        """Check if transforms are equal."""
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of
        # certain algorithms. One should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        """Compute the log det jacobian `log |dy/dx|` given input and output.

        References
        ----------
        https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/bijectors/tanh.py#L69-L80
        """
        return 2.0 * (math.log(2.0) - x - softplus(-2.0 * x))
