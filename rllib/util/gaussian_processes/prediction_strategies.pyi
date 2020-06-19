from typing import Optional

from gpytorch.distributions import Distribution
from gpytorch.likelihoods import Likelihood
from gpytorch.models.exact_prediction_strategies import DefaultPredictionStrategy
from gpytorch.utils.memoize import cached
from torch import Tensor

class SparsePredictionStrategy(DefaultPredictionStrategy):
    """Prediction strategy for Sparse GPs."""

    def __init__(
        self,
        train_inputs: Tensor,
        train_prior_dist: Distribution,
        train_labels: Tensor,
        likelihood: Likelihood,
        k_uu: Tensor,
        root: Optional[Tensor] = ...,
        inv_root: Optional[Tensor] = ...,
    ) -> None: ...
    @property  # type: ignore
    @cached(name="k_uu_inv_root")
    def k_uu_inv_root(self) -> Tensor: ...
    @property  # type: ignore
    @cached(name="mean_cache")
    def mean_cache(self) -> Tensor: ...
