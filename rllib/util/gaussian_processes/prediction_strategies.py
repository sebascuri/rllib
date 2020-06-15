"""Implementation of cached prediction strategies for Sparse GPs."""
import functools

from gpytorch import settings
from gpytorch.lazy import delazify
from gpytorch.models.exact_prediction_strategies import (
    DefaultPredictionStrategy,
    clear_cache_hook,
)
from gpytorch.utils.memoize import cached


class SparsePredictionStrategy(DefaultPredictionStrategy):
    """Prediction strategy for Sparse GPs."""

    def __init__(
        self,
        train_inputs,
        train_prior_dist,
        train_labels,
        likelihood,
        k_uu,
        root=None,
        inv_root=None,
    ):
        super().__init__(
            train_inputs,
            train_prior_dist,
            train_labels,
            likelihood,
            root=root,
            inv_root=inv_root,
        )
        self.k_uu = k_uu
        self.lik_train_train_covar = train_prior_dist.lazy_covariance_matrix

    @property  # type: ignore
    @cached(name="k_uu_inv_root")
    def k_uu_inv_root(self):
        """Get K_uu^-1/2."""
        train_train_covar = self.k_uu
        train_train_covar_inv_root = delazify(
            train_train_covar.root_inv_decomposition().root
        )
        return train_train_covar_inv_root

    @property  # type: ignore
    @cached(name="mean_cache")
    def mean_cache(self):
        r"""Get mean cache, namely \sigma^-1 k_uf y_f."""
        sigma = self.lik_train_train_covar
        sigma_inv_root = delazify(sigma.root_inv_decomposition().root)
        sigma_inv = sigma_inv_root @ sigma_inv_root.transpose(-2, -1)
        mean_cache = (sigma_inv @ self.train_labels.unsqueeze(-1)).squeeze(-1)

        if settings.detach_test_caches.on():
            mean_cache = mean_cache.detach()

        if mean_cache.grad_fn is not None:
            wrapper = functools.partial(clear_cache_hook, self)
            functools.update_wrapper(wrapper, clear_cache_hook)
            mean_cache.grad_fn.register_hook(wrapper)

        return mean_cache
