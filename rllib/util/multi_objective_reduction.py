"""Multi-objective Reduction classes."""
import torch


class AbstractMultiObjectiveReduction(object):
    """Abstract class for multi-objective reduction."""

    def __init__(self, dim=2):
        self.dim = dim

    def __call__(self, value):
        """Reduce the value."""
        raise NotImplementedError


class GetIndexMultiObjectiveReduction(AbstractMultiObjectiveReduction):
    """Get index when reducing.

    Parameters
    ----------
    idx: int, optional. (default=0).
        Index to select from the value.
    """

    def __init__(self, idx=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(idx, torch.Tensor):
            idx = torch.tensor([idx]).long()
        self.idx = idx

    def __call__(self, value):
        """Reduce the value."""
        return torch.index_select(input=value, dim=self.dim, index=self.idx).squeeze(
            dim=self.dim
        )


class MeanMultiObjectiveReduction(AbstractMultiObjectiveReduction):
    """Take mean when reducing."""

    def __call__(self, value):
        """Reduce the value."""
        return value.mean(dim=self.dim)


class SumMultiObjectiveReduction(AbstractMultiObjectiveReduction):
    """Take a sum when reducing."""

    def __call__(self, value):
        """Reduce the value."""
        return value.sum(dim=self.dim)


class MaxMultiObjectiveReduction(AbstractMultiObjectiveReduction):
    """Take maximum when reducing."""

    def __call__(self, value):
        """Reduce the value."""
        return value.max(dim=self.dim)


class MinMultiObjectiveReduction(AbstractMultiObjectiveReduction):
    """Take minimum when reducing."""

    def __call__(self, value):
        """Reduce the value."""
        return value.min(dim=self.dim)


class WeightedMultiObjectiveReduction(AbstractMultiObjectiveReduction):
    """Take minimum when reducing."""

    def __init__(self, weight, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight = weight

    def __call__(self, value):
        """Reduce the value."""
        if self.dim == -1 or self.dim == value.ndim:
            return value @ self.weight
        else:
            out = self.weight[0] * torch.index_select(
                input=value, index=torch.tensor([0]), dim=self.dim
            )
            for i in range(1, len(self.weight)):
                out += self.weight[i] * torch.index_select(
                    input=value, index=torch.tensor([i]), dim=self.dim
                )

            return out


class NoMultiObjectiveReduction(AbstractMultiObjectiveReduction):
    """Do not reduce at all."""

    def __call__(self, value):
        """Reduce the value."""
        return value
