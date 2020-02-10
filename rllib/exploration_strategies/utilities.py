"""Utilities for exploration strategies."""

from torch.distributions import Categorical, MultivariateNormal


def argmax(action_distribution):
    """Return the arguments that maximizes a distribution.

    Parameters
    ----------
    action_distribution: torch.distributions.Distribution

    Returns
    -------
    ndarray or int

    """
    if type(action_distribution) is Categorical:
        return action_distribution.logits.argmax().numpy()
    elif type(action_distribution) is MultivariateNormal:
        return action_distribution.loc.detach().numpy()
    else:
        raise NotImplementedError("""
        Action Distribution should be of type Categorical or MultivariateNormal but {}
        type was passed.
        """.format(type(action_distribution)))
