"""Interface for exploration strategies."""


from abc import ABC, abstractmethod


class AbstractExplorationStrategy(ABC):
    """Interface for policies to control an environment.

    Attributes
    ----------
    __call__(action_distribution, steps): ndarray or int
    return the action that the action_distribution

    """

    @abstractmethod
    def __call__(self, action_distribution, steps=None):
        """Get action from the exploration strategy.

        Parameters
        ----------
        action_distribution: torch.distributions.Distribution
            Action distribution suggested by the policy.
        steps: int, optional
            Number of steps performed so far.

        Returns
        -------
        action: array_like
            action sample that the exploration strategy selects.

        """
        raise NotImplementedError
