from abc import ABC, abstractmethod


class AbstractTransform(ABC):
    """Abstract transform to apply on a dataset.
    """
    @abstractmethod
    def update(self, trajectory):
        """Update parameters of transformer.

        Parameters
        ----------
        trajectory: ndarray

        Returns
        -------
        None
        """
        raise NotImplementedError

    @abstractmethod
    def __call__(self, observation):
        """Apply the transformation

        Parameters
        ----------
        observation: Observation

        Returns
        -------
        observation: Observation
        """
        raise NotImplementedError

    @abstractmethod
    def reverse(self, observation):
        """Reverse the transformation

        Parameters
        ----------
        observation: Observation

        Returns
        -------
        observation: Observation
        """
        raise NotImplementedError
