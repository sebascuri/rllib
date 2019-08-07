"""Interface for Transformers of a dataset."""


from abc import ABC, abstractmethod


class AbstractTransform(ABC):
    """Abstract transform to apply on a dataset.

    Methods
    -------
    __call__(observation): Observation
        normalize a raw observation.
    inverse(observation): Observation
        revert the normalization of the observation.
    update(observation):
        update the parameters of the transformer.

    """

    @abstractmethod
    def __call__(self, observation):
        """Apply the transformation.

        Parameters
        ----------
        observation: Observation
            raw observation.

        Returns
        -------
        observation: Observation
            normalized observation.

        """
        raise NotImplementedError

    @abstractmethod
    def inverse(self, observation):
        """Apply the inverse transformation.

        Parameters
        ----------
        observation: Observation
            normalized observation.

        Returns
        -------
        observation: Observation
            un-normalized observation.

        """
        raise NotImplementedError

    @abstractmethod
    def update(self, observation):
        """Update parameters of transformer.

        Parameters
        ----------
        observation: Observation

        """
        raise NotImplementedError
