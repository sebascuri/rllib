"""Interface for Transformers of a dataset."""


from abc import ABCMeta, abstractmethod


class AbstractTransform(object, metaclass=ABCMeta):
    """Abstract transform to apply on a dataset.

    Methods
    -------
    __call__(observation): Observation
        transform a raw observation.
    inverse(observation): Observation
        revert the transformation of the observation.
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
            transformed observation.

        """
        raise NotImplementedError

    @abstractmethod
    def inverse(self, observation):
        """Apply the inverse transformation.

        Parameters
        ----------
        observation: Observation
            transformed observation.

        Returns
        -------
        observation: Observation
            un-transformed observation.

        """
        raise NotImplementedError

    def update(self, observation):
        """Update parameters of transformer.

        Parameters
        ----------
        observation: Observation

        """
        pass
