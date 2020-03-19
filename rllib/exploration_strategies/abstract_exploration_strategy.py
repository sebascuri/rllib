"""Interface for exploration strategies."""


from abc import ABCMeta, abstractmethod

from rllib.util.parameter_decay import Constant


class AbstractExplorationStrategy(object, metaclass=ABCMeta):
    """Interface for policies to control an environment.

    Parameters
    ----------
    param: ParamDecay
        initial value of exploration parameter.
    dimension: int, optional
        rate of decay of exploration parameter.

    Attributes
    ----------
    __call__(action_distribution, steps): ndarray or int
    return the action that the action_distribution

    """

    def __init__(self, param, dimension=1):
        if type(param) is float:
            param = Constant(param)
        self.param = param
        self.dimension = dimension

    @abstractmethod
    def __call__(self, state=None):
        """Get noise from exploration strategy."""
        raise NotImplementedError
