"""ModelBasedAlgorithm."""

from rllib.value_function import AbstractValueFunction

from .abstract_algorithm import AbstractAlgorithm
from .abstract_mb_algorithm import AbstractMBAlgorithm

class SVG1(AbstractAlgorithm, AbstractMBAlgorithm):
    critic: AbstractValueFunction
    critic_target: AbstractValueFunction
