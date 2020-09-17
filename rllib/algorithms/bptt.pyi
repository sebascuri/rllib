from rllib.value_function import AbstractValueFunction

from .abstract_algorithm import AbstractAlgorithm

class BPTT(AbstractAlgorithm):
    critic: AbstractValueFunction
    critic_target: AbstractValueFunction
