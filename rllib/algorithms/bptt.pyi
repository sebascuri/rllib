"""Back-Propagation Through Time Algorithm."""
from rllib.value_function import AbstractValueFunction

from .abstract_algorithm import AbstractAlgorithm
from .abstract_mb_algorithm import AbstractMBAlgorithm

class BPTT(AbstractAlgorithm, AbstractMBAlgorithm): ...
