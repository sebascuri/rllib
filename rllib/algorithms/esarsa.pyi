from rllib.value_function import IntegrateQValueFunction

from .sarsa import SARSA

class ESARSA(SARSA):
    value_function: IntegrateQValueFunction

class GradientExpectedSARSA(ESARSA): ...
