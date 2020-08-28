from rllib.algorithms.dyna import Dyna

from .derived_model_based_agent import DerivedMBAgent

class DynaAgent(DerivedMBAgent):
    algorithm: Dyna
