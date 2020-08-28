from rllib.algorithms.steve import STEVE

from .derived_model_based_agent import DerivedMBAgent

class STEVEAgent(DerivedMBAgent):
    algorithm: STEVE
