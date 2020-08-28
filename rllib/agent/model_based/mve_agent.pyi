from rllib.algorithms.mve import MVE

from .derived_model_based_agent import DerivedMBAgent

class MVEAgent(DerivedMBAgent):
    algorithm: MVE
