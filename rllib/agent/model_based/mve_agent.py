"""Model-Based MVE Agent."""

from rllib.algorithms.mve import MVE

from .derived_model_based_agent import DerivedMBAgent


class MVEAgent(DerivedMBAgent):
    """Implementation of a MVE-Agent."""

    def __init__(self, td_k=False, *args, **kwargs):
        super().__init__(derived_algorithm_=MVE, td_k=td_k, *args, **kwargs)
