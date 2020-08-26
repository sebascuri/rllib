"""Model-Based MVE Agent."""

from rllib.algorithms.mve import mve_expand

from .derived_model_based_agent import DerivedMBAgent


class MVEAgent(DerivedMBAgent):
    """Implementation of a MVE-Agent."""

    def __init__(self, td_k=True, *args, **kwargs):
        super().__init__(derived_algorithm_=mve_expand, td_k=td_k, *args, **kwargs)
