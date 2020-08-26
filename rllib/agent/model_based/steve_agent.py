"""Model-Based Steve Agent."""

from rllib.algorithms.steve import steve_expand

from .derived_model_based_agent import DerivedMBAgent


class STEVEAgent(DerivedMBAgent):
    """Implementation of a STEVE-Agent."""

    def __init__(self, *args, **kwargs):
        super().__init__(derived_algorithm_=steve_expand, *args, **kwargs)
