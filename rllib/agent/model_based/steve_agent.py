"""Model-Based Steve Agent."""

from rllib.algorithms.steve import STEVE

from .derived_model_based_agent import DerivedMBAgent


class STEVEAgent(DerivedMBAgent):
    """Implementation of a STEVE-Agent."""

    def __init__(self, *args, **kwargs):
        super().__init__(derived_algorithm_=STEVE, *args, **kwargs)
