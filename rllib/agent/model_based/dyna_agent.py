"""Model-Based DYNA Agent."""

from rllib.algorithms.dyna import Dyna

from .derived_model_based_agent import DerivedMBAgent


class DynaAgent(DerivedMBAgent):
    """Implementation of a Dyna-Agent."""

    def __init__(self, *args, **kwargs):
        super().__init__(derived_algorithm_=Dyna, *args, **kwargs)
