"""Model-Based SVG Agent."""

from rllib.algorithms.svg import SVG

from .derived_model_based_agent import DerivedMBAgent


class SVGAgent(DerivedMBAgent):
    """Implementation of a SVG-Agent."""

    def __init__(self, *args, **kwargs):
        super().__init__(derived_algorithm_=SVG, *args, **kwargs)
