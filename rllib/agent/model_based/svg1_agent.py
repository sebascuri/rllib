"""Model-Based SVG Agent."""

from rllib.algorithms.svg1 import SVG1

from .bptt_agent import BPTTAgent


class SVG1Agent(BPTTAgent):
    """Implementation of a SVG-Agent."""

    @classmethod
    def default(cls, environment, *args, **kwargs):
        """See `AbstractAgent.default'."""
        return super().default(environment, bptt_algorithm_=SVG1)
