"""Model-Based SVG Agent."""

from rllib.algorithms.svg1 import SVG1

from .bptt_agent import BPTTAgent


class SVG1Agent(BPTTAgent):
    """Implementation of a SVG-Agent."""

    def __init__(
        self,
        policy,
        critic,
        dynamical_model,
        reward_model,
        criterion,
        termination=None,
        num_steps=1,
        num_samples=15,
        *args,
        **kwargs,
    ):

        super().__init__(
            policy=policy,
            critic=critic,
            dynamical_model=dynamical_model,
            reward_model=reward_model,
            criterion=criterion,
            termination=termination,
            num_steps=num_steps,
            num_samples=num_samples,
            algorithm=SVG1,
            *args,
            **kwargs,
        )
