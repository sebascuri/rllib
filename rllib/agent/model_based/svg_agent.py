"""Model-Based SVG Agent."""

from rllib.algorithms.svg import SVG
from rllib.dataset.experience_replay import ExperienceReplay

from .bptt_agent import BPTTAgent


class SVGAgent(BPTTAgent):
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
        memory = ExperienceReplay(max_len=50000, num_steps=num_steps)
        super().__init__(
            policy=policy,
            critic=critic,
            dynamical_model=dynamical_model,
            reward_model=reward_model,
            criterion=criterion,
            termination=termination,
            num_steps=num_steps,
            num_samples=num_samples,
            memory=memory,
            algorithm=SVG,
            *args,
            **kwargs,
        )
