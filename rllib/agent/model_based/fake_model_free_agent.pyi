from typing import Any
from .model_based_agent import ModelBasedAgent
from rllib.agent import AbstractAgent

class FakeModelFreeAgent(ModelBasedAgent):

    model_free_agent: AbstractAgent
    fake_episodes: int
    fake_horizon: int
    def __init__(
        self,
        model_free_agent: AbstractAgent,
        fake_episodes: int = ...,
        fake_horizon: int = ...,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
