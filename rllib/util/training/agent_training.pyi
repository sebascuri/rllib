from typing import Any

import torch.nn as nn

from rllib.agent import AbstractAgent
from rllib.environment import AbstractEnvironment

def train_agent(
    agent: AbstractAgent,
    environment: AbstractEnvironment,
    plot_flag: bool = ...,
    *args: Any,
    **kwargs: Any,
) -> None: ...
def evaluate_agent(
    agent: AbstractAgent,
    environment: AbstractEnvironment,
    num_episodes: int,
    max_steps: int,
    render: bool = ...,
) -> None: ...
