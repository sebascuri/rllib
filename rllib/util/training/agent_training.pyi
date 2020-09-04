from typing import Callable, List, Optional

import torch.nn as nn

from rllib.agent import AbstractAgent
from rllib.environment import AbstractEnvironment

def train_agent(
    agent: AbstractAgent,
    environment: AbstractEnvironment,
    num_episodes: int,
    max_steps: int,
    plot_flag: bool = ...,
    print_frequency: int = ...,
    plot_frequency: int = ...,
    save_milestones: Optional[List[int]] = ...,
    render: bool = ...,
    plot_callbacks: Optional[List[Callable[[AbstractAgent, int], None]]] = ...,
) -> None: ...
def evaluate_agent(
    agent: AbstractAgent,
    environment: AbstractEnvironment,
    num_episodes: int,
    max_steps: int,
    render: bool = ...,
) -> None: ...
