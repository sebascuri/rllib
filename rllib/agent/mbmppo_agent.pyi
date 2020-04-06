"""Model-Based MPPO Agent."""
from typing import List

import torch
from torch.optim.optimizer import Optimizer

from .abstract_agent import AbstractAgent
from rllib.algorithms.mppo import MBMPPO
from rllib.dataset.transforms import AbstractTransform
from rllib.dataset.experience_replay import ExperienceReplay
from rllib.dataset.datatypes import Observation



class MBMPPOAgent(AbstractAgent):
    mppo: MBMPPO
    mppo_optimizer: Optimizer
    model_optimizer: Optimizer
    dataset: ExperienceReplay
    sim_dataset: ExperienceReplay

    num_mppo_iter: int
    batch_size: int
    num_simulation_steps: int
    num_simulation_trajectories: int
    state_refresh_interval: int
    initial_states: torch.Tensor
    new_episode: bool
    trajectory: List[Observation]
    sim_trajectory: Observation

    def __init__(self, environment: str, mppo: MBMPPO,
                 transformations: List[AbstractTransform],
                 max_memory: int = 10000, batch_size: int = 64,
                 num_model_iter: int = 30,
                 num_mppo_iter: int = 100,
                 num_simulation_steps: int = 200,
                 num_simulation_trajectories: int =8,
                 state_refresh_interval: int = 2,
                 gamma: float = 1.0, exploration_steps: int = 0, exploration_episodes: int = 0,
                 comment: str = '') -> None: ...