from typing import Type, Union

from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer

from rllib.algorithms.dpg import DPG
from rllib.dataset import ExperienceReplay
from rllib.policy import AbstractPolicy
from rllib.util.parameter_decay import ParameterDecay
from rllib.value_function import AbstractQFunction

from .off_policy_agent import OffPolicyAgent

class DPGAgent(OffPolicyAgent):
    algorithm: DPG
    def __init__(
        self,
        q_function: AbstractQFunction,
        policy: AbstractPolicy,
        criterion: Type[_Loss],
        optimizer: Optimizer,
        memory: ExperienceReplay,
        exploration_noise: Union[float, ParameterDecay],
        num_iter: int = ...,
        batch_size: int = ...,
        target_update_frequency: int = ...,
        policy_noise: float = ...,
        noise_clip: float = ...,
        train_frequency: int = ...,
        num_rollouts: int = ...,
        gamma: float = ...,
        exploration_steps: int = ...,
        exploration_episodes: int = ...,
        tensorboard: bool = ...,
        comment: str = ...,
    ) -> None: ...
