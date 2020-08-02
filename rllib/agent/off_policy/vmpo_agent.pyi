from typing import Optional, Type, Union

from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer

from rllib.agent.off_policy.off_policy_agent import OffPolicyAgent
from rllib.algorithms.vmpo import VMPO
from rllib.dataset import ExperienceReplay
from rllib.policy import AbstractPolicy
from rllib.util.parameter_decay import ParameterDecay
from rllib.value_function import AbstractValueFunction

class VMPOAgent(OffPolicyAgent):

    algorithm: VMPO
    optimizer: Optimizer
    target_update_frequency: int
    num_iter: int
    def __init__(
        self,
        policy: AbstractPolicy,
        value_function: AbstractValueFunction,
        optimizer: Optimizer,
        memory: ExperienceReplay,
        criterion: Type[_Loss],
        num_action_samples: int = ...,
        epsilon: Union[ParameterDecay, float] = ...,
        epsilon_mean: Union[ParameterDecay, float] = ...,
        epsilon_var: Optional[Union[ParameterDecay, float]] = ...,
        regularization: bool = ...,
        top_k_fraction: float = ...,
        num_iter: int = ...,
        batch_size: int = ...,
        target_update_frequency: int = ...,
        train_frequency: int = ...,
        num_rollouts: int = ...,
        policy_update_frequency: int = ...,
        gamma: float = ...,
        exploration_steps: int = ...,
        exploration_episodes: int = ...,
        tensorboard: bool = ...,
        comment: str = ...,
    ) -> None: ...
