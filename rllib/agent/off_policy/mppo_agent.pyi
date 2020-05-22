"""MPPO Agent Implementation."""
from typing import Union

from torch.optim.optimizer import Optimizer
from torch.nn.modules.loss import _Loss

from rllib.agent.off_policy_agent import OffPolicyAgent
from rllib.algorithms.mppo import MPPO
from rllib.dataset import ExperienceReplay
from rllib.policy import AbstractQFunctionPolicy
from rllib.value_function import AbstractQFunction
from rllib.util.parameter_decay import ParameterDecay


class MPPOAgent(OffPolicyAgent):
    """Implementation of an agent that runs MPPO."""
    algorithm: MPPO
    optimizer: Optimizer
    target_update_frequency: int
    num_iter: int

    def __init__(self, env_name: str,
                 policy: AbstractQFunction, q_function: AbstractQFunctionPolicy,
                 optimizer: Optimizer,
                 memory: ExperienceReplay,
                 criterion: _Loss,
                 num_action_samples: int = 15,
                 entropy_reg: float = 0.,
                 epsilon: Union[ParameterDecay, float] = None,
                 epsilon_mean: Union[ParameterDecay, float] = None,
                 epsilon_var: Union[ParameterDecay, float] = None,
                 eta: Union[ParameterDecay, float] = None,
                 eta_mean: Union[ParameterDecay, float] = None,
                 eta_var: Union[ParameterDecay, float] = None,
                 num_iter: int = 100, batch_size: int = 64,
                 target_update_frequency: int = 4,
                 train_frequency: int = 0, num_rollouts: int = 1, gamma: float = 1.0,
                 exploration_steps: int = 0, exploration_episodes: int = 0,
                 comment: str = '') -> None: ...
