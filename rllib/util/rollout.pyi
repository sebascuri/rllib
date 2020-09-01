from typing import Any, Callable, List, Optional, Tuple, Union

from numpy import ndarray
from torch import Tensor

from rllib.agent import AbstractAgent
from rllib.dataset.datatypes import Action, Distribution, Observation, State, Trajectory
from rllib.environment import AbstractEnvironment
from rllib.model import AbstractModel
from rllib.policy import AbstractPolicy

def step_env(
    environment: AbstractEnvironment,
    state: Union[int, ndarray],
    action: Union[int, ndarray],
    action_scale: Action = ...,
    pi: Optional[Distribution] = ...,
    render: bool = ...,
) -> Tuple[Observation, Union[int, ndarray], bool, dict]: ...
def step_model(
    dynamical_model: AbstractModel,
    reward_model: AbstractModel,
    termination_model: AbstractModel,
    state: Tensor,
    action: Tensor,
    done: Tensor,
    action_scale: Action = ...,
    pi: Optional[Distribution] = ...,
) -> Tuple[Observation, Tensor, Tensor]: ...
def record(
    environment: AbstractEnvironment,
    agent: AbstractAgent,
    save_dir: str,
    num_episodes: int = ...,
    max_steps: int = ...,
) -> None: ...
def rollout_agent(
    environment: AbstractEnvironment,
    agent: AbstractAgent,
    num_episodes: int = ...,
    max_steps: int = ...,
    render: bool = ...,
    print_frequency: int = ...,
    plot_frequency: int = ...,
    save_milestones: Optional[List[int]] = ...,
    plot_callbacks: Optional[List[Callable[[AbstractAgent, int], None]]] = ...,
) -> None: ...
def rollout_policy(
    environment: AbstractEnvironment,
    policy: AbstractPolicy,
    num_episodes: int = ...,
    max_steps: int = ...,
    render: bool = ...,
    **kwargs: Any,
) -> List[Trajectory]: ...
def rollout_model(
    dynamical_model: AbstractModel,
    reward_model: AbstractModel,
    policy: AbstractPolicy,
    initial_state: State,
    termination_model: Optional[AbstractModel] = ...,
    max_steps: int = ...,
    **kwargs: Any,
) -> Trajectory: ...
def rollout_actions(
    dynamical_model: AbstractModel,
    reward_model: AbstractModel,
    action_sequence: Action,
    initial_state: State,
    termination_model: Optional[AbstractModel] = ...,
) -> Trajectory: ...
