from typing import Any, Optional

from torch import Tensor

from rllib.agent.abstract_agent import AbstractAgent
from rllib.dataset.datatypes import Observation, TupleDistribution
from rllib.model import AbstractModel

def get_target(model: AbstractModel, observation: Observation) -> Tensor: ...
def get_prediction(
    model: AbstractModel,
    observation: Observation,
    dynamical_model: Optional[AbstractModel] = ...,
) -> TupleDistribution: ...
def gaussian_cdf(x: Tensor, mean: Tensor, chol_std: Tensor) -> Tensor: ...
def calibration_count(
    target: Tensor, mean: Tensor, chol_std: Tensor, buckets: Tensor
) -> Tensor: ...
def calibration_score(
    model: AbstractModel,
    observation: Observation,
    bins: int = ...,
    dynamical_model: Optional[AbstractModel] = ...,
) -> Tensor: ...
def sharpness(
    model: AbstractModel,
    observation: Observation,
    dynamical_model: Optional[AbstractModel] = ...,
) -> Tensor: ...
def model_mse(
    model: AbstractModel,
    observation: Observation,
    dynamical_model: Optional[AbstractModel] = ...,
) -> Tensor: ...
def model_loss(
    model: AbstractModel,
    observation: Observation,
    dynamical_model: Optional[AbstractModel] = ...,
) -> Tensor: ...
def rollout_predictions(
    dynamical_model: AbstractModel,
    model: AbstractModel,
    initial_state: Tensor,
    action_sequence: Tensor,
) -> TupleDistribution: ...

class Evaluate(object):
    agent: AbstractAgent
    def __init__(self, agent: AbstractAgent) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(self, *args: Any) -> None: ...
