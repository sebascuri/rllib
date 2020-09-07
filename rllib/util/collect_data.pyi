"""Utilities to collect data."""
from typing import List, Union

from torch.distributions import Distribution

from rllib.dataset.datatypes import Observation
from rllib.environment import AbstractEnvironment
from rllib.model import AbstractModel
from rllib.policy import AbstractPolicy

def collect_environment_transitions(
    state_dist: Distribution,
    policy: Union[Distribution, AbstractPolicy],
    environment: AbstractEnvironment,
    num_samples: int,
) -> List[Observation]: ...
def collect_model_transitions(
    state_dist: Distribution,
    policy: Union[Distribution, AbstractPolicy],
    dynamical_model: AbstractModel,
    reward_model: AbstractModel,
    num_samples: int,
) -> List[Observation]: ...
