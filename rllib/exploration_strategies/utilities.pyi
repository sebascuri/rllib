from torch.distributions import Categorical, MultivariateNormal
from typing import Union
import numpy as np

Distribution = Union[Categorical, MultivariateNormal]
Action = Union[np.ndarray, int]

def argmax(action_distribution: Distribution) -> np.ndarray: ...
