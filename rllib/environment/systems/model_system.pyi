import numpy as np

from .abstract_system import AbstractSystem
from rllib.model import AbstractModel

class ModelSystem(AbstractSystem):
    dynamical_model: AbstractModel
    def __init__(self, dynamical_model: AbstractModel) -> None: ...
