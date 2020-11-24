from rllib.dataset.datatypes import Action, State

from .abstract_system import AbstractSystem

class GaussianNoiseSystem(AbstractSystem):
    _system: AbstractSystem
    _transition_noise_scale: float
    _measurement_noise_scale: float
    def __init__(
        self,
        system: AbstractSystem,
        transition_noise_scale: float,
        measurement_noise_scale: float = ...,
    ) -> None: ...
    def step(self, action: Action) -> State: ...
    def reset(self, state: State) -> State: ...
    @property
    def state(self) -> State: ...
    @state.setter
    def state(self, value: State) -> None: ...
