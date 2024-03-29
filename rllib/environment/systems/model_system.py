"""Implementation of a System from a Learned Model."""

from rllib.util.neural_networks.utilities import to_torch
from rllib.util.utilities import tensor_to_distribution

from .abstract_system import AbstractSystem


class ModelSystem(AbstractSystem):
    """An system based on a dynamical model.

    Parameters
    ----------
    dynamical_model: AbstractModel.
        dynamical model.
    """

    def __init__(self, dynamical_model):
        self.dynamical_model = dynamical_model
        super().__init__(
            dim_state=dynamical_model.dim_state, dim_action=dynamical_model.dim_action
        )

    def step(self, action):
        """See `AbstractSystem.step'."""
        state, action = to_torch(self.state), to_torch(action)
        self.state = (
            tensor_to_distribution(self.dynamical_model(state, action)).sample().numpy()
        )
        return self.state

    def reset(self, state):
        """See `AbstractSystem.reset'."""
        self.state = state
        return self.state

    @property
    def state(self):
        """See `AbstractSystem.state'."""
        return self._state

    @state.setter
    def state(self, value):
        self._state = value
