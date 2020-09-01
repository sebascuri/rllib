"""Model implemented by querying an environment."""
import torch

from .abstract_model import AbstractModel


class EnvironmentModel(AbstractModel):
    """Implementation of a Dynamical Model implemented by querying an environment.

    Parameters
    ----------
    environment: AbstractEnvironment

    """

    def __init__(self, environment, *args, **kwargs):
        super().__init__(
            environment.dim_state,
            environment.dim_action,
            num_states=environment.num_states,
            num_actions=environment.num_actions,
        )
        self.environment = environment

    @classmethod
    def default(cls, environment, *args, **kwargs):
        """See AbstractModel.default()."""
        return cls(environment)

    def forward(self, state, action, next_state=None):
        """Get Next-State distribution."""
        self.environment.state = state
        next_state, reward, done, _ = self.environment.step(action)
        return next_state, torch.zeros(1)

    @property
    def name(self):
        """Get Model name."""
        return f"{self.environment.name} Model"
