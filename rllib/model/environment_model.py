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
            dim_state=environment.dim_state,
            dim_action=environment.dim_action,
            num_states=environment.num_states,
            num_actions=environment.num_actions,
            deterministic=True,
            *args,
            **kwargs,
        )
        self.environment = environment

    @classmethod
    def default(cls, environment, *args, **kwargs):
        """See AbstractModel.default()."""
        return cls(environment)

    def forward(self, state, action, _=None):
        """Get Next-State distribution."""
        self.environment.state = state
        next_state, reward, done, _ = self.environment.step(action)
        if self.model_kind == "dynamics":
            return next_state, torch.zeros(1)
        elif self.model_kind == "rewards":
            return reward, torch.zeros(1)
        elif self.model_kind == "termination":
            return (
                torch.zeros(*done.shape, 2)
                .scatter_(
                    dim=-1, index=(~done).long().unsqueeze(-1), value=-float("inf")
                )
                .squeeze(-1)
            )
        else:
            raise NotImplementedError(f"{self.model_kind} not implemented")

    @property
    def name(self):
        """Get Model name."""
        return f"{self.environment.name} Model"
