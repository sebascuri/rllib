"""Model implemented by querying an environment."""
import torch

from rllib.util.neural_networks.utilities import to_torch

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
            return to_torch(next_state), torch.zeros(1)
        elif self.model_kind == "rewards":
            return to_torch(reward), torch.zeros(1)
        elif self.model_kind == "termination":
            done_prob = torch.zeros(*done.shape, 2)
            inf = -float("inf") * torch.ones(*done.shape, 2)
            idx = (~done).long().unsqueeze(-1)
            return done_prob.scatter_(dim=-1, src=inf, index=idx)
        else:
            raise NotImplementedError(f"{self.model_kind} not implemented")

    @property
    def name(self):
        """Get Model name."""
        return f"{self.environment.name} Model"
