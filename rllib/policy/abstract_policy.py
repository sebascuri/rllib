from abc import ABC, abstractmethod
import torch
from torch.distributions import MultivariateNormal, Categorical


class AbstractPolicy(ABC):
    """Interface for policies to control an environment.

    The public methods are:
        random
        action

    The public attributes are:
        dim_state
        dim_action
    """

    def __init__(self, dim_state, dim_action, num_action=None, scale=1.):
        """Initialize Policy

        Parameters
        ----------
        dim_state: int
        dim_action: int
        num_action: int, optional
        scale: float, optional
        """
        super().__init__()
        self.dim_state = dim_state
        self.dim_action = dim_action
        self._num_action = num_action
        self._scale = scale

    @abstractmethod
    def action(self, state):
        """Return the action distribution of the policy.

        Parameters
        ----------
        state: array_like

        Returns
        -------
        action: torch.distributions.Distribution

        """
        raise NotImplementedError

    def random(self):
        if self.discrete_action:  # Categorical
            return Categorical(torch.ones(self._num_action))
        else:  # Categorical
            return MultivariateNormal(
                loc=torch.zeros(self.dim_action),
                covariance_matrix=self._scale * torch.eye(self.dim_action))

    @property
    def discrete_action(self):
        return self._num_action is not None
