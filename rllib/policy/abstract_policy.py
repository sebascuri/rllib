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

    def __init__(self, dim_state, dim_action, num_states=None, num_actions=None,
                 temperature=1.):
        """Initialize Policy

        Parameters
        ----------
        dim_state: int
        dim_action: int
        num_states: int, optional
        num_actions: int, optional
        temperature: float, optional
        """
        super().__init__()
        self.dim_state = dim_state
        self.dim_action = dim_action
        self.num_states = num_states
        self.num_actions = num_actions
        self.temperature = temperature

    @abstractmethod
    def __call__(self, state):
        """Return the action distribution of the policy.

        Parameters
        ----------
        state: array_like

        Returns
        -------
        action: torch.distributions.Distribution

        """
        raise NotImplementedError

    @property
    @abstractmethod
    def parameters(self):
        raise NotImplementedError

    @parameters.setter
    @abstractmethod
    def parameters(self, new_params):
        raise NotImplementedError

    def random(self):
        if self.discrete_action:  # Categorical
            return Categorical(torch.ones(self.num_actions))
        else:  # Categorical
            return MultivariateNormal(
                loc=torch.zeros(self.dim_action),
                covariance_matrix=self.temperature * torch.eye(self.dim_action)
            )

    @property
    def discrete_action(self):
        return self.num_actions is not None

    @property
    def discrete_states(self):
        return self.num_states is not None
