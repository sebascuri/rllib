from abc import ABC, abstractmethod


class AbstractPolicy(ABC):
    """Interface for policies to control an environment.

    The public methods are:
        random_action
        action

    The public attributes are:
        action_dim
    """
    def __init__(self, action_space):
        """Initialize Policy

        Parameters
        ----------
        action_space: torch.distributions.Distribution

        """
        super(AbstractPolicy, self).__init__()
        self._action_space = action_space  # this could also be though of as the prior.

    def random_action(self):
        """Get a random action from the action space.

        Returns
        -------
        action: torch.distributions.Distribution

        """
        return self._action_space

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

    @property
    def _action_dim(self):
        """Action dimensions.

        Returns
        -------
        action_dim: int

        """
        if self.is_discrete:
            return self._action_space.logits.shape[0]
        else:
            return self._action_space.mean.shape[0]

    @property
    def is_discrete(self):
        """Check if policy output is discrete.

        Returns
        -------
        flag: bool

        """
        return self._action_space.has_enumerate_support
