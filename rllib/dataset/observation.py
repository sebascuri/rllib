from collections import namedtuple

_observation = namedtuple('Observation',
                          ('state', 'action', 'reward', 'next_state'))


class Observation(_observation):
    """An observation is a tuple made of (state, action, reward, next_state).

    The public attributes are:
        state
        reward
        action
        next_state

    """
    @property
    def state_dim(self):
        """State dimension.

        Returns
        -------
        dim: int
            Dimension of action space.

        """
        return len(self.state)

    @property
    def action_dim(self):
        """Action dimension.

        Returns
        -------
        dim: int
            Dimension of action space.

        """
        return len(self.action)
