"""Policies parametrized with Neural Networks."""


from .abstract_policy import AbstractPolicy
from rllib.util.neural_networks import HeteroGaussianNN, CategoricalNN, FelixNet
from rllib.util.neural_networks import update_parameters, one_hot_encode


__all__ = ['NNPolicy', 'FelixPolicy']


class NNPolicy(AbstractPolicy):
    """Implementation of a Policy implemented with a Neural Network.

    Parameters
    ----------
    dim_state: int
        dimension of state.
    dim_action: int
        dimension of action.
    num_states: int, optional
        number of discrete states (None if state is continuous).
    num_actions: int, optional
        number of discrete actions (None if action is continuous).
    layers: list, optional
        width of layers, each layer is connected with ReLUs non-linearities.
    biased_head: bool, optional
        flag that indicates if head of NN has a bias term or not.

    """

    def __init__(self, dim_state, dim_action, num_states=None, num_actions=None,
                 layers=None, tau=1.0, biased_head=True):
        super().__init__(dim_state, dim_action, num_states, num_actions)
        if self.discrete_state:
            in_dim = self.num_states
        else:
            in_dim = self.dim_state

        if self.discrete_action:
            self.policy = CategoricalNN(in_dim, self.num_actions, layers,
                                        biased_head=biased_head)
        else:
            self.policy = HeteroGaussianNN(in_dim, self.dim_action, layers,
                                           biased_head=biased_head)
        self._tau = tau

    def __call__(self, state):
        """Get distribution over actions."""
        if self.discrete_state:
            state = one_hot_encode(state, self.num_states)
        return self.policy(state)

    @property
    def parameters(self):
        """Return parameters of nn.Module that parametrize the policy.

        Returns
        -------
        generator
        """
        return self.policy.parameters()

    @parameters.setter
    def parameters(self, new_params):
        """See `AbstractPolicy.parameters'."""
        update_parameters(self.policy.parameters(), new_params, tau=self._tau)


class FelixPolicy(NNPolicy):
    """Implementation of a NN Policy using FelixNet (designed by Felix Berkenkamp).

    Parameters
    ----------
    dim_state: int
        dimension of state.
    dim_action: int
        dimension of action.
    num_states: int, optional
        number of discrete states (None if state is continuous).
    num_actions: int, optional
        number of discrete actions (None if action is continuous).

    Notes
    -----
    This class is only implemented for continuous state and action spaces.

    """

    def __init__(self, dim_state, dim_action, num_states=None, num_actions=None):
        super().__init__(dim_state, dim_action, num_states, num_actions)

        if self.discrete_state or self.discrete_action:
            raise ValueError("Felix Policy is for Continuous Problems")

        self.policy = FelixNet(dim_state, dim_action)
