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
    temperature: float, optional
        temperature scaling of output distribution.
    layers: list, optional
        width of layers, each layer is connected with ReLUs non-linearities.
    biased_head: bool, optional
        flag that indicates if head of NN has a bias term or not.

    """

    def __init__(self, dim_state, dim_action, num_states=None, num_actions=None,
                 temperature=1.0, layers=None, biased_head=True):
        super().__init__(dim_state, dim_action, num_states, num_actions, temperature)
        if self.discrete_state:
            in_dim = self.num_states
        else:
            in_dim = self.dim_state

        if self.discrete_action:
            self._policy = CategoricalNN(in_dim, self.num_actions, layers,
                                         self.temperature, biased_head=biased_head)
        else:
            self._policy = HeteroGaussianNN(in_dim, self.dim_action, layers,
                                            self.temperature, biased_head=biased_head)

    def __call__(self, state):
        """Get distribution over actions."""
        if self.discrete_state:
            state = one_hot_encode(state, self.num_states)
        return self._policy(state)

    @property
    def parameters(self):
        """Return parameters of nn.Module that parametrize the policy.

        Returns
        -------
        generator
        """
        return self._policy.parameters()

    @parameters.setter
    def parameters(self, new_params):
        """See `AbstractPolicy.parameters'."""
        update_parameters(self._policy.parameters(), new_params, tau=1.0)


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
    temperature: float, optional
        temperature scaling of output distribution.

    Notes
    -----
    This class is only implemented for continuous state and action spaces.

    """

    def __init__(self, dim_state, dim_action, num_states=None, num_actions=None,
                 temperature=1.0):
        super().__init__(dim_state, dim_action, num_states, num_actions, temperature)

        if self.discrete_state or self.discrete_action:
            raise ValueError("Felix Policy is for Continuous Problems")

        self._policy = FelixNet(dim_state, dim_action)
