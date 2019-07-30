from .abstract_policy import AbstractPolicy
from rllib.util.neural_networks import HeteroGaussianNN, CategoricalNN
from rllib.util.neural_networks import update_parameters


class NNPolicy(AbstractPolicy):
    def __init__(self, dim_state, dim_action, num_states=None, num_actions=None,
                 temperature=1.0, layers=None):
        super().__init__(dim_state, dim_action, num_states, num_actions, temperature)
        if self.discrete_states:
            in_dim = self.num_states
        else:
            in_dim = self.dim_state

        if self.discrete_action:
            self._policy = CategoricalNN(in_dim, self.num_actions, layers,
                                         self.temperature)
        else:
            self._policy = HeteroGaussianNN(in_dim, self.dim_action, layers,
                                            self.temperature)

    def __call__(self, state):
        return self._policy(state)

    @property
    def parameters(self):
        return self._policy.parameters()

    @parameters.setter
    def parameters(self, new_params):
        update_parameters(self._policy.parameters(), new_params, tau=1.0)
