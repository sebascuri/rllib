from .abstract_policy import AbstractPolicy
from rllib.util.neural_networks import GaussianNN, CategoricalNN


class NNPolicy(AbstractPolicy):
    def __init__(self, action_space, state_dim):
        super().__init__(action_space)
        self._state_dim = state_dim

        if self.discrete_action:
            self._policy = CategoricalNN(self.dim_state, self.dim_action)
        else:
            self._policy = GaussianNN(self.dim_state, self.dim_action)

    def action(self, state):
        return self._policy(state)
