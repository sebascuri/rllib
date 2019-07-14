from .abstract_policy import AbstractPolicy
from rllib.util.neural_networks import GaussianNN, CategoricalNN


class NNPolicy(AbstractPolicy):
    def __init__(self, action_space, state_dim):
        super(NNPolicy, self).__init__(action_space)
        self._state_dim = state_dim

        if self.is_discrete:
            self._policy = CategoricalNN(self._state_dim, self._action_dim)
        else:
            self._policy = GaussianNN(self._state_dim, self._action_dim)

    def action(self, state):
        return self._policy(state)
