from .abstract_policy import AbstractPolicy


class RandomPolicy(AbstractPolicy):
    def __init__(self, action_space, state_dim):
        super(RandomPolicy, self).__init__(action_space)
        self._state_dim = state_dim

    def action(self, state):
        return self.random_action()
