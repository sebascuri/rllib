from .abstract_policy import AbstractPolicy


class RandomPolicy(AbstractPolicy):
    def __init__(self, dim_state, dim_action, num_action=None, scale=1.0):
        super().__init__(dim_state, dim_action, num_action, scale)

    def action(self, state):
        return self.random()
