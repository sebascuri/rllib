from abc import ABC, abstractmethod


class AbstractValueFunction(ABC):
    def __init__(self, dim_state, num_states=None):
        self.dim_state = dim_state
        self.num_states = num_states

    @property
    def discrete_state(self):
        return self.num_states is not None

    @abstractmethod
    def __str__(self):
        raise NotImplementedError

    @abstractmethod
    def __call__(self, state, action=None):
        raise NotImplementedError

    @property
    @abstractmethod
    def parameters(self):
        raise NotImplementedError

    @parameters.setter
    @abstractmethod
    def parameters(self, new_params):
        raise NotImplementedError


class AbstractQFunction(AbstractValueFunction):
    def __init__(self, dim_state, dim_action, num_states=None, num_actions=None):
        super().__init__(dim_state=dim_state, num_states=num_states)
        self.dim_action = dim_action
        self.num_actions = num_actions

    @abstractmethod
    def __call__(self, state, action=None):
        raise NotImplementedError

    @property
    def discrete_action(self):
        return self.num_actions is not None

    def max(self, state):
        pass

    def argmax(self, state):
        pass

    def extract_policy(self, temperature=1.):
        pass
