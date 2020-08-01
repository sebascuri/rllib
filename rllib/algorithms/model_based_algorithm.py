"""ModelBasedAlgorithm."""

from rllib.dataset.utilities import stack_list_of_tuples
from rllib.util.neural_networks.utilities import repeat_along_dimension
from rllib.util.rollout import rollout_model


class ModelBasedAlgorithm(object):
    """Model Based Algorithm.

    A model based algorithm simulates trajectories with the current model.
    """

    def __init__(
        self,
        dynamical_model,
        reward_model,
        policy,
        num_steps=1,
        num_action_samples=15,
        termination=None,
        dist_params=None,
    ):
        super().__init__()
        self.dynamical_model = dynamical_model
        self.reward_model = reward_model
        self.policy = policy
        self.num_steps = num_steps
        self.num_action_samples = num_action_samples
        self.termination = termination
        self.dist_params = {} if dist_params is None else dist_params

    def simulate(self, state):
        """Simulate trajectories starting from a state."""
        state = repeat_along_dimension(state, number=self.num_action_samples, dim=0)
        trajectory = rollout_model(
            self.dynamical_model,
            self.reward_model,
            self.policy,
            state,
            max_steps=self.num_steps,
            termination=self.termination,
            **self.dist_params,
        )
        return stack_list_of_tuples(trajectory)
