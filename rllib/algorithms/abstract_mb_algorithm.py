"""ModelBasedAlgorithm."""
from rllib.model import TransformedModel
from rllib.util.neural_networks.utilities import repeat_along_dimension
from rllib.util.rollout import rollout_model


class AbstractMBAlgorithm(object):
    """Model Based Algorithm.

    A model based algorithm simulates trajectories with a model.
    """

    def __init__(
        self,
        dynamical_model,
        reward_model,
        num_steps=1,
        num_samples=1,
        termination=None,
        *args,
        **kwargs,
    ):
        super().__init__()
        if not isinstance(dynamical_model, TransformedModel):
            dynamical_model = TransformedModel(dynamical_model, [])
        self.dynamical_model = dynamical_model
        self.reward_model = reward_model
        self.num_steps = num_steps
        self.num_samples = num_samples
        self.termination = termination

    def simulate(self, state, policy):
        """Simulate trajectories starting from a state."""
        if self.num_samples > 0:
            state = repeat_along_dimension(state, number=self.num_samples, dim=0)
        state = state.reshape(-1, *self.dynamical_model.dim_state)
        return rollout_model(
            dynamical_model=self.dynamical_model,
            reward_model=self.reward_model,
            policy=policy,
            initial_state=state,
            action_scale=policy.action_scale,
            max_steps=self.num_steps,
            termination=self.termination,
        )
