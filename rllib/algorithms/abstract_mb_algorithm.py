"""ModelBasedAlgorithm."""
from rllib.model import TransformedModel
from rllib.util.neural_networks.utilities import repeat_along_dimension
from rllib.util.rollout import rollout_model
from rllib.value_function import (
    AbstractQFunction,
    AbstractValueFunction,
    IntegrateQValueFunction,
)


class AbstractMBAlgorithm(object):
    """Model Based Algorithm.

    A model based algorithm has a dynamical_model and a reward_model and, it has a
    simulate method that simulates trajectories following a policy.

    Parameters
    ----------
    dynamical_model: AbstractModel.
        Dynamical model to simulate.
    reward_model: AbstractReward.
        Reward model to simulate.
    num_steps: int.
        Number of steps to simulate.
    num_samples: int.
        Number of parallel samples to simulate.
    termination: Termination, optional.
        Termination condition to evaluate while simulating.

    Methods
    -------
    simulate(self, state: State, policy: AbstractPolicy) -> Trajectory:
        Simulate a set of particles starting from `state' and following `policy'.
    """

    def __init__(
        self,
        dynamical_model,
        reward_model,
        num_steps=1,
        num_samples=1,
        termination_model=None,
        *args,
        **kwargs,
    ):
        super().__init__()
        if not isinstance(dynamical_model, TransformedModel):
            dynamical_model = TransformedModel(dynamical_model, [])
        if not isinstance(reward_model, TransformedModel):
            reward_model = TransformedModel(reward_model, [])
        self.dynamical_model = dynamical_model
        self.reward_model = reward_model
        self.termination_model = termination_model

        assert self.dynamical_model.model_kind == "dynamics"
        assert self.reward_model.model_kind == "rewards"
        if self.termination_model is not None:
            assert self.termination_model.model_kind == "termination"

        self.num_steps = num_steps
        self.num_samples = num_samples

        if hasattr(self, "critic_target"):
            if isinstance(self.critic_target, AbstractValueFunction):
                self.value_target = self.critic_target
            elif isinstance(self.critic_target, AbstractQFunction):
                if hasattr(self, "policy"):
                    self.value_target = IntegrateQValueFunction(
                        self.critic_target, self.policy, num_samples=1
                    )
                else:
                    self.value_target = None
        else:
            self.value_target = None

    def simulate(self, state, policy):
        """Simulate a set of particles starting from `state' and following `policy'."""
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
            termination_model=self.termination_model,
        )
