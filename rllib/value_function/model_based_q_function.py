"""Value function that is computed by integrating a q-function."""

import torch

from rllib.algorithms.abstract_mb_algorithm import AbstractMBAlgorithm
from rllib.util.neural_networks.utilities import DisableGradient, unfreeze_parameters
from rllib.util.utilities import RewardTransformer
from rllib.util.value_estimation import mc_return

from .abstract_value_function import AbstractQFunction
from .integrate_q_value_function import IntegrateQValueFunction


class ModelBasedQFunction(AbstractQFunction):
    """Q function that arises from simulating the model.

    Parameters
    ----------
    policy: AbstractPolicy.
        Policy with which to rollout the model.
    value_function: AbstractValueFunction, optional.
        Value function with which to bootstrap the value estimate.
    gamma: float, optional (default=1.0).
        Discount factor.
    entropy_regularization: float, optional (default=0.0).
        Entropy regularization for rewards.
    reward_transformer: RewardTransformer, optional.
        Reward transformer module.
    """

    def __init__(
        self,
        dynamical_model,
        reward_model,
        num_steps=1,
        num_samples=15,
        termination_model=None,
        policy=None,
        value_function=None,
        gamma=1.0,
        lambda_=1.0,
        reward_transformer=RewardTransformer(),
        entropy_regularization=0.0,
        *args,
        **kwargs,
    ):
        super().__init__(
            dim_state=value_function.dim_state,
            dim_action=dynamical_model.dim_action,
            *args,
            **kwargs,
        )
        self.sim = AbstractMBAlgorithm(
            dynamical_model=dynamical_model,
            reward_model=reward_model,
            num_steps=num_steps,
            num_samples=num_samples,
            termination_model=termination_model,
        )
        assert num_steps > 0, "At least one-step ahead simulation."
        if policy is None:
            assert num_steps == 1, "If no policy is passed, then only one-step ahead."
        self.value_function = value_function
        self.lambda_ = lambda_
        self.policy = policy
        self.gamma = gamma
        self.reward_transformer = reward_transformer
        self.entropy_regularization = entropy_regularization

    def set_policy(self, new_policy):
        """Set policy."""
        self.policy = new_policy
        try:
            self.value_function.set_policy(new_policy)
        except AttributeError:
            pass

    def forward(self, state, action=torch.tensor(float("nan"))):
        """Get value at a given state-(action) through simulation.

        Parameters
        ----------
        state: Tensor.
            State where to evaluate the value.
        action: Tensor, optional.
            First action of simulation.
        """
        unfreeze_parameters(self.policy)
        with DisableGradient(
            self.sim.dynamical_model, self.sim.reward_model, self.sim.termination_model
        ):
            sim_observation = self.sim.simulate(state, self.policy, action)

        if isinstance(self.value_function, IntegrateQValueFunction):
            cm = DisableGradient(self.value_function.q_function)
        else:
            cm = DisableGradient(self.value_function)
        with cm:
            v = mc_return(
                sim_observation,
                gamma=self.gamma,
                lambda_=self.lambda_,
                value_function=self.value_function,
                reward_transformer=self.reward_transformer,
                entropy_regularization=self.entropy_regularization,
                reduction="none",
            )

        v = v.reshape(self.sim.num_samples, state.shape[0], -1).mean(0)
        v = v[:, 0]  # In cases of ensembles return first component.
        return v
