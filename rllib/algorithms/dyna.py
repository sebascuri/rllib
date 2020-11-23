"""ModelBasedAlgorithm."""
import torch

from .abstract_algorithm import AbstractAlgorithm
from .abstract_mb_algorithm import AbstractMBAlgorithm


class Dyna(AbstractAlgorithm, AbstractMBAlgorithm):
    """Dyna Algorithm."""

    def __init__(
        self,
        base_algorithm,
        dynamical_model,
        reward_model,
        num_steps=1,
        num_samples=15,
        termination_model=None,
        *args,
        **kwargs,
    ):
        self.base_algorithm_name = base_algorithm.__class__.__name__
        AbstractAlgorithm.__init__(
            self, **{**base_algorithm.__dict__, **dict(base_algorithm.named_modules())}
        )
        AbstractMBAlgorithm.__init__(
            self,
            dynamical_model,
            reward_model,
            num_steps=num_steps,
            num_samples=num_samples,
            termination_model=termination_model,
        )
        self.base_algorithm = base_algorithm
        self.base_algorithm.criterion = type(self.base_algorithm.criterion)(
            reduction="mean"
        )

    def forward(self, observation):
        """Rollout model and call base algorithm with transitions."""
        real_loss = self.base_algorithm.forward(observation)
        with torch.no_grad():
            state = observation.state[..., 0, :]
            sim_observation = self.simulate(state, self.policy, stack_obs=True)
        sim_loss = self.base_algorithm.forward(sim_observation)
        return real_loss + sim_loss

    def update(self):
        """Update base algorithm."""
        self.base_algorithm.update()

    def reset(self):
        """Reset base algorithm."""
        self.base_algorithm.reset()

    def info(self):
        """Get info from base algorithm."""
        return self.base_algorithm.info()

    def reset_info(self):
        """Reset info from base algorithm."""
        return self.base_algorithm.reset_info()

    def set_policy(self, new_policy):
        """Set policy in base algorithm."""
        self.base_algorithm.set_policy(new_policy)
