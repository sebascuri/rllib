"""A gradient based solver runs SGD on the action sequence."""
from torch.optim import Adam

from rllib.dataset.utilities import stack_list_of_tuples
from rllib.util.neural_networks.utilities import DisableGradient
from rllib.util.rollout import rollout_actions
from rllib.util.value_estimation import discount_sum

from .abstract_solver import MPCSolver


class GradientBasedSolver(MPCSolver):
    """Gradient based MPC solver."""

    def __init__(self, num_iter=5, lr=1e-2, *args, **kwargs):
        super().__init__(num_iter=num_iter, *args, **kwargs)
        self.lr = lr

    def get_candidate_action_sequence(self):
        """Get candidate action sequence."""
        actions = self.mean.detach().clone()
        actions.requires_grad = True
        return actions

    def get_best_action(self, action_sequence, returns):
        """Return action_sequence. Not implemented."""
        return action_sequence

    def update_sequence_generation(self, elite_actions) -> None:
        """Do Nothing. Not implemented."""
        pass

    def forward(self, state):
        """Compute SGD on actions estimation."""
        batch_shape = state.shape[:-1]
        self.initialize_actions(batch_shape)

        actions = self.get_candidate_action_sequence()
        optimizer = Adam([actions], lr=self.lr)

        for i in range(self.num_iter):
            optimizer.zero_grad()
            with DisableGradient(self.dynamical_model, self.reward_model):
                trajectory = rollout_actions(
                    self.dynamical_model, self.reward_model, actions, state
                )

            returns = discount_sum(
                stack_list_of_tuples(trajectory, dim=-2).reward, gamma=self.gamma
            )
            (-returns).sum().backward()
            optimizer.step()

        self.mean = actions.detach().clone()
        return actions
