"""Implementation of GP-UCB algorithm."""

import gpytorch
import torch

from rllib.agent import AbstractAgent
from rllib.policy import AbstractPolicy
from rllib.util.gaussian_processes import ExactGP, SparseGP
from rllib.util.gaussian_processes.utilities import add_data_to_gp, bkb
from rllib.util.parameter_decay import Constant, ParameterDecay


class GPUCBPolicy(AbstractPolicy):
    """GP UCB Policy.

    Implementation of GP-UCB algorithm.
    GP-UCB uses a GP to maintain the predictions of a distribution over actions.

    The algorithm selects the action as:
    x = arg max mean(x) + beta * std(x)
    where mean(x) and std(x) are the mean and standard devations of the GP at loc x.

    Parameters
    ----------
    gp: initialized GP model.
    x: discretization of domain.
    beta: exploration parameter.

    References
    ----------
    Srinivas, N., Krause, A., Kakade, S. M., & Seeger, M. (2009).
    Gaussian process optimization in the bandit setting: No regret and experimental
    design.
    """

    def __init__(self, gp, x, beta=2.0, noisy=False):
        if x.ndim == 1:
            dim_action = (1,)
        else:
            dim_action = x.shape
        super().__init__(
            dim_state=(),
            dim_action=dim_action,
            num_states=1,
            num_actions=-1,
            deterministic=True,
            action_scale=1.0,
        )
        self.gp = gp
        self.gp.eval()
        self.gp.likelihood.eval()
        self.x = x
        self.noisy = noisy
        if not isinstance(beta, ParameterDecay):
            beta = Constant(beta)
        self.beta = beta

    def forward(self, state):
        """Call the GP-UCB algorithm."""
        if self.noisy:
            lower, upper = self.x[0], self.x[-1]
            test_x = lower + torch.rand(len(self.x)) * (upper - lower)
        else:
            test_x = self.x

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self.gp(test_x)
            ucb = pred.mean + self.beta() * pred.stddev

            max_id = torch.argmax(ucb)
            next_point = test_x[[[max_id]]]
            return next_point, torch.zeros(1)

    def update(self):
        """Update policy parameters."""
        self.beta.update()


class GPUCBAgent(AbstractAgent):
    """Agent that implements the GP-UCB algorithm.

    Parameters
    ----------
    gp: initialized GP model.
    x: discretization of domain.
    beta: exploration parameter.

    References
    ----------
    Srinivas, N., Krause, A., Kakade, S. M., & Seeger, M. (2009).
    Gaussian process optimization in the bandit setting: No regret and experimental
    design. ICML.

    Calandriello, D., Carratino, L., Lazaric, A., Valko, M., & Rosasco, L. (2019).
    Gaussian process optimization with adaptive sketching: Scalable and no regret. COLT.

    Chowdhury, S. R., & Gopalan, A. (2017).
    On kernelized multi-armed bandits. JMLR.
    """

    def __init__(self, gp, x, beta=2.0, noisy=False, *args, **kwargs):
        self.policy = GPUCBPolicy(gp, x, beta, noisy=noisy)
        super().__init__(
            train_frequency=1, num_rollouts=0, gamma=1, comment=gp.name, *args, **kwargs
        )

    def observe(self, observation) -> None:
        """Observe and update GP."""
        super().observe(observation)  # Already calls self.policy.update()
        add_data_to_gp(
            self.policy.gp, observation.action.unsqueeze(-1), observation.reward
        )
        self.logger.update(num_gp_inputs=len(self.policy.gp.train_targets))
        if isinstance(self.policy.gp, SparseGP):
            inducing_points = torch.cat(
                (self.policy.gp.xu, observation.action.unsqueeze(-1)), dim=-2
            )

            inducing_points = bkb(self.policy.gp, inducing_points)
            self.policy.gp.set_inducing_points(inducing_points)
            self.logger.update(num_inducing_points=self.policy.gp.xu.shape[0])

    @classmethod
    def default(cls, environment, *args, **kwargs):
        """See `AbstractAgent.default'."""
        x = torch.linspace(-1, 6, 100)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        likelihood.noise_covar.noise = 0.1 ** 2
        x0 = x[x > 0.2][[0]].unsqueeze(-1)
        _, y0, _, _ = environment.step(x0.numpy())
        y0 = torch.tensor(y0, dtype=torch.get_default_dtype())

        model = ExactGP(x0, y0, likelihood)
        return cls(model, x, beta=2.0, noisy=False, *args, **kwargs)
