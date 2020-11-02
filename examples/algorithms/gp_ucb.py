"""Working example of GP-UCB."""

import gpytorch
import matplotlib.pyplot as plt
import torch

from rllib.agent.bandit import GPUCBAgent
from rllib.environment.bandit_environment import BanditEnvironment
from rllib.reward.gp_reward import GPBanditReward
from rllib.util.gaussian_processes import ExactGP, RandomFeatureGP, SparseGP
from rllib.util.rollout import rollout_agent


def plot_gp(x: torch.Tensor, model: gpytorch.models.GP, num_samples: int) -> None:
    """Plot 1-D GP.

    Parameters
    ----------
    x: points to plot.
    model: GP model.
    num_samples: number of random samples from gp.
    """
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = model(x)
        mean = pred.mean.numpy()
        error = 2 * pred.stddev.numpy()

    # Plot GP
    plt.fill_between(x, mean - error, mean + error, lw=0, alpha=0.4, color="C0")

    # Plot mean
    plt.plot(x, mean, color="C0")

    # Plot samples.
    for _ in range(num_samples):
        plt.plot(x.numpy(), pred.sample().numpy())


def plot(agent, step, objective, axes):
    """Plot GP-UCB agent at current time-step."""
    axis = axes[step % 5][step // 5]

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        test_x = agent.policy.x
        pred = agent.policy.gp(test_x)
        true_values = objective(None, test_x, None)[0].numpy()

        test_x = test_x.numpy()
        mean = pred.mean.numpy()
        error = 2 * pred.stddev.numpy()

        # Plot gp prediction
        axis.fill_between(
            test_x, mean - error, mean + error, lw=0, alpha=0.4, color="C0"
        )
        axis.plot(test_x, mean, color="C0")

        # Plot ground-truth
        axis.plot(test_x, true_values, "--", color="k")

        # Plot data
        axis.plot(
            agent.policy.gp.train_inputs[0].numpy(),
            agent.policy.gp.train_targets.numpy(),
            "x",
            markeredgewidth=2,
            markersize=5,
            color="C1",
        )

        axis.set_xlim(test_x[0], test_x[-1])
        axis.set_ylim(-2.1, 2.1)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_xlabel("Inputs")
        axis.set_ylabel("Objective")


if __name__ == "__main__":
    NUM_POINTS = 1000
    STEPS = 10
    SEED = 42
    torch.manual_seed(SEED)

    # Define objective function
    X = torch.tensor([-1.0, 1.0, 2.5, 4.0, 6])
    Y = 2 * torch.tensor([-0.5, 0.3, -0.2, 0.6, -0.5])

    x = torch.linspace(-1, 6, NUM_POINTS)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood.noise_covar.noise = 0.1 ** 2
    objective_function = ExactGP(X, Y, likelihood)
    objective_function.eval()
    objective = GPBanditReward(objective_function)

    plt.plot(X.numpy(), Y.numpy(), "*")
    plot_gp(x, objective_function, num_samples=0)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

    x0 = x[x > 0.2][[0]].unsqueeze(-1)
    y0 = objective(None, x0, None)[0].type(torch.get_default_dtype())

    for key, model in {
        "Exact": ExactGP(x0, y0, likelihood),
        "RFF": RandomFeatureGP(x0, y0, likelihood, 50, approximation="RFF"),
        "OFF": RandomFeatureGP(x0, y0, likelihood, 50, approximation="OFF"),
        "QFF": RandomFeatureGP(x0, y0, likelihood, 50, approximation="QFF"),
        "DTC": SparseGP(x0, y0, likelihood, inducing_points=x0, approximation="DTC"),
        "SOR": SparseGP(x0, y0, likelihood, inducing_points=x0, approximation="SOR"),
        "FITC": SparseGP(x0, y0, likelihood, inducing_points=x0, approximation="FITC"),
    }.items():
        model.length_scale = torch.tensor(1.0)
        environment = BanditEnvironment(
            objective, x_min=x[[0]].numpy(), x_max=x[[-1]].numpy()
        )
        agent = GPUCBAgent(model, x, beta=2.0)
        state = environment.reset()

        fig, axes = plt.subplots(5, 2)
        rollout_agent(
            environment,
            agent,
            num_episodes=STEPS,
            max_steps=1,
            callback_frequency=1,
            callbacks=[
                lambda a, step, ax=axes: plot(a, step, objective, ax)  # type: ignore
            ],
        )

        fig.suptitle(key, y=1)
        plt.show()
        print(agent)
        agent.logger.export_to_json()
