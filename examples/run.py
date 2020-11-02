"""Default Runner file for RLLib Experiments.."""

import argparse
import importlib

from gym.envs import registry

from rllib.agent import AGENTS
from rllib.environment import GymEnvironment
from rllib.util.training.agent_training import evaluate_agent, train_agent
from rllib.util.utilities import RewardTransformer, set_random_seed

try:
    from dm_control.suite import BENCHMARKING

    from rllib.environment.dm_environment import DMSuiteEnvironment
except Exception:  # dm_control not installed.
    BENCHMARKING = []

gym_envs = list(registry.env_specs.keys())
dm_envs = [f"{env}/{task}" for (env, task) in BENCHMARKING]


def init_experiment(args, **kwargs):
    """Initialize experiment."""
    arg_dict = vars(args)
    arg_dict.update(kwargs)
    arg_dict = {k: v for k, v in arg_dict.items() if v is not None}
    arg_dict.pop("environment")
    # %% Set Random seeds.
    set_random_seed(args.seed)

    # %% Initialize environment.
    if args.environment in gym_envs:
        environment = GymEnvironment(args.environment, seed=args.seed)
    else:
        env_name, env_task = args.environment.split("/")
        environment = DMSuiteEnvironment(env_name, env_task, seed=args.seed)

    # %% Initialize module.
    agent_module = importlib.import_module("rllib.agent")
    agent = getattr(agent_module, f"{args.agent}Agent").default(
        environment,
        reward_transformer=RewardTransformer(scale=args.reward_scale),
        **arg_dict,
    )
    agent.logger.save_hparams(arg_dict)

    return agent, environment


def train(agent, environment, args):
    """Train agent."""
    train_agent(
        agent=agent,
        environment=environment,
        num_episodes=args.num_train,
        max_steps=args.max_steps,
        eval_frequency=args.eval_frequency,
        print_frequency=args.print_frequency,
        render=args.render_train,
    )


def evaluate(agent, environment, args):
    """Evaluate agent."""
    evaluate_agent(
        agent=agent,
        environment=environment,
        num_episodes=args.num_test,
        max_steps=args.max_steps,
        render=args.render_test,
    )
    agent.logger.export_to_json()  # Save statistics.


def main(args, **kwargs):
    """Run main function with arguments."""
    agent, environment = init_experiment(args, **kwargs)
    train(agent, environment, args)
    evaluate(agent, environment, args)


def get_experiment_parser():
    """Get experiment parser."""
    parser = argparse.ArgumentParser("Run experiment on RLLib.")
    parser.add_argument(
        "environment", type=str, help="Environment name.", choices=gym_envs + dm_envs
    )
    parser.add_argument("agent", type=str, help="Agent name.", choices=AGENTS)
    parser.add_argument(
        "--base-agent", default="SAC", type=str, help="Base agent name.", choices=AGENTS
    )
    parser.add_argument("--seed", type=int, default=0, help="Random Seed.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount Factor.")
    parser.add_argument("--reward-scale", type=float, default=1.0, help="Reward Scale.")

    parser.add_argument("--lr", type=float, default=None, help="Learning rate.")
    parser.add_argument(
        "--eta", type=float, default=None, help="Entropy Regularization coefficient."
    )
    parser.add_argument(
        "--entropy-regularization",
        dest="entropy_regularization",
        action="store_true",
        default=None,
        help="Weather to use entropy regularization.",
    )
    parser.add_argument(
        "--entropy-constraint",
        dest="entropy_regularization",
        action="store_false",
        help="Weather to use an entropy constraint.",
    )

    parser.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help="Posterior KL-divergence coefficient (See MPO algorithm).",
    )

    parser.add_argument(
        "--epsilon-mean",
        type=float,
        default=None,
        help="KL-divergence mean coefficient.",
    )
    parser.add_argument(
        "--epsilon-var",
        type=float,
        default=None,
        help="KL-divergence variance coefficient.",
    )

    parser.add_argument(
        "--kl-regularization",
        dest="kl_regularization",
        action="store_true",
        default=None,
        help="Weather to use kl regularization.",
    )
    parser.add_argument(
        "--kl-constraint",
        dest="kl_regularization",
        action="store_false",
        help="Weather to use an kl constraint.",
    )

    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Number of steps to predict. Useful for models and experience replay.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of parallel samples to compute integrals.",
    )

    parser.add_argument("--max-steps", type=int, default=1000, help="Maximum steps.")
    parser.add_argument("--num-train", type=int, default=200, help="Training episodes.")
    parser.add_argument("--num-test", type=int, default=5, help="Testing episodes.")
    parser.add_argument(
        "--print-frequency", type=int, default=1, help="Frequency to print results."
    )
    parser.add_argument(
        "--eval-frequency", type=int, default=10, help="Frequency to evaluate the mean."
    )
    parser.add_argument(
        "--exploration-episodes",
        type=int,
        default=10,
        help="Exploration Episodes before learning the policy.",
    )
    parser.add_argument(
        "--exploration-steps",
        type=int,
        default=1000,
        help="Exploration steps before learning the policy.",
    )
    parser.add_argument(
        "--model-learn-exploration-episodes",
        type=int,
        default=5,
        help="Exploration Episodes before learning the model.",
    )
    parser.add_argument("--render-train", action="store_true", default=False)
    parser.add_argument("--render-test", action="store_true", default=False)
    return parser


if __name__ == "__main__":
    main(get_experiment_parser().parse_args(), num_steps=4)
