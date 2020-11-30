"""Default Runner file for RLLib Experiments.."""

import argparse

from gym.envs import registry

from examples.util import evaluate, init_experiment, train
from rllib.agent import AGENTS

try:
    from dm_control.suite import BENCHMARKING
except Exception:  # dm_control not installed.
    BENCHMARKING = []

gym_envs = list(registry.env_specs.keys())
dm_envs = [f"{env}/{task}" for (env, task) in BENCHMARKING]


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
        "--config-file", type=str, help="File with agent configuration.", default=None
    )

    parser.add_argument("--seed", type=int, default=0, help="Random Seed.")
    parser.add_argument("--max-steps", type=int, default=1000, help="Maximum steps.")
    parser.add_argument("--num-train", type=int, default=200, help="Training episodes.")
    parser.add_argument("--num-test", type=int, default=5, help="Testing episodes.")
    parser.add_argument(
        "--print-frequency", type=int, default=1, help="Frequency to print results."
    )
    parser.add_argument(
        "--eval-frequency", type=int, default=10, help="Frequency to evaluate the mean."
    )
    parser.add_argument("--render-train", action="store_true", default=False)
    parser.add_argument("--render-test", action="store_true", default=False)
    return parser


if __name__ == "__main__":
    main(get_experiment_parser().parse_args())
