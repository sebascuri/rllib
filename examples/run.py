"""Default Runner file for RLLib Experiments.."""

import argparse

from examples.util import evaluate, init_experiment, load_from_directory, train
from rllib.agent import AGENTS


def main(args, **kwargs):
    """Run main function with arguments."""
    # Initialize agent and environment.
    agent, environment = init_experiment(args, **kwargs)

    # Load training agent.
    load_from_directory(agent, args.log_dir)

    # Train and evaluate.
    train(agent, environment, args)
    if args.num_test > 0:
        evaluate(agent, environment, args)


def get_experiment_parser():
    """Get experiment parser."""
    parser = argparse.ArgumentParser("Run experiment on RLLib.")
    parser.add_argument(
        "--env-config",
        type=str,
        help="File with environment config.",
        default="config/envs/cart-pole.yaml",
    )
    parser.add_argument("agent", type=str, help="Agent name.", choices=AGENTS)
    parser.add_argument(
        "--agent-config", type=str, help="File with agent configuration.", default=None
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        help="Directory with agent, by default will get last directory.",
        default=None,
    )

    parser.add_argument("--seed", type=int, default=0, help="Random Seed.")
    parser.add_argument("--num-train", type=int, default=200, help="Training episodes.")
    parser.add_argument("--num-test", type=int, default=0, help="Testing episodes.")
    parser.add_argument(
        "--print-frequency", type=int, default=1, help="Frequency to print results."
    )
    parser.add_argument(
        "--eval-frequency", type=int, default=0, help="Frequency to evaluate the mean."
    )
    parser.add_argument("--render-train", action="store_true", default=False)
    parser.add_argument("--render-test", action="store_true", default=False)
    return parser


if __name__ == "__main__":
    main(get_experiment_parser().parse_args())
