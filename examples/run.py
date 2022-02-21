"""Default Runner file for RLLib Experiments.."""

from examples.experiment_parser import parser
from examples.util import evaluate, init_experiment, train


def main(args):
    """Run main function with arguments."""
    # Initialize agent and environment.
    agent, environment = init_experiment(args)

    # Train and evaluate.
    train(agent, environment, args)
    if args.num_test > 0:
        evaluate(agent, environment, args)


if __name__ == "__main__":
    main(parser.parse_args())
