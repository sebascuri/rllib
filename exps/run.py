"""Default Runner file for RLLib Experiments.."""

import argparse
import importlib

from gym.envs import registry

from rllib.agent import AGENTS
from rllib.environment import GymEnvironment
from rllib.util.training.agent_training import evaluate_agent, train_agent
from rllib.util.utilities import set_random_seed

try:
    from dm_control.suite import BENCHMARKING

    from rllib.environment.dm_environment import DMSuiteEnvironment
except Exception:  # dm_control not installed.
    BENCHMARKING = []

gym_envs = list(registry.env_specs.keys())
dm_envs = [f"{env}/{task}" for (env, task) in BENCHMARKING]


def main(args):
    """Run main function with arguments."""
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
        exploration_episodes=args.exploration_episodes,
        base_agent_name=args.base_agent,
    )
    agent.algorithm.num_steps = 4

    # %% Train Agent.
    train_agent(
        agent=agent,
        environment=environment,
        num_episodes=args.num_train,
        max_steps=args.max_steps,
        print_frequency=1,
        render=False,
    )

    # %% Evaluate agent.
    evaluate_agent(
        agent=agent,
        environment=environment,
        num_episodes=args.num_test,
        max_steps=args.max_steps,
        render=args.render_test,
    )
    agent.logger.export_to_json()  # Save statistics.


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run experiment on RLLib.")
    parser.add_argument(
        "environment", type=str, help="Environment name.", choices=gym_envs + dm_envs
    )
    parser.add_argument("agent", type=str, help="Agent name.", choices=AGENTS)
    parser.add_argument(
        "--base-agent", default="SAC", type=str, help="Base agent name.", choices=AGENTS
    )
    parser.add_argument("--seed", type=int, default=0, help="Random Seed.")
    parser.add_argument("--max-steps", type=int, default=1000, help="Maximum steps.")
    parser.add_argument("--num-train", type=int, default=200, help="Training episodes.")
    parser.add_argument("--num-test", type=int, default=5, help="Testing episodes.")
    parser.add_argument(
        "--exploration-episodes", type=int, default=0, help="Exploration Episodes."
    )
    parser.add_argument("--render-test", action="store_true", default=False)

    main(parser.parse_args())
