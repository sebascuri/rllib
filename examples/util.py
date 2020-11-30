"""Running utilities."""
import importlib

import yaml
from gym.envs import registry

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


def parse_config_file(file_dir=None):
    """Parse configuration file."""
    if file_dir is None:
        return {}
    with open(file_dir, "r") as file:
        kwargs = yaml.safe_load(file)
    return kwargs


def init_experiment(args, **kwargs):
    """Initialize experiment."""
    arg_dict = vars(args)
    arg_dict.update(kwargs)
    arg_dict.update(parse_config_file(args.config_file))
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
        reward_transformer=RewardTransformer(scale=arg_dict.get("reward_scale", 1.0)),
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
