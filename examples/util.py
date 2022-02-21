"""Running utilities."""
import importlib
import os

import yaml
from gym.envs import registry

from examples.experiment_parser import Experiment
from rllib.environment import GymEnvironment
from rllib.util.training.agent_training import evaluate_agent, train_agent
from rllib.util.utilities import load_random_state, set_random_seed

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


class EnvironmentBuilder:
    """Environment Builder."""

    def __init__(self, args: Experiment):
        self.args = args
        self.name = args.environment

    def create_environment(self):
        """Create environment."""
        if self.name in gym_envs:
            environment = GymEnvironment(self.name, seed=self.args.seed)
        else:
            env_name, env_task = self.name.split("/")
            environment = DMSuiteEnvironment(env_name, env_task, seed=self.args.seed)
        return environment


class AgentBuilder:
    """Agent builder."""

    def __init__(self, args: Experiment):
        self.args = args
        self.agent_config = parse_config_file(self.args.agent_config)

    def create_agent(self, environment):
        """Create agent."""
        agent_module = importlib.import_module("rllib.agent")
        agent = getattr(agent_module, f"{self.args.agent}Agent").default(
            environment, tensorboard=self.args.tensorboard, **self.agent_config
        )
        return agent


def init_experiment(args: Experiment):
    """Initialize experiment."""
    # Set Random seeds.
    set_random_seed(args.seed)

    # Initialize environment.
    env_builder = EnvironmentBuilder(args)
    environment = env_builder.create_environment()

    # Initialize Agent.
    agent_builder = AgentBuilder(args)
    agent = agent_builder.create_agent(environment)

    if args.load_from_dir:
        assert args.log_dir is not None
        load_from_directory(agent, args.log_dir)

    return agent, environment


def load_from_directory(agent, directory=None):
    """Load an agent from a directory."""
    if directory is None:  # find latest agent.
        path, current_log_dir = os.path.split(agent.logger.log_dir)
        listdir = os.listdir(path)
        if len(listdir) < 2:
            return
        environment = current_log_dir.split("_")[0]
        latest_directory_list = sorted(
            filter(lambda x: x.startswith(environment), listdir)
        )
        if len(latest_directory_list) < 2:
            return
        latest_directory = latest_directory_list[-2]
        directory = os.path.join(path, latest_directory)

    # Load agent.
    agent.load(f"{directory}/last.pkl")
    agent.logger.change_log_dir(directory)

    # Load random state.
    load_random_state(directory)


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
