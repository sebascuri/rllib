"""Python Script Template."""
from dataclasses import dataclass, field
from typing import Optional

from argparse_dataclass import ArgumentParser


@dataclass
class Experiment:
    """Definition of Experiment dataclass."""

    agent: str = field(default="SAC", metadata=dict(help="Agent Name"))
    environment: str = field(
        default="HalfCheetah-v3", metadata=dict(help="Environment Name")
    )

    agent_config: Optional[str] = field(
        default=None, metadata=dict(help="File with agent config.")
    )
    log_dir: Optional[str] = field(
        default=None, metadata=dict(help="Directory with agent.")
    )
    load_from_dir: bool = field(
        default=False,
        metadata=dict(
            help="Whether to load from dir. Must be specified through --log_dir."
        ),
    )

    seed: int = field(default=0, metadata=dict(help="Random reed."))

    max_steps: int = field(
        default=1000, metadata=dict(help="Maximum number of steps per episode.")
    )

    num_train: int = field(
        default=200, metadata=dict(help="Number of training episodes.")
    )
    num_test: int = field(default=0, metadata=dict(help="Number of testing episodes."))
    print_frequency: int = field(
        default=1, metadata=dict(help="Frequency to print results.")
    )
    eval_frequency: int = field(
        default=0,
        metadata=dict(
            help="Frequency to evaluate the agent, i.e. without exploration."
        ),
    )

    render_train: bool = field(
        default=False,
        metadata=dict(help="Whether to render the environment during training."),
    )
    render_test: bool = field(
        default=False,
        metadata=dict(help="Whether to render the environment during testing."),
    )
    tensorboard: bool = field(
        default=False, metadata=dict(help="Whether to save results using tensorboard.")
    )


parser = ArgumentParser(Experiment)
