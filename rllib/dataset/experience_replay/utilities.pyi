from rllib.agent import AbstractAgent
from rllib.environment import AbstractEnvironment
from .experience_replay import ExperienceReplay


def init_er_from_er(target_er: ExperienceReplay,
                    source_er: ExperienceReplay) -> None: ...


def init_er_from_environment(target_er: ExperienceReplay,
                             environment: AbstractEnvironment) -> None: ...


def init_er_from_rollout(target_er: ExperienceReplay, agent: AbstractAgent,
                         environment: AbstractEnvironment, max_steps: int = 1000
                         ) -> None: ...