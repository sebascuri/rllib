from collections import namedtuple

Observation = namedtuple('Observation',
                         ('state', 'action', 'reward', 'next_state', 'done'))

from .dataset import TrajectoryDataset
from .experience_replay import ExperienceReplay
