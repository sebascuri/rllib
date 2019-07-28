from collections import namedtuple

Observation = namedtuple('Observation',
                         ('state', 'action', 'reward', 'next_state', 'done'))