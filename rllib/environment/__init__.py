"""Import environments."""
from gym.envs.registration import register

import rllib.environment.mdps
import rllib.environment.mujoco
import rllib.environment.system_environment
import rllib.environment.vectorized

from .abstract_environment import AbstractEnvironment
from .gym_environment import *
from .mdp import *
from .system_environment import *
from .utilities import *

mini_atary_entry = "rllib.environment.miniatari_environment:MiniAtariEnv"
register(
    id="MiniAsterix-v0", entry_point=mini_atary_entry, kwargs={"env_name": "asterix"}
)
register(
    id="MiniBreakout-v0", entry_point=mini_atary_entry, kwargs={"env_name": "breakout"}
)
register(
    id="MiniFreeway-v0", entry_point=mini_atary_entry, kwargs={"env_name": "freeway"}
)
register(
    id="MiniSeaquest-v0", entry_point=mini_atary_entry, kwargs={"env_name": "seaquest"}
)
register(
    id="MiniSpaceInvaders-v0",
    entry_point=mini_atary_entry,
    kwargs={"env_name": "space_invaders"},
)
