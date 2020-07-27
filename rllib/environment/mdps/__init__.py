"""Common MDPs in RL literature."""
from gym.envs.registration import register

from .baird_star import BairdStar
from .boyan_chain import BoyanChain
from .double_chain import DoubleChainProblem
from .grid_world import EasyGridWorld
from .random_mdp import RandomMDP
from .single_chain import SingleChainProblem
from .two_state import TwoStateProblem

register(id="BairdStar-v0", entry_point="rllib.environment.mdps.baird_star:BairdStar")
register(
    id="BoyanChain-v0", entry_point="rllib.environment.mdps.boyan_chain:BoyanChain"
)
register(
    id="DoubleChainProblem-v0",
    entry_point="rllib.environment.mdps.double_chain:DoubleChainProblem",
)
register(
    id="EasyGridWorld-v0", entry_point="rllib.environment.mdps.grid_world:EasyGridWorld"
)
register(id="RandomMDP-v0", entry_point="rllib.environment.mdps.random_mdp:RandomMDP")
register(
    id="SingleChainProblem-v0",
    entry_point="rllib.environment.mdps.single_chain:SingleChainProblem",
)
register(
    id="TwoStateProblem-v0",
    entry_point="rllib.environment.mdps.two_state:TwoStateProblem",
)
