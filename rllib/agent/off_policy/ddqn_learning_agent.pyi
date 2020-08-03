from rllib.agent import QLearningAgent
from rllib.algorithms.ddqn import DDQN

class DDQNAgent(QLearningAgent):
    algorithm: DDQN
