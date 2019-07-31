import matplotlib.pyplot as plt
from rllib.agent import QLearningAgent, GQLearningAgent, DQNAgent, DDQNAgent
from rllib.util import rollout_agent
from rllib.value_function import NNQFunction
from rllib.dataset import ExperienceReplay
from rllib.exploration_strategies import EpsGreedy
from rllib.environment.systems import InvertedPendulum
from rllib.environment import SystemEnvironment
import numpy as np
import torch.nn.functional as func
import torch.optim
import pickle


def reward_function(state, action):
    cos_angle = torch.cos(state[..., 0])
    velocity = state[..., 1]
    angle_tolerance = tolerance(cos_angle, lower=0.95, upper=1., margin=0.1)
    velocity_tolerance = tolerance(velocity, lower=-.5, upper=0.5, margin=0.5)
    return angle_tolerance * velocity_tolerance