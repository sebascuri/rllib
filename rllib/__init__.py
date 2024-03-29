import gym
import gym_toytext
import numpy as np
import torch

torch.set_default_dtype(torch.float32)
gym.logger.set_level(gym.logger.ERROR)
np.set_printoptions(precision=3)
