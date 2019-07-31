import numpy as np


__all__ = ['mc_value', 'sum_discounted_rewards']


def mc_value_slow(trajectory, gamma):
    q_estimate = []
    for t in range(len(trajectory)):
        q_t = 0
        for i, observation in enumerate(trajectory[t:]):
            q_t = q_t + gamma ** i * observation.reward
        q_estimate.append(q_t)

    return np.array(q_estimate)


def mc_value(trajectory, gamma):
    value_estimate = [0] * len(trajectory)
    value_estimate[-1] = trajectory[-1].reward

    for t in reversed(range(1, len(trajectory))):
        value_estimate[t-1] = trajectory[t].reward + gamma * value_estimate[t]

    return np.array(value_estimate)


def sum_discounted_rewards(trajectory, gamma):
    rewards = []
    for observation in trajectory:
        rewards.append(observation.reward)
    rewards = np.array(rewards)
    i = np.arange(len(rewards))
    return np.sum(rewards * np.power(gamma, i))


# if __name__ == "__main__":
#     from rllib.dataset import Observation
#     trajectory = []
#     for i in range(10):
#         observation = Observation(state=0, action=1, next_state=2, reward=1, done=0)
#         trajectory.append(observation)
#
#     gamma = 0.9
#     print(mc_value(trajectory, gamma),
#           mc_value_slow(trajectory, gamma),
#           sum_discounted_rewards(trajectory, gamma)
#           )
