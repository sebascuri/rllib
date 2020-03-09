import matplotlib.pyplot as plt
from rllib.util.rollout import rollout_agent


def rollout(environment, agent, num_episodes, max_steps):
    rollout_agent(environment, agent, num_episodes=num_episodes, max_steps=max_steps)

    for key, log in agent.logs.items():
        plt.plot(log.episode_log)
        plt.xlabel('Episode')
        plt.ylabel(' '.join(key.split('_')).capitalize())
        plt.title('{} in {}'.format(agent.name, environment.name))
        plt.show()

    rollout_agent(environment, agent, max_steps=max_steps, num_episodes=1, render=True)
