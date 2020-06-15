#!/usr/bin/env bash

python experiments/value_function_learning.py
python experiments/pendulum_swingup_exploration.py
python experiments/exploration_experiment.py
python experiments/tabular_environments/planning.py

python experiments/algorithms/actor_critic.py
python experiments/algorithms/dpg.py
python experiments/algorithms/gp_ucb.py
python experiments/algorithms/q_learning.py
python experiments/algorithms/reinforce.py
python experiments/algorithms/sarsa.py
