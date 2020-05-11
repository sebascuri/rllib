"""Run optimistic exploration experiments."""

from lsf_runner import make_commands, init_runner

runner = init_runner('GPUCRL_CartPole', num_threads=1)

cmd_list = make_commands(
    'mpc.py',
    base_args={'train-episodes': 15},
    fixed_hyper_args={},
    common_hyper_args={
        'exploration': ['thompson', 'optimistic', 'expected'],
        'model-kind': ['ProbabilisticEnsemble'],
        'action-cost': [0.1, 0.05, 0.01],
    },
    algorithm_hyper_args={},
)
runner.run(cmd_list)
