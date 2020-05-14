"""Run optimistic exploration experiments."""

from lsf_runner import make_commands, init_runner

runner = init_runner('GPUCRL_Reacher', num_threads=1, wall_time=1439)

cmd_list = make_commands(
    'mpc.py',
    base_args={},
    fixed_hyper_args={},
    common_hyper_args={
        'exploration': ['thompson', 'optimistic', 'expected'],
        'model-kind': ['ProbabilisticEnsemble', 'DeterministicEnsemble'],
        'action-cost': [0, 0.01, 0.05],
    },
    algorithm_hyper_args={},
)
runner.run(cmd_list)
