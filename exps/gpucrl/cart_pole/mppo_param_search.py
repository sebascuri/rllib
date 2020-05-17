"""Run Cart Pole MB-MMPO Hparam search."""

from lsf_runner import make_commands, init_runner

runner = init_runner(f'GPUCRL_CartPole_mbmppo', num_threads=1)

cmd_list = make_commands(
    'mbmppo.py',
    base_args={'exploration': 'expected'},
    fixed_hyper_args={},
    common_hyper_args={
        'plan-horizon': [0, 1, 4],
        'mppo-num-iter': [50, 100, 200],
        'mppo-eta': [.5, 1., 1.5],
        'mppo-eta-mean': [.7, 1., 1.3],
        'mppo-eta-var': [.1, 1., 5.],
        'sim-initial-states-num-trajectories': [8],
        'sim-initial-dist-num-trajectories': [16, 32]
    },
    algorithm_hyper_args={},
)
runner.run(cmd_list)
