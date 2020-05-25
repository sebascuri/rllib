"""Run HalfCheetah MPC."""

from lsf_runner import make_commands, init_runner
from exps.gpucrl.pusher import ACTION_COST
import os


runner = init_runner(f'GPUCRL_HalfCheetah_mpc', num_threads=2, wall_time=1439)

cmd_list = make_commands(
    'mpc.py',
    base_args={},
    fixed_hyper_args={},
    common_hyper_args={
        'seed': [0, 1, 2, 3, 4],
        'exploration': ['thompson', 'optimistic', 'expected'],
        'model-kind': ['ProbabilisticEnsemble', 'DeterministicEnsemble'],
        'action-cost': [0, ACTION_COST, 5 * ACTION_COST, 10 * ACTION_COST],
    },
    algorithm_hyper_args={},
)
runner.run(cmd_list)
if 'AWS' in os.environ:
    os.system("sudo shutdown")
