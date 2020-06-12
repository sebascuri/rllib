"""Run optimistic exploration experiments."""

from lsf_runner import make_commands, init_runner
from exps.gpucrl.inverted_pendulum import ACTION_COST


runner = init_runner(
    f"GPUCRL_Inverted_Pendulum_plot",
    num_threads=1, num_workers=2)

cmd_list = make_commands(
    'mbmppo.py',
    base_args={
        'num-threads': 1,
        'model-kind': 'ProbabilisticEnsemble',
        'model-learn-num-iter': 50,
        'seed': 1
    },
    fixed_hyper_args={},
    common_hyper_args={
        'exploration':  ['expected', 'thompson'],  # 'expected', 'thompson', 'optimistic'
        'action-cost': [0],  # 0, ACTION_COST, 2 * ACTION_COST],  # ,
    },
    algorithm_hyper_args={},
)

runner.run_batch(cmd_list)