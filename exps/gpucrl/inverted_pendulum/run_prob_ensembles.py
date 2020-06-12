"""Run optimistic exploration experiments."""

from lsf_runner import make_commands, init_runner
from exps.gpucrl.inverted_pendulum import ACTION_COST


#  supermodularity.


nn_ensemble_hyper_params = {
    'model-kind': ['ProbabilisticEnsemble'],
    'model-learn-num-iter': [50],
}

for algorithm_hyper_args in [nn_ensemble_hyper_params]:
    runner = init_runner(
        f"GPUCRL_Inverted_Pendulum_{algorithm_hyper_args['model-kind'][0]}",
        num_threads=1, num_workers=18)

    cmd_list = make_commands(
        'mbmppo.py',
        base_args={'num-threads': 1},
        fixed_hyper_args={},
        common_hyper_args={
            'seed': [0, 1, 2, 3, 4],
            'exploration':  ['expected', 'optimistic', 'thompson'],
            'action-cost': [0, ACTION_COST, 2 * ACTION_COST],
        },
        algorithm_hyper_args=algorithm_hyper_args,
    )

    runner.run_batch(cmd_list)