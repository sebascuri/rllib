"""Run optimistic exploration experiments."""

from lsf_runner import make_commands, init_runner


nn_ensemble_hyper_params = {
    'model-kind': ['ProbabilisticEnsemble', 'DeterministicEnsemble'],
    'model-learn-num-iter': [50]
}

for algorithm_hyper_args in [nn_ensemble_hyper_params]:
    runner = init_runner(
        f"GPUCRL_Inverted_Pendulum_{algorithm_hyper_args['model-kind'][0]}",
        num_threads=1, num_workers=12)

    cmd_list = make_commands(
        'mbmppo.py',
        base_args={},
        fixed_hyper_args={},
        common_hyper_args={
            'seed': [1, 2, 3, 4],
            'exploration': ['expected', 'optimistic', 'thompson'],
            'action-cost': [0, 0.1, 0.2, 0.5],
        },
        algorithm_hyper_args=algorithm_hyper_args,
    )

    runner.run_batch(cmd_list)
