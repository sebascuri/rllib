"""Run optimistic exploration experiments."""

from lsf_runner import make_commands, init_runner

gp_hyper_params = {
    'model-kind': ['ExactGP'],
    'model-max-num-points': [int(1e10), 200],
    'num-model-iter': [0, 30]
}

sparse_gp_hyper_params = {
    'model-kind': ['SparseGP'],
    'model-max-num-points': [int(1e10), 200],
    'model-sparse-approximation': ['DTC', 'FITC'],
    'num-model-iter': [0]
}

features_gp_hyper_params = {
    'model-kind': ['FeatureGP'],
    'model-num-features': [256, 625, 1296],
    'model-feature-approximation': ['QFF', 'RFF'],
    'num-model-iter': [0]
}

nn_ensemble_hyper_params = {
    'model-kind': ['ProbabilisticEnsemble', 'DeterministicEnsemble'],
    'num-model-iter': [50]
}

for algorithm_hyper_args in [gp_hyper_params, sparse_gp_hyper_params,
                             features_gp_hyper_params]:
    runner = init_runner(
        f"GPUCRL_Inverted_Pendulum_{algorithm_hyper_args['model-kind'][0]}",
        num_threads=1)

    cmd_list = make_commands(
        'optimistic_exploration.py',
        base_args={'train-episodes': 15},
        fixed_hyper_args={},
        common_hyper_args={
            'optimistic': [True, False],
            'mppo-eta': [0.5, 1., 1.5],
            'mppo-eta-mean': [0.5, 1.1, 1.7],
            'mppo-eta-var': [0.9, 1.1],
            'action-cost': [0, 0.2, 0.5],
            'sim-num-steps': [100, 200, 400],
            'plan-horizon': [0, 1, 4],
        },
        algorithm_hyper_args=algorithm_hyper_args,
    )
    runner.run_batch(cmd_list)