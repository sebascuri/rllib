"""Run optimistic exploration experiments."""

from lsf_runner import make_commands, init_runner

algorithm_hyper_args = {
    'model-kind': ['FeatureGP'],
    'model-num-features': [256, 625, 1296],
    'model-feature-approximation': ['QFF', 'RFF'],
}

runner = init_runner(
    f"GPUCRL_Inverted_Pendulum_{algorithm_hyper_args['model-kind'][0]}",
    num_threads=1, num_workers=12)

cmd_list = make_commands(
    'mbmppo.py',
    base_args={'num-threads': 1},
    fixed_hyper_args={},
    common_hyper_args={
        'seed': [0, 1, 2, 3, 4],
        'exploration': ['thompson'],
        'action-cost': [0, 0.1, 0.2],
    },
    algorithm_hyper_args=algorithm_hyper_args,
)

runner.run_batch(cmd_list)
