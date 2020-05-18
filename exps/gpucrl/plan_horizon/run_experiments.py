"""Python Script Template."""

from lsf_runner import make_commands, init_runner

runner = init_runner(f'GPUCRL_CartPole_PLAN_HORIZON', num_threads=1)
runner.num_workers = 1
cmd_list = make_commands(
    '../cart_pole/mbmppo.py',
    base_args={'exploration': 'expected'},
    fixed_hyper_args={},
    common_hyper_args={
        'plan-horizon': [0, 1, 4],
        'seed': [1, 2, 3, 4]
    },
    algorithm_hyper_args={},
)
runner.run(cmd_list)
