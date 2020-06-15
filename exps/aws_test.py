"""Python Script Template."""

import os

from lsf_runner import init_runner, make_commands

runner = init_runner(f"gp_regression", num_threads=1, wall_time=1439)

cmd_list = make_commands(
    "gp_regression.py",
    base_args={},
    fixed_hyper_args={},
    common_hyper_args={"seed": [0, 1, 2, 3, 4]},
    algorithm_hyper_args={},
)
runner.run(cmd_list)
if "AWS" in os.environ:
    os.system("sudo shutdown")
