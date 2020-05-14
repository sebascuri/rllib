"""Python Script Template."""

from exps.gpucrl.util import parse_results

base_dir = 'runs/Cartpoleenv/MPCAgent/'
results = parse_results(base_dir)

for name, result in results.items():
    print(name, result[0])
