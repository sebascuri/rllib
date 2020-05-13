from dotmap import DotMap

from exps.gpucrl.inverted_pendulum.util import get_agent_and_environment
from exps.gpucrl.inverted_pendulum.plotters import plot_pendulum_trajectories
from exps.gpucrl.mb_mppo_arguments import parser
from exps.gpucrl.util import train_and_evaluate

parser.description = 'Run Swing-up Inverted Pendulum using Model-Based MPPO.'
parser.set_defaults(action_cost=0.2,
                    environment_max_steps=400,
                    train_episodes=20,
                    render_train=True)

args = parser.parse_args()
params = DotMap(vars(args))
environment, agent = get_agent_and_environment(params, 'mbmppo')

train_and_evaluate(agent, environment, params,
                   plot_callbacks=[plot_pendulum_trajectories])
