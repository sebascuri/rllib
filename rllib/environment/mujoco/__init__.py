"""Python Script Template."""
from gym.envs.registration import register

base = "rllib.environment.mujoco"

register(id="MBAnt-v0", entry_point=f"{base}.ant:MBAntEnv")
register(id="MBHalfCheetah-v0", entry_point=f"{base}.half_cheetah:MBHalfCheetahEnv")
register(id="MBHopper-v0", entry_point=f"{base}.hopper:MBHopperEnv")
register(id="MBHumanoid-v0", entry_point=f"{base}.hopper:MBHumanoidEnv")
register(id="MBSwimmer-v0", entry_point=f"{base}.hopper:MBSwimmerEnv")
register(
    id="PendulumSwingUp-v0", entry_point=f"{base}.pendulum_swing_up:PendulumSwingUpEnv"
)
register(id="MBWalker2d-v0", entry_point=f"{base}.hopper:MBWalker2dEnv")
