"""Python Script Template."""
from gym.envs.registration import register

base = "rllib.environment.mujoco"

register(id="MBAnt-v0", entry_point=f"{base}.ant:MBAntEnv")
register(id="MBCartPole-v0", entry_point=f"{base}.cart_pole:MBCartPoleEnv")
register(id="MBHalfCheetah-v0", entry_point=f"{base}.half_cheetah:MBHalfCheetahEnv")
register(id="MBHopper-v0", entry_point=f"{base}.hopper:MBHopperEnv")
register(id="MBHumanoid-v0", entry_point=f"{base}.humanoid:MBHumanoidEnv")
register(
    id="MBInvertedPendulum-v0",
    entry_point=f"{base}.inverted_pendulum:MBInvertedPendulumEnv",
)
register(
    id="MBInvertedDoublePendulum-v0",
    entry_point=f"{base}.inverted_double_pendulum:MBInvertedDoublePendulumEnv",
)
register(
    id="PendulumSwingUp-v0", entry_point=f"{base}.pendulum_swing_up:PendulumSwingUpEnv"
)
register(id="MBPusher-v0", entry_point=f"{base}.pusher:MBPusherEnv")

register(id="MBReacher2d-v0", entry_point=f"{base}.reacher_2d:MBReacherEnv")
register(id="MBReacher3d-v0", entry_point=f"{base}.reacher_3d:MBReacher3DEnv")

register(id="MBSwimmer-v0", entry_point=f"{base}.swimmer:MBSwimmerEnv")
register(id="MBWalker2d-v0", entry_point=f"{base}.walker_2d:MBWalker2dEnv")
