"""Python Script Template."""
from gym.envs.registration import register

register(
    id="HalfCheetah-fullobs-v0", entry_point="exps.half_cheetah.utils:HalfCheetahEnvV2",
)
