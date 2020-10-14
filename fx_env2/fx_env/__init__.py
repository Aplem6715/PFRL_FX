from gym.envs.registration import register

register(
    id='FxEnv-v0',
    entry_point='fx_env.fx_env.envs:FxEnv',
)
