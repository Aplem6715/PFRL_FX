from gym.envs.registration import register

register(
    id='FxEnv-v0',
    entry_point='fx-env.envs:FxEnv',
)
