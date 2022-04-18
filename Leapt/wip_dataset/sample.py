# prefill data storage
key_env = jax.random.PRNGKey(42)
core_env = env_fn(
    action_repeat=FLAGS.action_repeat,
    batch_size=FLAGS.num_envs * jax.local_device_count(),
    episode_length=FLAGS.episode_length,
)
action_size = core_env.action_size
key_envs = jax.random.split(key_env, jax.local_device_count())
step_fn = jax.jit(core_env.step)
reset_fn = jax.jit(jax.vmap(core_env.reset))
first_state = reset_fn(key_envs)

def collect_data(key, state):
    key, key_sample = jax.random.split(key)
    action = jax.random.uniform(key_sample, shape=(action_size), minval=-1.0, maxval=1.0)
    nstate = step_fn(state, action)

    concatenated_data = jnp.concatenate(
        [
            state.obs,
            nstate.obs,
            action,
            jnp.expand_dims(nstate.reward, axis=-1),
            jnp.expand_dims(1 - nstate.done, axis=-1),
            jnp.expand_dims(nstate.info["truncation"], axis=-1),
        ],
        axis=-1,
    )

    return key, nstate, concatenated_data

def init_replay_buffer(state, replay_buffer):

    (state, replay_buffer), _ = jax.lax.scan(
        (lambda a, b: (collect_data(*a), ())),
        (key_envs, first_state),
        (),
        length=FLAGS.num_env * jax.local_device_count(),
    )
    return state, replay_buffer
