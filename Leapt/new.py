

init_replay_buffer = jax.pmap(init_replay_buffer, axis_name="i")
