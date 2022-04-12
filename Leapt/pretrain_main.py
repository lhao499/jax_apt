import os
import uuid

import jax
from absl import app, flags
from brax import envs
from brax.io import html, metrics, model

from .leapt import train_leapt
from .sac import train_sac
from .utils import (WandBLogger, define_flags_with_default, get_user_flags,
                    prefix_metrics)

FLAGS_DEF = define_flags_with_default(
    learner="sac",
    env="ant",
    total_env_steps=50000000,
    eval_frequency=10,
    seed=0,
    num_envs=4,
    action_repeat=1,
    unroll_length=30,
    batch_size=4,
    num_minibatches=1,
    num_update_epochs=1,
    reward_scaling=10.0,
    entropy_cost=3e-4,
    episode_length=1000,
    discounting=0.99,
    learning_rate=5e-4,
    max_gradient_norm=1e9,
    logdir="",
    normalize_observations=True,
    num_videos=1,
    min_replay_size=8192,
    max_replay_size=1048576,
    grad_updates_per_step=1.0,
    logging=WandBLogger.get_default_config(),
    log_all_worker=False,
)


def main(unused_argv):

    FLAGS = flags.FLAGS
    variant = get_user_flags(FLAGS, FLAGS_DEF)

    env_fn = envs.create_fn(FLAGS.env)

    jax_devices = jax.local_devices()
    n_devices = len(jax_devices)
    assert FLAGS.batch_size % n_devices == 0

    variant["jax_process_index"] = jax.process_index()
    variant["jax_process_count"] = jax.process_count()
    logger = WandBLogger(
        config=FLAGS.logging,
        variant=variant,
        enable=FLAGS.log_all_worker or (jax.process_index() == 0),
    )
    logger.log(prefix_metrics(variant, "variant"))

    with metrics.Writer(FLAGS.logdir) as writer:
        writer.write_hparams(
            {
                "log_frequency": FLAGS.eval_frequency,
                "num_envs": FLAGS.num_envs,
                "total_env_steps": FLAGS.total_env_steps,
            }
        )
        if FLAGS.learner == "sac":
            inference_fn, params, _ = train_sac(
                environment_fn=env_fn,
                logger=logger,
                num_envs=FLAGS.num_envs,
                action_repeat=FLAGS.action_repeat,
                normalize_observations=FLAGS.normalize_observations,
                num_timesteps=FLAGS.total_env_steps,
                log_frequency=FLAGS.eval_frequency,
                batch_size=FLAGS.batch_size,
                min_replay_size=FLAGS.min_replay_size,
                max_replay_size=FLAGS.max_replay_size,
                learning_rate=FLAGS.learning_rate,
                discounting=FLAGS.discounting,
                seed=FLAGS.seed,
                reward_scaling=FLAGS.reward_scaling,
                grad_updates_per_step=FLAGS.grad_updates_per_step,
                episode_length=FLAGS.episode_length,
                progress_fn=writer.write_scalars,
            )
        if FLAGS.learner == "leapt":
            inference_fn, params, _ = train_leapt(
                environment_fn=env_fn,
                logger=logger,
                num_envs=FLAGS.num_envs,
                action_repeat=FLAGS.action_repeat,
                normalize_observations=FLAGS.normalize_observations,
                num_timesteps=FLAGS.total_env_steps,
                log_frequency=FLAGS.eval_frequency,
                batch_size=FLAGS.batch_size,
                min_replay_size=FLAGS.min_replay_size,
                max_replay_size=FLAGS.max_replay_size,
                learning_rate=FLAGS.learning_rate,
                discounting=FLAGS.discounting,
                seed=FLAGS.seed,
                reward_scaling=FLAGS.reward_scaling,
                grad_updates_per_step=FLAGS.grad_updates_per_step,
                episode_length=FLAGS.episode_length,
                progress_fn=writer.write_scalars,
            )

    # Save to flax serialized checkpoint.
    filename = f"{FLAGS.env}_{FLAGS.learner}.pkl"
    path = os.path.join(FLAGS.logdir, filename)
    model.save_params(path, params)

    # Output an episode trajectory.
    env = env_fn()

    @jax.jit
    def jit_next_state(state, key):
        new_key, tmp_key = jax.random.split(key)
        act = inference_fn(params, state.obs, tmp_key)
        return env.step(state, act), new_key

    for i in range(FLAGS.num_videos):
        rng = jax.random.PRNGKey(FLAGS.seed + i)
        rng, env_key = jax.random.split(rng)
        state = env.reset(env_key)
        qps = []
        while not state.done:
            qps.append(state.qp)
            state, rng = jit_next_state(state, rng)

        html_path = f"{FLAGS.logdir}/trajectory_{uuid.uuid4()}.html"
        html.save_html(html_path, env.sys, qps)


if __name__ == "__main__":
    app.run(main)
