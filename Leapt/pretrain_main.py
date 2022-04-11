import os
import uuid

import jax
from absl import app, flags
from brax import envs
from brax.io import html, metrics, model

from .sac import train_sac
from .leapt import train_leapt

FLAGS = flags.FLAGS

flags.DEFINE_enum("learner", "sac", ["sac", "leapt"], "Which algorithm to run.")
flags.DEFINE_string("env", "ant", "Name of environment to train.")
flags.DEFINE_integer(
    "total_env_steps", 50000000, "Number of env steps to run training for."
)
flags.DEFINE_integer("eval_frequency", 10, "How many times to run an eval.")
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_integer("num_envs", 4, "Number of envs to run in parallel.")
flags.DEFINE_integer("action_repeat", 1, "Action repeat.")
flags.DEFINE_integer("unroll_length", 30, "Unroll length.")
flags.DEFINE_integer("batch_size", 4, "Batch size.")
flags.DEFINE_integer("num_minibatches", 1, "Number")
flags.DEFINE_integer(
    "num_update_epochs",
    1,
    "Number of times to reuse each transition for gradient " "computation.",
)
flags.DEFINE_float("reward_scaling", 10.0, "Reward scale.")
flags.DEFINE_float("entropy_cost", 3e-4, "Entropy cost.")
flags.DEFINE_integer("episode_length", 1000, "Episode length.")
flags.DEFINE_float("discounting", 0.99, "Discounting.")
flags.DEFINE_float("learning_rate", 5e-4, "Learning rate.")
flags.DEFINE_float("max_gradient_norm", 1e9, "Maximal norm of a gradient update.")
flags.DEFINE_string("logdir", "", "Logdir.")
flags.DEFINE_bool(
    "normalize_observations", True, "Whether to apply observation normalization."
)
flags.DEFINE_integer(
    "max_devices_per_host",
    None,
    "Maximum number of devices to use per host. If None, "
    "defaults to use as much as it can.",
)
flags.DEFINE_integer("num_videos", 1, "Number of videos to record after training.")

# SAC hps.
flags.DEFINE_integer(
    "min_replay_size", 8192, "Minimal replay buffer size before the training starts."
)
flags.DEFINE_integer("max_replay_size", 1048576, "Maximal replay buffer size.")
flags.DEFINE_float(
    "grad_updates_per_step",
    1.0,
    "How many SAC gradient updates to run per one step in the " "environment.",
)


def main(unused_argv):

    env_fn = envs.create_fn(FLAGS.env)

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
                max_devices_per_host=FLAGS.max_devices_per_host,
                seed=FLAGS.seed,
                reward_scaling=FLAGS.reward_scaling,
                grad_updates_per_step=FLAGS.grad_updates_per_step,
                episode_length=FLAGS.episode_length,
                progress_fn=writer.write_scalars,
            )
        if FLAGS.learner == 'leapt':
          inference_fn, params, _ = train_leapt(
              environment_fn=env_fn,
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
              max_devices_per_host=FLAGS.max_devices_per_host,
              seed=FLAGS.seed,
              reward_scaling=FLAGS.reward_scaling,
              grad_updates_per_step=FLAGS.grad_updates_per_step,
              episode_length=FLAGS.episode_length,
              progress_fn=writer.write_scalars)

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
