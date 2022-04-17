if __name__ == "__main__":
    import os
    import uuid

    import jax
    from absl import app, flags
    from brax import envs
    from brax.io import html, metrics, model

    from .leapt import train_leapt
    from .sac import train_sac
    from .utils import WandBLogger, define_flags_with_default, get_user_flags, prefix_metrics
    from .data import Dataset, EmptyDataset, LoadDataset
    from pathlib import Path
    from flax import jax_utils
    import numpy as np
    import logging


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
        dataset=Dataset.get_default_config(),
        log_all_worker=False,
        knn_avg=True,
        knn_k=64,
        use_apt_to_play=False,
        save_every_play=False,
        save_last_play=False,
        load_data_dir="",
        use_reward_to_adapt=False,
        sample_chuck_size=0,
        adapt_updates_per_step=256,
    )


def main(unused_argv):

    FLAGS = flags.FLAGS
    variant = get_user_flags(FLAGS, FLAGS_DEF)

    if FLAGS.load_data_dir == "":
        data_dir = Path(f"/tmp/{uuid.uuid4().hex}")
    else:
        data_dir = Path(FLAGS.load_data_dir)

    if FLAGS.learner == "leapt":
        num_updates = int(FLAGS.num_envs * FLAGS.grad_updates_per_step)
        num_sample_total = num_updates * FLAGS.batch_size
        data_n_worker = FLAGS.dataset.n_worker
        if FLAGS.sample_chuck_size == 0:
            sample_chuck_size = FLAGS.num_envs
        else:
            sample_chuck_size = FLAGS.sample_chuck_size
        data_batch_size = int(num_sample_total / sample_chuck_size)

        if FLAGS.use_reward_to_adapt:
            dataset = LoadDataset(FLAGS.dataset, data_dir, sample_chuck_size)
        elif FLAGS.use_apt_to_play:
            dataset = EmptyDataset(FLAGS.dataset, data_dir, sample_chuck_size)

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=data_batch_size,
            num_workers=data_n_worker,
            prefetch_factor=2,
            persistent_workers=True,
            drop_last=True,
        )

        data_storage = dataset._storage

        def generate_batch(it):
            def prepare_data(xs):
                def _prepare(x):
                    return x.reshape((jax.local_device_count(), num_updates, FLAGS.batch_size,  x.shape[-1]))

                return jax.tree_map(_prepare, xs)

            while True:
                for batch in it:
                    batch = np.asarray(batch)
                    yield prepare_data(batch)

        data_iter = iter(jax_utils.prefetch_to_device(generate_batch(data_loader), 2))
    else:
        data_storage = None
        data_iter = None
        data_n_worker = 1

    jax_devices = jax.local_devices()
    n_devices = len(jax_devices)
    assert FLAGS.batch_size % n_devices == 0

    variant["jax_process_index"] = jax.process_index()
    variant["jax_process_count"] = jax.process_count()
    variant["data_dir"] = str(data_dir)
    logging.info(f"data dir is {str(data_dir)}")
    logger = WandBLogger(
        config=FLAGS.logging,
        variant=variant,
        enable=FLAGS.log_all_worker or (jax.process_index() == 0),
    )
    logger.log(prefix_metrics(variant, "variant"))

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
                knn_avg=FLAGS.knn_avg,
                knn_k=FLAGS.knn_k,
                use_apt_to_play=FLAGS.use_apt_to_play,
                use_reward_to_adapt=FLAGS.use_reward_to_adapt,
                adapt_updates_per_step=FLAGS.adapt_updates_per_step,
                save_every_play=FLAGS.save_every_play,
                save_last_play=FLAGS.save_last_play,
                data_storage=data_storage,
                data_iter=data_iter,
                data_n_worker=data_n_worker,
                data_dir=data_dir,
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
    import torch
    torch.multiprocessing.set_start_method("spawn")
    app.run(main)
