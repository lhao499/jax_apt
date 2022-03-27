if __name__ == "__main__":
    import os

    os.environ["MUJOCO_GL"] = "egl"

    import dataclasses
    import pickle
    import pprint
    import random
    import tempfile
    import uuid
    from copy import copy
    from functools import partial
    from pathlib import Path

    import absl.app
    import absl.flags
    import flax
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    import torch
    import wandb
    from dm_env import specs
    from flax import jax_utils
    from tqdm.auto import tqdm, trange

    torch.multiprocessing.set_start_method("spawn")

    from .environment import Environment
    from .model import ICM, Critic, Policy, SamplerPolicy
    from .replay_buffer import TestDataset, TrainDataset
    from .sampler import RolloutStorage
    from .td3 import TD3
    from .utils import (Timer, VideoRecorder, WandBLogger,
                        define_flags_with_default, get_metrics, get_user_flags,
                        next_rng, prefix_metrics, set_random_seed)

    FLAGS_DEF = define_flags_with_default(
        seed=42,
        n_total_step=2000000,
        n_train_step_per_iter=1,
        n_sample_step_per_iter=1,
        batch_size=256,
        dataloader_n_workers=16,
        test_freq=1e5,
        log_freq=1e3,
        save_model_freq=0,
        n_test_traj=5,
        max_traj_length=1000,
        td3=TD3.get_default_config(),
        logging=WandBLogger.get_default_config(),
        replay=TrainDataset.get_default_config(),
        env=Environment.get_default_config(),
        policy=Policy.get_default_config(),
        critic=Critic.get_default_config(),
        icm=ICM.get_default_config(),
        online=False,
        log_all_worker=False,
        replay_dir="/shared/hao/dataset/APTv2",
    )


def main(argv):
    FLAGS = absl.flags.FLAGS
    variant = get_user_flags(FLAGS, FLAGS_DEF)

    train_env = Environment(FLAGS.env).environment
    test_env = Environment(FLAGS.env).environment
    action_dim = Environment(FLAGS.env).action_dim
    observation_dim = Environment(FLAGS.env).observation_dim
    obs_type = FLAGS.env.obs_type
    data_specs = (
        train_env.observation_spec(),
        train_env.action_spec(),
        specs.Array((1,), np.float32, "reward"),
        specs.Array((1,), np.float32, "discount"),
    )
    phys_specs = (
        specs.Array((train_env._env.physics.state().shape[0],), np.float32, "physics"),
    )
    replay_dir = Path(FLAGS.replay_dir) / str(uuid.uuid4().hex)
    replay_dir.mkdir(parents=True, exist_ok=True)
    replay = TrainDataset(
        FLAGS.replay, data_specs, phys_specs, replay_dir, FLAGS.dataloader_n_workers
    )

    def collect_random_data(replay, env):
        replay_storage = replay._storage
        done = True
        for _ in trange(
            FLAGS.dataloader_n_workers * int(1e3),
            ncols=0,
            desc="Collecting random data",
        ):
            if done:
                time_step = env.reset()
                replay_storage.add(time_step, dict(physics=env._env.physics.state()))
            a = np.random.uniform(size=(action_dim,), low=-1, high=1.0).astype(
                np.float32
            )
            time_step = env.step(a)
            done = time_step.last()
            replay_storage.add(time_step, dict(physics=env._env.physics.state()))

    collect_random_data(replay, train_env)

    replay_loader = torch.utils.data.DataLoader(
        replay,
        batch_size=FLAGS.batch_size,
        num_workers=FLAGS.dataloader_n_workers,
        prefetch_factor=2,
        persistent_workers=True,
        drop_last=True,
    )

    for _ in zip(
        trange(5, ncols=0, desc="Warming up dataloader"),
        replay_loader,
    ):
        pass

    jax_devices = jax.local_devices()
    n_devices = len(jax_devices)
    assert FLAGS.batch_size % n_devices == 0

    variant["jax_process_index"] = jax_process_index = jax.process_index()
    variant["jax_process_count"] = jax_process_count = jax.process_count()
    variant["replay_dir"] = str(replay_dir)

    logger = WandBLogger(
        config=FLAGS.logging,
        variant=variant,
        enable=FLAGS.log_all_worker or (jax_process_index == 0),
    )
    logger.log(prefix_metrics(variant, "variant"))
    set_random_seed(FLAGS.seed * (jax_process_index + 1))

    video_recoder = VideoRecorder(root_dir=replay_dir, is_train=False)

    train_sampler = RolloutStorage(train_env, FLAGS.max_traj_length, None)
    test_sampler = RolloutStorage(test_env, FLAGS.max_traj_length, video_recoder)

    policy = Policy(FLAGS.policy, action_dim)
    critic = Critic(FLAGS.critic)
    icm = ICM(FLAGS.icm, action_dim)

    td3 = TD3()
    td3.update_default_config(FLAGS.td3)
    state = td3.create_state(
        policy,
        critic,
        icm,
        observation_dim,
        action_dim,
        obs_type,
    )
    sampler_policy = SamplerPolicy(policy, {"params": state["policy"].params})

    train_sampler.sample_traj(
        next_rng(),
        sampler_policy.update_params({"params": state["policy"].params}),
        max(FLAGS.dataloader_n_workers, 10),
        deterministic=False,
        replay_storage=replay._storage,
        random=True,
    )

    def generate_batch(it):

        def prepare_data(xs):
            def _prepare(x):
                return x.reshape((jax.local_device_count(), -1) + x.shape[1:])
            return jax.tree_map(_prepare, xs)

        while True:
            for batch in it:
                batch = {k: v.numpy() for k, v in batch.items()}
                yield prepare_data(batch)

    replay_iter = iter(jax_utils.prefetch_to_device(generate_batch(replay_loader), 2))

    sharded_rng = jax.device_put_sharded(next_rng(n_devices), jax_devices)

    @partial(jax.pmap, axis_name="pmap", donate_argnums=0)
    def sync_state_fn(state):
        i = jax.lax.axis_index("pmap")

        def select(x):
            return jax.lax.psum(jnp.where(i == 0, x, jnp.zeros_like(x)), "pmap")

        return jax.tree_map(select, state)

    state = jax_utils.replicate(state, jax_devices)
    state = sync_state_fn(state)

    for step in trange(FLAGS.n_total_step, ncols=0):
        for _ in range(FLAGS.n_train_step_per_iter):
            batch = next(replay_iter)
            state, sharded_rng, train_metrics = td3.train(state, batch, sharded_rng)

        train_sampler.sample_step(
            jax_utils.unreplicate(sharded_rng),
            sampler_policy.update_params(
                {"params": jax_utils.unreplicate(state)["policy"].params}
            ),
            FLAGS.n_sample_step_per_iter,
            deterministic=False,
            replay_storage=replay._storage,
        )

        if step % FLAGS.log_freq == 0:
            log_metrics = get_metrics(train_metrics, unreplicate=True)
            log_metrics = prefix_metrics(log_metrics, "train")
            log_metrics.update({"step": step, "step": step})
            logger.log(log_metrics)
            tqdm.write("\n" + pprint.pformat(log_metrics) + "\n")

        if step % FLAGS.test_freq == 0:
            data, _ = test_sampler.sample_traj(
                jax_utils.unreplicate(sharded_rng),
                sampler_policy.update_params(
                    {"params": jax_utils.unreplicate(state)["policy"].params}
                ),
                FLAGS.n_test_traj,
                deterministic=True,
            )

            video_recoder.log_to_wandb()
            log_metrics["average_return"] = data["r_traj"]
            log_metrics["env_steps"] = len(replay._storage)
            log_metrics = prefix_metrics(log_metrics, "test")
            log_metrics["step"] = step
            logger.log(log_metrics)

        if FLAGS.save_model_freq > 0 and step % FLAGS.save_model_freq == 0:
            state = sync_state_fn(state)

            save_data = {
                "step": step,
                "step": step,
                "variant": variant,
                "state": jax.device_get(jax_utils.unreplicate(state)),
            }
            if jax_process_index == 0:
                logger.save_pickle(save_data, "model.pkl")


if __name__ == "__main__":
    absl.app.run(main)
