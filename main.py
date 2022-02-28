import pickle
import tempfile
import uuid
from copy import copy
from pathlib import Path

import absl.app
import absl.flags
import jax
import numpy as np
import tqdm
import wandb
import torch
from dm_env import specs
from flax import jax_utils

from common.dmc import make
from model import DoubleCritic, SamplerPolicy, TanhGaussianPolicy
from replay_buffer import ReplayBufferStorage, make_replay_loader
from sampler import RolloutStorage
from td3 import TD3
from utils import Timer, define_flags_with_default, get_user_flags, prefix_metrics

FLAGS_DEF = define_flags_with_default(
    env="walker_stand",
    obs_type="states",
    max_traj_length=1000,
    replay_buffer_size=2000001,
    seed=42,
    save_model=False,
    policy_arch="256-256",
    qf_arch="256-256",
    n_epochs=2000000,
    n_train_step_per_epoch=1,
    n_sample_step_per_epoch=1,
    eval_period=50000,
    eval_n_trajs=5,
    frame_stack=1,
    action_repeat=1,
    batch_size=256,
    save_replay_buffer=False,
    n_worker=4,
    td3=TD3.get_default_config(),
    online=False,
    cnn_features="32-64-128-256",
    cnn_strides="2-2-2-2",
    cnn_padding="SAME",
    latent_dim=50,
    downstream=False,
    replay_dir="/shared/hao/dataset/big",
    experiment_id="",
    checkpoint="",
    pin_memory=False,
    persistent_workers=True,
)


def main(argv):
    FLAGS = absl.flags.FLAGS
    variant = get_user_flags(FLAGS, FLAGS_DEF)

    if FLAGS.downstream:
        experiment_id = "-".join(["downstream", str(FLAGS.experiment_id), str(FLAGS.replay_dir).split("/")[-1], str(uuid.uuid4().hex)])
        replay_dir = Path(FLAGS.replay_dir)
    else:
        experiment_id = "-".join(["pretrain", str(FLAGS.experiment_id), str(uuid.uuid4().hex)])
        replay_dir = Path(FLAGS.replay_dir) / experiment_id
        replay_dir.mkdir(parents=True, exist_ok=True)

    wandb.init(
        config=copy(variant),
        project="TD3",
        dir=Path(tempfile.mkdtemp()),
        id=experiment_id,
        mode="online" if FLAGS.online else "offline",
        settings=wandb.Settings(start_method="thread"),
    )

    rng = jax.random.PRNGKey(FLAGS.seed)

    train_env = make(
        FLAGS.env,
        FLAGS.obs_type,
        FLAGS.frame_stack,
        FLAGS.action_repeat,
        rng,
        nchw=False,
    )
    test_env = make(
        FLAGS.env,
        FLAGS.obs_type,
        FLAGS.frame_stack,
        FLAGS.action_repeat,
        rng + 1000,
        nchw=False,
    )

    dummy_env = make(
        FLAGS.env,
        FLAGS.obs_type,
        FLAGS.frame_stack,
        FLAGS.action_repeat,
        rng,
        nchw=False,
    )
    action_dim = dummy_env.action_spec().shape[0]
    observation_dim = dummy_env.observation_spec().shape

    train_sampler = RolloutStorage(train_env, FLAGS.max_traj_length)
    eval_sampler = RolloutStorage(test_env, FLAGS.max_traj_length)
    data_specs = (
        train_env.observation_spec(),
        train_env.action_spec(),
        specs.Array((1,), np.float32, "reward"),
        specs.Array((1,), np.float32, "discount"),
    )
    phys_specs = (
        specs.Array((dummy_env._env.physics.state().shape[0],), np.float32, "physics"),
    )
    replay_storage = ReplayBufferStorage(data_specs, phys_specs, replay_dir / "replay", FLAGS.replay_buffer_size, FLAGS.save_replay_buffer)
    replay_loader = make_replay_loader(
        replay_storage,
        FLAGS.replay_buffer_size,
        FLAGS.batch_size * jax.local_device_count(),
        FLAGS.n_worker,
        FLAGS.save_replay_buffer,
        FLAGS.td3.nstep,
        FLAGS.td3.discount,
        FLAGS.downstream,
        dummy_env,
        replay_dir / "replay",
        FLAGS.pin_memory,
        FLAGS.persistent_workers,
    )

    def prepare_data(xs):
        local_device_count = jax.local_device_count()

        def _prepare(x):
            # x = x._numpy()  # for zero-copy conversion between TF and Numpy
            # Reshape data to shard to multiple devices.
            # [N, ...] -> [C, N // C, ...]
            return x.reshape((local_device_count, -1) + x.shape[1:])
        return jax.tree_map(_prepare, xs)

    replay_iter = None

    def get_replay_iter(replay_iter, replay_loader):
        if replay_iter is None:
            replay_iter = jax_utils.prefetch_to_device(map(prepare_data, replay_loader), 2)
        return replay_iter

    policy = TanhGaussianPolicy(
        action_dim,
        FLAGS.policy_arch,
        FLAGS.obs_type,
        FLAGS.td3.expl_noise,
        FLAGS.td3.policy_noise,
        FLAGS.td3.clip_noise,
        FLAGS.cnn_features,
        FLAGS.cnn_strides,
        FLAGS.cnn_padding,
        FLAGS.latent_dim,
    )
    qf = DoubleCritic(
        FLAGS.qf_arch,
        FLAGS.obs_type,
        FLAGS.cnn_features,
        FLAGS.cnn_strides,
        FLAGS.cnn_padding,
        FLAGS.latent_dim,
    )

    td3 = TD3()
    td3.update_default_config(FLAGS.td3)
    state, rng = td3.create_state(
        policy, qf, observation_dim, action_dim, rng, FLAGS.obs_type, FLAGS.downstream
    )
    sampler_policy = SamplerPolicy(
        policy, {"params": jax_utils.unreplicate(state)["policy"].params}
    )

    if not FLAGS.downstream:
        # wait until collecting max(n_worker, 10) trajectories for data loader
        _, rng = train_sampler.sample_traj(
            rng,
            sampler_policy.update_params(
                {"params": jax_utils.unreplicate(state)["policy"].params}
            ),
            max(FLAGS.n_worker, 10),
            deterministic=False,
            replay_storage=replay_storage,
            random=True,
        )

    for epoch in tqdm.tqdm(range(FLAGS.n_epochs)):
        metrics = {}
        with Timer() as rollout_timer:
            if not FLAGS.downstream:
                _, rng = train_sampler.sample_step(
                    rng,
                    sampler_policy.update_params(
                        {"params": jax_utils.unreplicate(state)["policy"].params}
                    ),
                    FLAGS.n_sample_step_per_epoch,
                    deterministic=False,
                    replay_storage=replay_storage,
                )

        with Timer() as train_timer:
            for batch_idx in range(FLAGS.n_train_step_per_epoch):
                replay_iter = get_replay_iter(replay_iter, replay_loader)
                batch = next(replay_iter)
                state, rng, train_metrics = td3.train(state, batch, rng)

        with Timer() as eval_timer:
            if epoch == 0 or (epoch + 1) % FLAGS.eval_period == 0:
                data, rng = eval_sampler.sample_traj(
                    rng,
                    sampler_policy.update_params(
                        {"params": jax_utils.unreplicate(state)["policy"].params}
                    ),
                    FLAGS.eval_n_trajs,
                    deterministic=True,
                )

                if FLAGS.save_model:
                    save_data = {"td3": td3, "variant": variant, "epoch": epoch}
                    with open(replay_dir / f"model_epoch_{epoch}.pkl", "wb") as fout:
                        pickle.dump(save_data, fout)

        if epoch == 0 or (epoch + 1) % FLAGS.eval_period == 0:
            metrics["average_return"] = data["r_traj"]
            metrics["env_steps"] = len(replay_storage)
            metrics["epoch"] = epoch
            metrics.update(prefix_metrics(train_metrics, "td3"))
            metrics["rollout_time"] = rollout_timer()
            metrics["train_time"] = train_timer()
            metrics["eval_time"] = eval_timer()
            metrics["epoch_time"] = rollout_timer() + train_timer() + eval_timer()
            wandb.log(metrics)

    wandb.finish()
    if FLAGS.save_model:
        save_data = {"td3": td3, "variant": variant, "epoch": epoch}
        with open(replay_dir / f"model_epoch_{epoch}.pkl", "wb") as fout:
            pickle.dump(save_data, fout)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    absl.app.run(main)
