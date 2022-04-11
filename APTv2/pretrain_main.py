if __name__ == "__main__":
    import os

    os.environ["MUJOCO_GL"] = "egl"

    import dataclasses
    import pickle
    import pprint
    import random
    import tempfile
    import uuid
    from copy import copy, deepcopy
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
    from absl import logging
    from dm_env import specs
    from flax import jax_utils
    from flax.training.train_state import TrainState
    from tqdm.auto import tqdm, trange

    torch.multiprocessing.set_start_method("spawn")

    from .data import UnlabelDataset
    from .environment import Environment
    from .model import ICM, Critic, Policy, SamplerPolicy
    from .utils import (WandBLogger, batched_random_crop,
                        define_flags_with_default, get_metrics, get_user_flags,
                        mse_loss, next_rng, prefix_metrics, set_random_seed,
                        update_target_network, value_and_multi_grad)

    FLAGS_DEF = define_flags_with_default(
        seed=42,
        n_total_iter=2000000,
        n_train_step_per_iter=1,
        n_sample_step_per_iter=1,
        batch_size=1024,
        dataloader_n_workers=16,
        test_freq=1e5,
        log_freq=1e3,
        save_model_freq=0,
        n_test_traj=5,
        max_traj_length=1000,
        logging=WandBLogger.get_default_config(),
        data=UnlabelDataset.get_default_config(),
        env=Environment.get_default_config(),
        policy=Policy.get_default_config(),
        critic=Critic.get_default_config(),
        icm=ICM.get_default_config(),
        online=False,
        log_all_worker=False,
        policy_lr=3e-4,
        critic_lr=3e-4,
        icm_lr=3e-4,
        soft_target_update_rate=1e-2,
        nstep=3,
        knn_k=3,
        knn_avg=True,
    )


def create_train_step(
    obs_type,
    observation_dim,
    action_dim,
    policy_lr,
    critic_lr,
    icm_lr,
    knn_k,
    knn_avg,
    soft_target_update_rate,
    model_keys,
):
    if obs_type != "states":
        preprocess = batched_random_crop
        copy_encoder = True
    else:
        preprocess = lambda _, x: x
        copy_encoder = False

    def dist_fn(x, y, ord):
        return jnp.sum(jnp.linalg.norm(x - y, ord))

    def distance_fn(a, b):
        return jax.vmap(jax.vmap(partial(dist_fn, ord=2), (None, 0)), (0, None))(a, b)

    def APTEnt(data):
        k = min(knn_k, data.shape[0])
        neg_distance = -distance_fn(data, data)
        neg_distance, _ = jax.lax.top_k(neg_distance, k)
        distance = -neg_distance
        if knn_avg:
            entropy = distance.reshape(-1, 1)  # (b * k, 1)
            entropy = entropy.reshape((data.shape[0], k))  # (b, k)
            entropy = entropy.mean(axis=1, keepdims=True)  # (b, 1)
        else:
            distance = jax.lax.sort(distance, dimension=-1)
            entropy = distance[:, -1].reshape(-1, 1)  # (b, 1)
        return entropy

    def create_state(policy, qf, icm):
        state = {}

        dummy_obs = jnp.zeros((10, *observation_dim))
        dummy_action = jnp.zeros((10, action_dim))

        policy_params = policy.init(next_rng(), next_rng(), dummy_obs)["params"]
        qf_params = qf.init(next_rng(), dummy_obs, dummy_action)["params"]
        icm_params = icm.init(next_rng(), dummy_obs, dummy_action, dummy_obs)["params"]

        state["policy"] = TrainState.create(
            params=policy_params,
            tx=optax.adam(policy_lr),
            apply_fn=policy.apply,
        )
        state["qf"] = TrainState.create(
            params=qf_params,
            tx=optax.adam(critic_lr),
            apply_fn=qf.apply,
        )
        target_qf_params = deepcopy({"qf": qf_params})
        state["icm"] = TrainState.create(
            params=icm_params,
            tx=optax.adam(icm_lr),
            apply_fn=icm.apply,
        )
        return state, target_qf_params

    def loss_fn(params, state, batch, rng, target_qf_params):

        obs = preprocess(rng, batch["obs"])
        action = batch["action"]
        discount = jnp.squeeze(batch["discount"], axis=1)
        next_obs = preprocess(rng, batch["next_obs"])

        _, embedding = state["icm"].apply_fn(
            {"params": params["icm"]}, obs, action, next_obs
        )
        reward = jnp.squeeze(APTEnt(embedding), axis=1)

        loss = {}

        """ Policy loss """
        rng, split_rng = jax.random.split(rng)
        new_action = state["policy"].apply_fn(
            {"params": params["policy"]},
            split_rng,
            obs,
            deterministic=True,
        )
        q_new_action, _ = state["qf"].apply_fn(
            {"params": params["qf"]}, obs, new_action
        )
        policy_loss = -q_new_action.mean()

        loss["policy"] = policy_loss

        """ Q function loss """
        q1_pred, q2_pred = state["qf"].apply_fn({"params": params["qf"]}, obs, action)

        rng, split_rng = jax.random.split(rng)
        new_next_action = state["policy"].apply_fn(
            {"params": params["policy"]},
            split_rng,
            next_obs,
            clip=True,
        )
        target_q1, target_q2 = state["qf"].apply_fn(
            {"params": target_qf_params["qf"]}, next_obs, new_next_action
        )
        target_q_values = jax.lax.min(target_q1, target_q2)

        q_target = jax.lax.stop_gradient(reward + discount * target_q_values)
        qf1_loss = mse_loss(q1_pred, q_target)
        qf2_loss = mse_loss(q2_pred, q_target)

        loss["qf"] = qf1_loss + qf2_loss

        """forward backward prediction loss"""
        icm_loss, _ = state["icm"].apply_fn(
            {"params": params["icm"]}, obs, action, next_obs
        )
        icm_loss = icm_loss.mean()
        loss["icm"] = icm_loss

        return tuple(loss[key] for key in model_keys), locals()

    @partial(jax.pmap, axis_name="batch", donate_argnums=(1, 2, 3))
    def wrapped(batch, state, rng, target_qf_params):
        rng, split_rng = jax.random.split(rng)
        params = {key: state[key].params for key in model_keys}
        (_, aux_values), grads = value_and_multi_grad(
            loss_fn, len(model_keys), has_aux=True
        )(params, state, batch, split_rng, target_qf_params)
        grads = jax.lax.pmean(grads, "batch")

        state = {
            key: state[key].apply_gradients(grads=grads[i][key])
            for i, key in enumerate(model_keys)
        }

        new_target_qf_params = {}
        new_target_qf_params["qf"] = update_target_network(
            state["qf"].params, target_qf_params["qf"], soft_target_update_rate
        )

        if copy_encoder:
            """
            Copy encoder from critic to actor
            """
            new_policy_params = state["policy"].params.copy(
                {"Encoder": state["qf"].params["Encoder"]}
            )
            state["policy"] = state["policy"].replace(params=new_policy_params)

        metrics = jax.lax.pmean(
            dict(
                policy_loss=aux_values["policy_loss"],
                qf1_loss=aux_values["qf1_loss"],
                qf2_loss=aux_values["qf2_loss"],
                icm_loss=aux_values["icm_loss"].mean(),
                average_qf1=aux_values["q1_pred"].mean(),
                average_qf2=aux_values["q2_pred"].mean(),
                average_target_q=aux_values["target_q_values"].mean(),
                train_reward=aux_values["reward"].mean(),
            ),
            axis_name="batch",
        )

        return state, metrics, rng, new_target_qf_params

    return create_state, wrapped


def create_sample_step(env, data_storage=None):
    reset_fn = jax.jit(jax.vmap(env.reset))
    keys = jax.random.split(jax.random.PRNGKey(42), jax.local_device_count())
    step_fn = jax.jit(env.step)

    def sample_fn(
        sample_state, sampler_policy, rng, step, deterministic=False, random=False
    ):
        @partial(jax.pmap, axis_name="pmap")
        def wrapped(sample_state, rng):
            def sample_step(carry):
                state, new_rng = carry
                new_rng, split_rng = jax.random.split(new_rng)
                action = sampler_policy(
                    split_rng, state.obs, deterministic=deterministic, random=random
                )
                next_state = step_fn(state, action)
                if data_storage is not None:
                    # data_storage.add(
                    #     state.obs,
                    #     action,
                    #     next_state.obs,
                    #     next_state.reward,
                    #     next_state.done,
                    #     next_state.qp,
                    # )
                    data_storage.add(
                        np.array(state.obs),
                        np.array(action),
                        np.array(next_state.obs),
                        np.array(next_state.reward),
                        np.array(next_state.done),
                        np.array(next_state.qp),
                    )
                return next_state, new_rng

            (sample_state, rng), _ = jax.lax.scan(
                lambda carry, _: (sample_step(carry), ()),
                ((sample_state, rng)),
                (),
                length=step,
            )

            return sample_state, rng

        rng = jax.device_put_sharded(
            next_rng(jax.local_device_count()), jax.local_devices()
        )
        return wrapped(sample_state, rng)

    return reset_fn(keys), sample_fn


def main(argv):
    FLAGS = absl.flags.FLAGS
    variant = get_user_flags(FLAGS, FLAGS_DEF)

    train_env, observation_dim, action_dim = Environment(
        FLAGS.env, FLAGS.max_traj_length
    ).setup()
    test_env, *_ = Environment(FLAGS.env, FLAGS.max_traj_length).setup()
    obs_type = FLAGS.env.obs_type
    data_dir = Path(FLAGS.logging.output_dir) / f"data_{str(uuid.uuid4().hex)}"
    data_dir.mkdir(parents=True, exist_ok=True)
    dataset = UnlabelDataset(
        FLAGS.data,
        data_dir,
        FLAGS.dataloader_n_workers,
        FLAGS.max_traj_length,
    )

    jax_devices = jax.local_devices()
    n_devices = len(jax_devices)
    assert FLAGS.batch_size % n_devices == 0

    logging.info(f"Data dir is {str(data_dir)}")
    variant["data_dir"] = str(data_dir)
    variant["jax_process_index"] = jax_process_index = jax.process_index()
    variant["jax_process_count"] = jax_process_count = jax.process_count()
    logger = WandBLogger(
        config=FLAGS.logging,
        variant=variant,
        enable=FLAGS.log_all_worker or (jax_process_index == 0),
    )
    logger.log(prefix_metrics(variant, "variant"))
    set_random_seed(FLAGS.seed * (jax_process_index + 1))

    train_sample_state, train_sample_fn = create_sample_step(
        train_env, dataset._storage
    )
    test_sample_state, test_sample_fn = create_sample_step(train_env, None)
    warmup_sample_state, warmup_sample_fn = create_sample_step(
        train_env, dataset._storage
    )
    warmup_sample_fn(
        warmup_sample_state,
        lambda *args, **kwargs: jax.random.uniform(
            next_rng(), (FLAGS.env.n_env, action_dim), minval=-1.0, maxval=1.0
        ),
        next_rng(),
        int(1e4),
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=FLAGS.batch_size,
        num_workers=FLAGS.dataloader_n_workers,
        prefetch_factor=2,
        persistent_workers=True,
        drop_last=True,
    )

    for _ in zip(
        trange(10, ncols=0, desc="Warming up dataloader"),
        data_loader,
    ):
        pass

    policy = Policy(FLAGS.policy, action_dim)
    critic = Critic(FLAGS.critic)
    icm = ICM(FLAGS.icm, action_dim)

    create_state, train_step_fn = create_train_step(
        obs_type,
        observation_dim,
        action_dim,
        FLAGS.policy_lr,
        FLAGS.critic_lr,
        FLAGS.icm_lr,
        FLAGS.knn_k,
        FLAGS.knn_avg,
        FLAGS.soft_target_update_rate,
        model_keys=tuple(["policy", "qf", "icm"]),
    )
    state, target_qf_params = create_state(policy, critic, icm)
    sampler_policy = SamplerPolicy(policy, {"params": state["policy"].params})

    train_sample_state = train_sample_fn(
        train_sample_state,
        sampler_policy.update_params({"params": state["policy"].params}),
        next_rng(),
        max(FLAGS.dataloader_n_workers, 10) * int(1e3),
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

    data_iter = iter(jax_utils.prefetch_to_device(generate_batch(data_loader), 2))

    sharded_rng = jax.device_put_sharded(next_rng(n_devices), jax_devices)

    @partial(jax.pmap, axis_name="pmap", donate_argnums=0)
    def sync_state_fn(state):
        i = jax.lax.axis_index("pmap")

        def select(x):
            return jax.lax.psum(jnp.where(i == 0, x, jnp.zeros_like(x)), "pmap")

        return jax.tree_map(select, state)

    state = jax_utils.replicate(state, jax_devices)
    target_qf_params = jax_utils.replicate(target_qf_params, jax_devices)
    state = sync_state_fn(state)

    for step in trange(FLAGS.n_total_iter, ncols=0):
        for _ in range(FLAGS.n_train_step_per_iter):
            batch = next(data_iter)
            state, train_metrics, sharded_rng, target_qf_params = train_step_fn(
                batch, state, sharded_rng, target_qf_params
            )

        train_sample_state = train_sample_fn(
            train_sample_state,
            sampler_policy.update_params({"params": state["policy"].params}),
            jax_utils.unreplicate(sharded_rng),
            FLAGS.n_sample_step_per_iter,
        )

        if step % FLAGS.log_freq == 0:
            log_metrics = get_metrics(train_metrics, unreplicate=True)
            log_metrics = prefix_metrics(log_metrics, "train")
            log_metrics.update({"step": step, "step": step})
            logger.log(log_metrics)
            tqdm.write("\n" + pprint.pformat(log_metrics) + "\n")

        if step % FLAGS.test_freq == 0:
            test_sample_state = test_sample_fn(
                test_sample_state,
                sampler_policy.update_params({"params": state["policy"].params}),
                jax_utils.unreplicate(sharded_rng),
                FLAGS.n_test_traj * FLAGS.max_traj_length,
                deterministic=True,
            )

            log_metrics = {}
            test_metrics = test_sample_state.info["eval_metrics"]
            test_metrics.completed_episodes.block_until_ready()
            log_metrics.update(test_metrics)
            log_metrics["env_steps"] = len(dataset._storage)
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
