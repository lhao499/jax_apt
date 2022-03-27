from collections import namedtuple
from copy import deepcopy
from dataclasses import dataclass
from functools import partial

import flax
import jax
import jax.numpy as jnp
import optax
from flax import jax_utils
from flax.training.train_state import TrainState
from ml_collections import ConfigDict

from .model import update_target_network
from .utils import (batched_random_crop, mse_loss, next_rng,
                    nonparametric_entropy, value_and_multi_grad)


@dataclass
class TD3:
    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.discount = 0.99
        config.policy_lr = 3e-4
        config.qf_lr = 3e-4
        config.icm_lr = 3e-4
        config.optimizer_type = "adam"
        config.soft_target_update_rate = 5e-3
        config.nstep = 3
        config.knn_k = 3
        config.knn_avg = True
        config.knn_log = False
        config.knn_clip = 0.0
        config.knn_pow = 2

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())

        return config

    def update_default_config(self, updates):
        self.config = self.get_default_config(updates)

    def create_state(self, policy, qf, icm, observation_dim, action_dim, obs_type):
        state = {}

        optimizer_class = {
            "adam": optax.adam,
            "sgd": optax.sgd,
        }[self.config.optimizer_type]

        self._obs_type = obs_type
        dummy_obs = jnp.zeros((10, *observation_dim))

        policy_params = policy.init(next_rng(), next_rng(), dummy_obs)["params"]
        state["policy"] = TrainState.create(
            params=policy_params,
            tx=optimizer_class(self.config.policy_lr),
            apply_fn=policy.apply,
        )

        qf_params = qf.init(next_rng(), dummy_obs, jnp.zeros((10, action_dim)))[
            "params"
        ]
        state["qf"] = TrainState.create(
            params=qf_params,
            tx=optimizer_class(self.config.qf_lr),
            apply_fn=qf.apply,
        )
        self._target_qf_params = deepcopy({"qf": qf_params})
        self._target_qf_params = jax_utils.replicate(self._target_qf_params)

        icm_params = icm.init(
            next_rng(), dummy_obs, jnp.zeros((10, action_dim)), dummy_obs
        )["params"]
        state["icm"] = TrainState.create(
            params=icm_params,
            tx=optimizer_class(self.config.icm_lr),
            apply_fn=icm.apply,
        )

        self._model_keys = tuple(["policy", "qf", "icm"])

        self._copy_encoder = False if self._obs_type == "states" else True

        return state

    def train(self, state, batch, rng):
        state, metrics, rng, self._target_qf_params = train_step(
            state,
            rng,
            batch,
            self._target_qf_params,
            self._model_keys,
            self._copy_encoder,
            self._obs_type,
            self.config.soft_target_update_rate,
            self.config.knn_k,
            self.config.knn_avg,
            self.config.knn_log,
            self.config.knn_clip,
            self.config.knn_pow,
        )
        return state, rng, metrics


@partial(
    jax.pmap,
    static_broadcasted_argnums=list(range(4, 13)),
    axis_name="batch",
    donate_argnums=(0, 1, 3),
)
def train_step(
    state,
    rng,
    batch,
    target_qf_params,
    model_keys,
    copy_encoder,
    obs_type,
    soft_target_update_rate,
    knn_k,
    knn_avg,
    knn_log,
    knn_clip,
    knn_pow,
):
    def loss_fn(params, rng):
        randaug = batched_random_crop if obs_type != "states" else lambda _, x: x
        obs = randaug(rng, batch["obs"])
        action = batch["action"]
        discount = jnp.squeeze(batch["discount"], axis=1)
        next_obs = randaug(rng, batch["next_obs"])

        _, representation = state["icm"].apply_fn(
            {"params": params["icm"]}, obs, action, next_obs
        )
        reward = jnp.squeeze(
            nonparametric_entropy(
                representation, knn_k, knn_avg, knn_log, knn_clip, knn_pow
            ),
            axis=1,
        )

        loss = {}

        rng, split_rng = jax.random.split(rng)
        new_action = state["policy"].apply_fn(
            {"params": params["policy"]},
            split_rng,
            obs,
            deterministic=True,
        )

        """ Policy loss """
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

    rng, split_rng = jax.random.split(rng)
    params = {key: state[key].params for key in model_keys}
    (_, aux_values), grads = value_and_multi_grad(
        loss_fn, len(model_keys), has_aux=True
    )(params, split_rng)
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