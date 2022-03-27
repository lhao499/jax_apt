from functools import partial
from typing import Sequence, Tuple, Type

import distrax
import jax
import jax.numpy as jnp
from flax import linen as nn
from ml_collections import ConfigDict

from .utils import extend_and_repeat


class MLP(nn.Module):
    output_dim: int
    arch: str = "256-256"

    @nn.compact
    def __call__(self, input_tensor):
        x = input_tensor
        hidden_sizes = [int(h) for h in self.arch.split("-")]
        for idx, h in enumerate(hidden_sizes):
            x = nn.Dense(h)(x)
            x = nn.relu(x)
        output = nn.Dense(self.output_dim)(x)
        return output


class TwinMLP(nn.Module):
    arch: str = "256-256"
    num_qf: int = 2

    @nn.compact
    def __call__(self, observations, actions):
        x = jnp.concatenate([observations, actions], -1)
        vmap_fc = nn.vmap(
            MLP,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.num_qf,
        )
        qs = vmap_fc(output_dim=1, arch=self.arch)(x)
        return qs


class Critic(nn.Module):
    config_updates: ... = None
    cnn_features: str = "32-32-32-32"
    cnn_strides: str = "2-1-1-1"
    cnn_padding: str = "VALID"

    @staticmethod
    @nn.nowrap
    def get_default_config(updates=None):
        config = ConfigDict()
        config.arch = "256-256"
        config.obs_type = "states"
        config.latent_dim = 50

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def setup(self):
        self.config = self.get_default_config(self.config_updates)

        if self.config.obs_type == "states":
            self.encoder = Identity()
            self.projection = Identity()
        else:
            self.encoder = Encoder(
                tuple(map(int, self.cnn_features.split("-"))),
                tuple(map(int, self.cnn_strides.split("-"))),
                self.cnn_padding,
            )
            self.projection = Projection(self.config.latent_dim)

        self.fc = TwinMLP(arch=self.config.arch)

    def __call__(self, observations, actions):
        x = self.encoder(observations)
        x = self.projection(x)
        x = self.fc(x, actions)
        return jnp.squeeze(x, -1)


class Policy(nn.Module):
    config_updates: ... = None
    action_dim: int = None
    cnn_features: str = "32-32-32-32"
    cnn_strides: str = "2-1-1-1"
    cnn_padding: str = "VALID"

    @staticmethod
    @nn.nowrap
    def get_default_config(updates=None):
        config = ConfigDict()
        config.arch = "256-256"
        config.obs_type = "states"
        config.latent_dim = 50

        config.expl_noise = 0.1
        config.policy_noise = 0.2
        config.clip_noise = 0.5

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def setup(self):
        self.config = self.get_default_config(self.config_updates)

        if self.config.obs_type == "states":
            self.encoder = Identity()
            self.projection = Identity()
        else:
            self.encoder = Encoder(
                tuple(map(int, self.cnn_features.split("-"))),
                tuple(map(int, self.cnn_strides.split("-"))),
                self.cnn_padding,
            )
            self.projection = Projection(self.config.latent_dim)

        self.fc = MLP(output_dim=self.action_dim, arch=self.config.arch)

    def __call__(self, rng, observations, deterministic=False, clip=False):
        x = self.encoder(observations)
        x = jax.lax.stop_gradient(x)
        x = self.projection(x)
        actions = self.fc(x)
        actions = jnp.tanh(actions)  # first constraint the range of action
        if deterministic:
            return actions
        if clip:
            noise = (
                jax.random.normal(rng, shape=(self.action_dim,))
                * self.config.policy_noise
            )
            noise = noise.clip(-self.config.clip_noise, self.config.clip_noise)
            actions = actions + noise
            actions = jnp.tanh(actions)
            return actions
        else:
            noise = (
                jax.random.normal(rng, shape=(self.action_dim,))
                * self.config.expl_noise
            )
            actions = actions + noise
            actions = jnp.tanh(actions)
            return actions


class ICM(nn.Module):
    config_updates: ... = None
    action_dim: int = None
    cnn_features: str = "32-32-32-32"
    cnn_strides: str = "2-1-1-1"
    cnn_padding: str = "VALID"

    @staticmethod
    @nn.nowrap
    def get_default_config(updates=None):
        config = ConfigDict()
        config.arch = "256-256"
        config.obs_type = "states"
        config.icm_dim = 256

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def setup(self):
        self.config = self.get_default_config(self.config_updates)

        if self.config.obs_type == "states":
            self.encoder = Identity()
            self.projection = Identity()
        else:
            self.encoder = Encoder(
                tuple(map(int, self.cnn_features.split("-"))),
                tuple(map(int, self.cnn_strides.split("-"))),
                self.cnn_padding,
            )
            self.projection = Projection(self.config.latent_dim)

        self.trunk = [
            nn.Dense(self.config.icm_dim),
            nn.LayerNorm(self.config.icm_dim),
        ]

        self.forward_net = [MLP(output_dim=self.config.icm_dim, arch=self.config.arch)]

        self.backward_net = [
            MLP(output_dim=self.action_dim, arch=self.config.arch),
            lambda x: nn.tanh(x),
        ]

    def __call__(self, obs, action, next_obs):
        obs = self.encoder(obs)
        next_obs = self.encoder(next_obs)
        for layer in self.trunk:
            obs = layer(obs)
        for layer in self.trunk:
            next_obs = layer(next_obs)
        next_obs_hat = jnp.concatenate([obs, action], axis=-1)
        for layer in self.forward_net:
            next_obs_hat = layer(next_obs_hat)
        action_hat = jnp.concatenate([obs, next_obs], axis=-1)
        for layer in self.backward_net:
            action_hat = layer(action_hat)
        forward_error = jnp.linalg.norm(
            next_obs - next_obs_hat, axis=-1, ord=2, keepdims=True
        )
        backward_error = jnp.linalg.norm(
            action - action_hat, axis=-1, ord=2, keepdims=True
        )
        prediction_error = forward_error + backward_error
        representation = jax.lax.stop_gradient(next_obs_hat)
        return prediction_error, representation


class SamplerPolicy(object):
    def __init__(self, policy, params):
        self.policy = policy
        self.params = params

    def update_params(self, params):
        self.params = params
        return self

    @partial(jax.jit, static_argnames=("self", "deterministic"))
    def act(self, params, rng, observations, deterministic):
        return self.policy.apply(params, rng, observations, deterministic, clip=True)

    def __call__(self, rng, observations, deterministic=False, random=False):
        observations = jax.device_put(observations)
        if random:
            actions = jax.random.uniform(
                rng, (1, self.policy.action_dim), minval=-1, maxval=1.0
            )
        else:
            actions = self.act(self.params, rng, observations, deterministic)
            actions = jnp.tanh(actions)

        assert jnp.all(jnp.isfinite(actions))
        return jax.device_get(actions)


class Encoder(nn.Module):
    features: Sequence[int] = (32, 32, 32, 32)
    strides: Sequence[int] = (2, 1, 1, 1)
    padding: str = "VALID"

    @nn.compact
    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        assert len(self.features) == len(self.strides)

        x = observations.astype(jnp.float32) / 255.0
        for features, stride in zip(self.features, self.strides):
            x = nn.Conv(
                features,
                kernel_size=(3, 3),
                strides=(stride, stride),
                kernel_init=jax.nn.initializers.orthogonal(1e-2),
                padding=self.padding,
            )(x)
            x = nn.relu(x)

        if len(x.shape) == 4:
            x = x.reshape([x.shape[0], -1])
        else:
            x = x.reshape([-1])
        return x


class Identity(nn.Module):
    @nn.compact
    def __call__(self, x):
        return x


class Projection(nn.Module):
    latent_dim: int = 50

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.latent_dim)(x)
        x = nn.LayerNorm()(x)
        x = nn.tanh(x)
        return x


def update_target_network(main_params, target_params, tau):
    return jax.tree_multimap(
        lambda x, y: tau * x + (1.0 - tau) * y, main_params, target_params
    )


def multiple_action_q_function(forward):
    # Forward the q function with multiple actions on each state, to be used as a decorator
    def wrapped(self, observations, actions, **kwargs):
        multiple_actions = False
        batch_size = observations.shape[0]
        if actions.ndim == 3 and observations.ndim == 2:
            multiple_actions = True
            observations = extend_and_repeat(observations, 1, actions.shape[1]).reshape(
                -1, observations.shape[-1]
            )
            actions = actions.reshape(-1, actions.shape[-1])
        q_values = forward(self, observations, actions, **kwargs)
        if multiple_actions:
            q_values = q_values.reshape(batch_size, -1)
        return q_values

    return wrapped
