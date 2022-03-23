from functools import partial
from typing import Sequence, Tuple, Type

import distrax
import jax
import jax.numpy as jnp
from flax import linen as nn

from jax_utils import extend_and_repeat


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


class FullyConnectedNetwork(nn.Module):
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


class DoubleCriticFC(nn.Module):
    arch: str = "256-256"
    num_qf: int = 2

    @nn.compact
    def __call__(self, observations, actions):
        x = jnp.concatenate([observations, actions], -1)
        vmap_fc = nn.vmap(
            FullyConnectedNetwork,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.num_qf,
        )
        qs = vmap_fc(output_dim=1, arch=self.arch)(x)
        return qs


class DoubleCritic(nn.Module):
    arch: str = "256-256"
    obs_type: str = "states"
    cnn_features: str = "32-32-32-32"
    cnn_strides: str = "2-1-1-1"
    cnn_padding: str = "VALID"
    latent_dim: int = 50

    def setup(self):
        if self.obs_type == "states":
            self.encoder = Identity(name="Encoder")
            self.projection = Identity()
        elif self.obs_type == "pixels":
            self.encoder = Encoder(
                tuple(map(int, self.cnn_features.split("-"))),
                tuple(map(int, self.cnn_strides.split("-"))),
                self.cnn_padding,
                name="Encoder",
            )
            self.projection = Projection(self.latent_dim)
        else:
            raise NotImplementedError
        self.fc = DoubleCriticFC(arch=self.arch)

    def __call__(self, observations, actions):
        x = self.encoder(observations)
        x = self.projection(x)
        x = self.fc(x, actions)
        return jnp.squeeze(x, -1)


class TanhGaussianPolicy(nn.Module):
    action_dim: int
    arch: str = "256-256"
    obs_type: str = "states"
    expl_noise: float = 0.1
    policy_noise: float = 0.2
    clip_noise: float = 0.5
    cnn_features: Sequence[int] = (32, 32, 32, 32)
    cnn_strides: Sequence[int] = (2, 1, 1, 1)
    cnn_padding: str = "VALID"
    latent_dim: int = 50

    def setup(self):
        if self.obs_type == "states":
            self.encoder = Identity(name="Encoder")
            self.projection = Identity()
        elif self.obs_type == "pixels":
            self.encoder = Encoder(
                tuple(map(int, self.cnn_features.split("-"))),
                tuple(map(int, self.cnn_strides.split("-"))),
                self.cnn_padding,
                name="Encoder",
            )
            self.projection = Projection(self.latent_dim)
        else:
            raise NotImplementedError

        self.fc = FullyConnectedNetwork(output_dim=self.action_dim, arch=self.arch)

    def __call__(self, rng, observations, deterministic=False, clip=False):
        x = self.encoder(observations)
        x = jax.lax.stop_gradient(x)
        x = self.projection(x)
        actions = self.fc(x)
        actions = jnp.tanh(actions)  # first constraint the range of action
        if deterministic:
            return actions
        if clip:
            noise = jax.random.normal(rng, shape=(self.action_dim,)) * self.policy_noise
            noise = noise.clip(-self.clip_noise, self.clip_noise)
            actions = actions + noise
            actions = jnp.tanh(actions)
            return actions
        else:
            noise = jax.random.normal(rng, shape=(self.action_dim,)) * self.expl_noise
            actions = actions + noise
            actions = jnp.tanh(actions)
            return actions


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
            actions = jnp.tanh(actions)  # sampler actions always [-1, 1]

        assert jnp.all(jnp.isfinite(actions))
        return jax.device_get(actions)


class MPM(nn.Module):
    action_dim: int
    arch: str = "256-256"
    obs_type: str = "states"
    representation_dim: int = 256
    cnn_features: str = "32-32-32-32"
    cnn_strides: str = "2-1-1-1"
    cnn_padding: str = "VALID"

    def setup(self):
        if self.obs_type == "states":
            self.encoder = Identity(name="Encoder")
        elif self.obs_type == "pixels":
            self.encoder = Encoder(
                tuple(map(int, self.cnn_features.split("-"))),
                tuple(map(int, self.cnn_strides.split("-"))),
                self.cnn_padding,
                name="Encoder",
            )
        else:
            raise NotImplementedError
        self.trunk = [
            nn.Dense(self.representation_dim),
            nn.LayerNorm(self.representation_dim),
        ]

        self.forward_net = [
            FullyConnectedNetwork(output_dim=self.representation_dim, arch=self.arch)
        ]

        self.backward_net = [
            FullyConnectedNetwork(output_dim=self.action_dim, arch=self.arch),
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


class Sequential(nn.Module):
    layers: Sequence[Type[nn.Module]]

    @nn.compact
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Residual(nn.Module):
    layers: Sequence[Type[nn.Module]]

    @nn.compact
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x) + x
        return x


class PreNorm(nn.Module):
    layers: Sequence[Type[nn.Module]]

    def setup(self):
        self.norm = nn.LayerNorm()

    @nn.compact
    def __call__(self, x):
        for layer in self.layers:
            x = self.norm(x)
            x = layer(x)
        return x
