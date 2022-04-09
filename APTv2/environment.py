import functools
import os
import pprint
import random
import tempfile
import time
import uuid
from copy import copy
from pathlib import Path
from socket import gethostname
from typing import Callable, Optional, Union, overload

import absl.flags
import brax
import cloudpickle as pickle
import gcsfs
import gym
import numpy as np
import wandb
from absl import logging
from brax.envs import (
    acrobot,
    ant,
    fast,
    fetch,
    grasp,
    halfcheetah,
    hopper,
    humanoid,
    humanoid_standup,
    inverted_double_pendulum,
    inverted_pendulum,
    reacher,
    reacherangle,
    swimmer,
    ur5e,
    walker2d,
    wrappers,
)
from brax.envs.env import Env, State, Wrapper
from ml_collections import ConfigDict

_envs = {
    "acrobot": acrobot.Acrobot,
    "ant": ant.Ant,
    "fast": fast.Fast,
    "fetch": fetch.Fetch,
    "grasp": grasp.Grasp,
    "halfcheetah": halfcheetah.Halfcheetah,
    "hopper": hopper.Hopper,
    "humanoid": humanoid.Humanoid,
    "humanoidstandup": humanoid_standup.HumanoidStandup,
    "inverted_pendulum": inverted_pendulum.InvertedPendulum,
    "inverted_double_pendulum": inverted_double_pendulum.InvertedDoublePendulum,
    "reacher": reacher.Reacher,
    "reacherangle": reacherangle.ReacherAngle,
    "swimmer": swimmer.Swimmer,
    "ur5e": ur5e.Ur5e,
    "walker2d": walker2d.Walker2d,
}


def get_environment(env_name, **kwargs):
    return _envs[env_name](**kwargs)


def create(
    env_name: str,
    episode_length: int = 1000,
    action_repeat: int = 1,
    auto_reset: bool = True,
    batch_size: Optional[int] = None,
    eval_metrics: bool = False,
    **kwargs
) -> Env:
    """Creates an Env with a specified brax system."""
    env = _envs[env_name](**kwargs)
    if episode_length is not None:
        env = wrappers.EpisodeWrapper(env, episode_length, action_repeat)
    if batch_size:
        env = wrappers.VectorWrapper(env, batch_size)
    if auto_reset:
        env = wrappers.AutoResetWrapper(env)
    if eval_metrics:
        env = wrappers.EvalWrapper(env)

    return env  # type: ignore


def create_fn(env_name: str, **kwargs) -> Callable[..., Env]:
    """Returns a function that when called, creates an Env."""
    return functools.partial(create, env_name, **kwargs)


@overload
def create_gym_env(env_name: str,
                   batch_size: None = None,
                   seed: int = 0,
                   backend: Optional[str] = None,
                   **kwargs) -> gym.Env:
  ...


@overload
def create_gym_env(env_name: str,
                   batch_size: int,
                   seed: int = 0,
                   backend: Optional[str] = None,
                   **kwargs) -> gym.vector.VectorEnv:
  ...


def create_gym_env(env_name: str,
                   batch_size: Optional[int] = None,
                   seed: int = 0,
                   backend: Optional[str] = None,
                   **kwargs) -> Union[gym.Env, gym.vector.VectorEnv]:
  """Creates a `gym.Env` or `gym.vector.VectorEnv` from a Brax environment."""
  environment = create(env_name=env_name, batch_size=batch_size, **kwargs)
  if batch_size is None:
    return wrappers.GymWrapper(environment, seed=seed, backend=backend)
  if batch_size <= 0:
    raise ValueError(
        '`batch_size` should either be None or a positive integer.')
  return wrappers.VectorGymWrapper(environment, seed=seed, backend=backend)


class Environment(object):
    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.obs_type = "states"
        config.env_name = "walker2d"
        config.action_repeat = 3
        config.seed = 42
        config.batch_size = 2048
        config.eval_metrics = False

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, episode_length):
        self.config = self.get_default_config(config)
        # create_gym_env
        self._environment = functools.partial(create_gym_env, env_name=self.config.env_name)
        # (
        #     # episode_length=episode_length,
        #     # action_repeat=self.config.action_repeat,
        #     # seed=self.config.seed,
        #     batch_size=self.config.batch_size,
        #     # eval_metrics=self.config.eval_metrics,
        # )
        # self._environment = create_fn(
        #     self.config.env_name,
        #     # episode_length=episode_length,
        #     # action_repeat=self.config.action_repeat,
        #     # seed=self.config.seed,
        #     # batch_size=self.config.batch_size,
        #     # eval_metrics=self.config.eval_metrics,
        # )

    @property
    def environment(self):
        return self._environment

    @property
    def action_dim(self):
        return self._environment.action_space.shape[0]

    @property
    def observation_dim(self):
        return self._environment.observation_space.shape


if __name__ == "__main__":
    environment = Environment(Environment.get_default_config(), episode_length=1000)
    # print(environment.action_dim, environment.observation_dim)
    env = environment.environment
    # print(env.reset().shape, type(env.reset())) # (b, d_s), jax device array
    # output = env.step(env.action_space.sample())

    # env = create_fn("walker2d")
    env = env(action_repeat=1)
    import jax
    key_envs = jax.random.split(jax.random.PRNGKey(42), jax.local_device_count())
    step_fn = jax.jit(env.step)
    reset_fn = jax.jit(jax.vmap(env.reset))
    import ipdb; ipdb.set_trace()
    reset_fn(key_envs)
