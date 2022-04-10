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
from brax.envs import create_fn, create_gym_env
from ml_collections import ConfigDict


class Environment(object):
    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.obs_type = "states"
        config.env_name = "ant"
        config.action_repeat = 3
        config.seed = 42
        config.n_env = 2048

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, episode_length, eval_metrics=False):
        self.config = self.get_default_config(config)
        self._environment = create_fn(self.config.env_name)(
            episode_length=episode_length,
            action_repeat=self.config.action_repeat,
            batch_size=self.config.n_env,
            eval_metrics=eval_metrics,
        )

    def setup(self):
        return self.environment, self.observation_dim, self.action_dim

    @property
    def environment(self):
        return self._environment

    @property
    def action_dim(self):
        return self._environment.action_size

    @property
    def observation_dim(self):
        return (self._environment.observation_size,)


if __name__ == "__main__":
    environment = Environment(Environment.get_default_config(), episode_length=1000)
    # print(environment.action_dim, environment.observation_dim)
    env = environment.environment
    # print(env.reset().shape, type(env.reset())) # (b, d_s), jax device array
    # output = env.step(env.action_space.sample())

    # jit version
    import jax

    key_envs = jax.random.split(jax.random.PRNGKey(42), jax.local_device_count())
    step_fn = jax.jit(env.step)
    reset_fn = jax.jit(jax.vmap(env.reset))
    import ipdb

    ipdb.set_trace()
    reset_fn(key_envs)
    # reset_fn(key_envs)
