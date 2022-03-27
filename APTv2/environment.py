import os
import pprint
import random
import tempfile
import time
import uuid
from copy import copy
from pathlib import Path
from socket import gethostname

import absl.flags
import cloudpickle as pickle
import gcsfs
import numpy as np
import wandb
from absl import logging
from dm_env import specs
from .dmc_env.dmc import make
from ml_collections import ConfigDict
from ml_collections.config_dict import config_dict
from ml_collections.config_flags import config_flags


class Environment(object):
    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.obs_type = "states"
        config.env_name = "walker_stand"
        config.frame_stack = 3
        config.action_repeat = 3
        config.seed = 42
        config.nchw = False

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config):
        self.config = self.get_default_config(config)
        self._environment = make(
            self.config.env_name,
            self.config.obs_type,
            self.config.frame_stack,
            self.config.action_repeat,
            self.config.seed,
            self.config.nchw,
        )

    @property
    def environment(self):
        return self._environment

    @property
    def action_dim(self):
        return self._environment.action_spec().shape[0]

    @property
    def observation_dim(self):
        return self._environment.observation_spec().shape
