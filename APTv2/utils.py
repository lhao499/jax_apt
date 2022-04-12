import os
import pprint
import random
import re
import tempfile
import time
import uuid
from copy import copy
from functools import partial
from pathlib import Path
from socket import gethostname

import absl.flags
import cloudpickle as pickle
import flax
import gcsfs
import imageio
import jax
import jax.numpy as jnp
import numpy as np
import wandb
from absl import logging
from flax import jax_utils
from ml_collections import ConfigDict
from ml_collections.config_dict import config_dict
from ml_collections.config_flags import config_flags


class JaxRNG(object):
    def __init__(self, seed):
        self.rng = jax.random.PRNGKey(seed)

    def __call__(self, n=1):
        if n == 1:
            self.rng, next_rng = jax.random.split(self.rng)
            return next_rng
        rngs = jax.random.split(self.rng, n + 1)
        self.rng = rngs[0]
        return tuple(rngs[1:])


def init_rng(seed):
    global jax_utils_rng
    jax_utils_rng = JaxRNG(seed)


def next_rng(n=1):
    global jax_utils_rng
    return jax_utils_rng(n)


def get_metrics(metrics, unreplicate=False):
    if unreplicate:
        metrics = jax_utils.unreplicate(metrics)
    metrics = jax.device_get(metrics)
    return {key: float(val) for key, val in metrics.items()}


def extend_and_repeat(tensor, axis, repeat):
    return jnp.repeat(jnp.expand_dims(tensor, axis), repeat, axis=axis)


def mse_loss(val, target):
    return jnp.mean(jnp.square(val - target))


def value_and_multi_grad(fun, n_outputs, argnums=0, has_aux=False):
    def select_output(index):
        def wrapped(*args, **kwargs):
            if has_aux:
                x, *aux = fun(*args, **kwargs)
                return (x[index], *aux)
            else:
                x = fun(*args, **kwargs)
                return x[index]

        return wrapped

    grad_fns = tuple(
        jax.value_and_grad(select_output(i), argnums=argnums, has_aux=has_aux)
        for i in range(n_outputs)
    )

    def multi_grad_fn(*args, **kwargs):
        grads = []
        values = []
        for grad_fn in grad_fns:
            (value, *aux), grad = grad_fn(*args, **kwargs)
            values.append(value)
            grads.append(grad)
        return (tuple(values), *aux), tuple(grads)

    return multi_grad_fn


def batch_to_jax(batch):
    return jax.tree_map(jax.device_put, batch)


def random_crop(key, img, padding):
    crop_from = jax.random.randint(key, (2,), 0, 2 * padding + 1)
    crop_from = jnp.concatenate([crop_from, jnp.zeros((1,), dtype=jnp.int32)])
    padded_img = jnp.pad(
        img, ((padding, padding), (padding, padding), (0, 0)), mode="edge"
    )
    return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)


def batched_random_crop(key, imgs, padding=4):
    keys = jax.random.split(key, imgs.shape[0])
    return jax.vmap(random_crop, (0, 0, None))(keys, imgs, padding)


class Timer(object):
    def __init__(self):
        self._time = None

    def __enter__(self):
        self._start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self._time = time.time() - self._start_time

    def __call__(self):
        return self._time


class VideoRecorder:
    def __init__(
        self,
        root_dir,
        render_size=256,
        fps=20,
        camera_id=0,
        is_train=False,
    ):
        self.is_train = is_train
        self.save_dir = root_dir / ("train_video" if self.is_train else "test_video")
        self.render_size = render_size
        self.fps = fps
        self.frames = []
        self.camera_id = camera_id

    def init(self, env):
        self.frames = []
        self.record(env)

    def log_to_wandb(self):
        frames = np.transpose(np.array(self.frames), (0, 3, 1, 2))
        fps, skip = 6, 8
        wandb.log(
            {
                "train/video"
                if self.is_train
                else "eval/video": wandb.Video(
                    frames[::skip, :, ::2, ::2], fps=fps, format="gif"
                )
            }
        )

    def save(self, filename):
        self.log_to_wandb()
        path = self.save_dir / filename
        imageio.mimsave(str(path), self.frames, fps=self.fps)

    def record(self, env):
        if hasattr(env, "physics"):
            frame = env.physics.render(
                height=self.render_size,
                width=self.render_size,
                camera_id=self.camera_id,
            )
        else:
            frame = env.render()
        self.frames.append(frame)


def define_flags_with_default(**kwargs):
    for key, val in kwargs.items():
        if isinstance(val, ConfigDict):
            config_flags.DEFINE_config_dict(key, val)
        elif isinstance(val, bool):
            # Note that True and False are instances of int.
            absl.flags.DEFINE_bool(key, val, "automatically defined flag")
        elif isinstance(val, int):
            absl.flags.DEFINE_integer(key, val, "automatically defined flag")
        elif isinstance(val, float):
            absl.flags.DEFINE_float(key, val, "automatically defined flag")
        elif isinstance(val, str):
            absl.flags.DEFINE_string(key, val, "automatically defined flag")
        else:
            raise ValueError("Incorrect value type")
    return kwargs


def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    init_rng(seed)


def print_flags(flags, flags_def):
    logging.info(
        "Running training with hyperparameters: \n{}".format(
            pprint.pformat(
                [
                    "{}: {}".format(key, val)
                    for key, val in get_user_flags(flags, flags_def).items()
                ]
            )
        )
    )


def get_user_flags(flags, flags_def):
    output = {}
    for key in flags_def:
        val = getattr(flags, key)
        if isinstance(val, ConfigDict):
            output.update(flatten_config_dict(val, prefix=key))
        else:
            output[key] = val

    return output


def flatten_config_dict(config, prefix=None):
    output = {}
    for key, val in config.items():
        if isinstance(val, ConfigDict):
            output.update(flatten_config_dict(val, prefix=key))
        else:
            if prefix is not None:
                output["{}.{}".format(prefix, key)] = val
            else:
                output[key] = val
    return output


def prefix_metrics(metrics, prefix):
    return {"{}/{}".format(prefix, key): value for key, value in metrics.items()}


class WandBLogger(object):
    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.online = False
        config.prefix = "APTv2"
        config.project = "aptv2"
        config.output_dir = "/tmp/aptv2"
        config.gcs_output_dir = ""
        config.random_delay = 0.0
        config.experiment_id = config_dict.placeholder(str)
        config.anonymous = config_dict.placeholder(str)
        config.notes = config_dict.placeholder(str)

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, variant, enable=True):
        self.enable = enable
        self.config = self.get_default_config(config)

        if self.config.experiment_id is None:
            self.config.experiment_id = uuid.uuid4().hex

        if self.config.prefix != "":
            self.config.project = "{}--{}".format(
                self.config.prefix, self.config.project
            )

        if self.enable:
            if self.config.output_dir == "":
                self.config.output_dir = tempfile.mkdtemp()
            else:
                self.config.output_dir = os.path.join(
                    self.config.output_dir, self.config.experiment_id
                )
                os.makedirs(self.config.output_dir, exist_ok=True)

            if self.config.gcs_output_dir != "":
                self.config.gcs_output_dir = os.path.join(
                    self.config.gcs_output_dir, self.config.experiment_id
                )

        self._variant = copy(variant)

        if "hostname" not in self._variant:
            self._variant["hostname"] = gethostname()

        if self.config.random_delay > 0:
            time.sleep(np.random.uniform(0, self.config.random_delay))

        if self.enable:
            self.run = wandb.init(
                reinit=True,
                config=self._variant,
                project=self.config.project,
                dir=self.config.output_dir,
                id=self.config.experiment_id,
                anonymous=self.config.anonymous,
                notes=self.config.notes,
                settings=wandb.Settings(
                    start_method="thread",
                    _disable_stats=True,
                ),
                mode="online" if self.config.online else "offline",
            )
        else:
            self.run = None

    def log(self, *args, **kwargs):
        if self.enable:
            self.run.log(*args, **kwargs)

    def save_pickle(self, obj, filename):
        if self.enable:
            with open(os.path.join(self.config.output_dir, filename), "wb") as fout:
                pickle.dump(obj, fout)

            if self.config.gcs_output_dir != "":
                path = os.path.join(self.config.gcs_output_dir, filename)
                with gcsfs.GCSFileSystem().open(path, "wb") as fout:
                    pickle.dump(obj, fout)

    @property
    def experiment_id(self):
        return self.config.experiment_id

    @property
    def variant(self):
        return self.config.variant

    @property
    def output_dir(self):
        return Path(self.config.output_dir)


def update_target_network(main_params, target_params, tau):
    return jax.tree_multimap(
        lambda x, y: tau * x + (1.0 - tau) * y, main_params, target_params
    )


def load_pickle(path):
    if path.startswith("gs://"):
        with gcsfs.GCSFileSystem().open(path) as fin:
            data = pickle.load(fin)
    else:
        with open(path, "rb") as fin:
            data = pickle.load(fin)
    return data


def load_checkpoint(path):
    data = load_pickle(path)
    logging.info(
        "Loading checkpoint from %s, saved at step %d",
        path,
        data["step"],
    )
    return data
