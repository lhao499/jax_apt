import datetime
import io
import random
import traceback
from collections import defaultdict
from copy import copy
from pathlib import Path

import numpy as np
from ml_collections import ConfigDict
import logging


"""
Various datasets' implementations for Brax simulation
"""


class Dataset():
    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.fetch_every = 1
        config.max_size = int(1e42)

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config


class EmptyDataset(Dataset):
    @staticmethod
    def get_default_config(updates=None):
        return super(EmptyDataset, EmptyDataset).get_default_config(updates)

    def __init__(self, config, data_dir, sample_chuck_size):
        self.config = self.get_default_config(config)
        self._episode_fns = []
        self._episodes = dict()
        self._size = 0
        self._max_size = self.config.max_size

        self._fetch_every = self.config.fetch_every
        self._fetch_count = self.config.fetch_every

        self._sample_chuck_size = sample_chuck_size

        # storage
        self._data_dir = data_dir
        data_dir.mkdir(exist_ok=True)
        self._pre_episode_fns = []
        self._preload()

    def _sample_episode(self):
        eps_fn = random.choice(self._episode_fns)
        return self._episodes[eps_fn]

    def _store_episode(self, eps_fn):
        try:
            episode = load_episode(eps_fn)
        except:
            return False
        eps_len = episode_len(episode)
        while eps_len + self._size > self._max_size:
            early_eps_fn = self._episode_fns.pop(0)
            early_eps = self._episodes.pop(early_eps_fn)
            self._size -= episode_len(early_eps)
            # early_eps_fn.unlink(missing_ok=True) # NOTE: not relevant for keep replay buffer size
        self._episode_fns.append(eps_fn)
        self._episodes[eps_fn] = episode
        self._size += eps_len
        return True

    def _try_fetch(self):
        if self._fetch_count < self._fetch_every:
            return
        fetched_size = 0
        for eps_fn in reversed(self._pre_episode_fns):
            eps_idx, eps_len = [int(x) for x in eps_fn.stem.split("_")[1:]]
            if eps_fn in self._episodes.keys():
                break
            if fetched_size + eps_len > self._max_size:
                break
            fetched_size += eps_len
            if not self._store_episode(eps_fn):
                break

    def _sample(self):
        try:
            self._try_fetch()
        except:
            print("bad thing happened....")
            traceback.print_exc()
        self._fetch_count += 1
        episode = self._sample_episode()
        return dict(episode=process_episode(episode, self._sample_chuck_size))

    def __iter__(self):
        while True:
            yield self._sample()

    def _generate_chunks(self):
        while True:
            yield self._sample()

    def _save_episode(self, episode):
        eps_idx = self._num_episodes
        eps_len = episode_len(episode)
        self._num_episodes += 1
        self._num_transitions += eps_len
        ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        eps_fn = f"{ts}_{eps_idx}_{eps_len}.npz"
        save_episode(episode, self._data_dir / eps_fn)
        return self._data_dir / eps_fn

    def _preload(self):
        self._num_episodes = 0
        self._num_transitions = 0
        for fn in self._data_dir.glob("*.npz"):
            _, _, eps_len = fn.stem.split("_")
            self._num_episodes += 1
            self._num_transitions += int(eps_len)

    def add(self, episode):
        episode = dict(episode=episode)
        episode = {key: convert(value) for key, value in episode.items()}
        eps_fn = self._save_episode(episode)
        self._pre_episode_fns.append(eps_fn)


def process_episode(episode, num_sample):
    episode = episode["episode"]
    idx = np.random.randint(0, episode_len(episode), size=num_sample)
    return episode[:, idx, :]


def save_episode(episode, fn):
    with io.BytesIO() as f1:
        np.savez_compressed(f1, **episode)
        f1.seek(0)
        with fn.open("wb") as f2:
            f2.write(f1.read())


def load_episode(fn):
    with fn.open("rb") as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}
        return episode


def episode_len(episode):
    if isinstance(episode, dict):
        return list(episode.values())[0].shape[0]
    return episode.shape[1]


def convert(value):
    value = np.array(value)
    if np.issubdtype(value.dtype, np.floating):
        return value.astype(np.float32)
    elif np.issubdtype(value.dtype, np.signedinteger):
        return value.astype(np.int32)
    elif np.issubdtype(value.dtype, np.uint8):
        return value.astype(np.uint8)
    return value

class LoadDataset(Dataset):
    @staticmethod
    def get_default_config(updates=None):
        return super(LoadDataset, LoadDataset).get_default_config(updates)

    def __init__(self, config, data_dir, sample_chuck_size):
        self.config = self.get_default_config(config)
        self._episode_fns = []
        self._episodes = dict()
        self._size = 0
        self._data_dir = data_dir
        self._storage = None
        self._max_size = self.config.max_size

        self._sample_chuck_size = sample_chuck_size

        self._load()

    def _load(self):
        logging.info(f"Labeling data... from {self._data_dir}")
        eps_fns = sorted(self._data_dir.glob("*.npz"))
        for eps_fn in eps_fns:
            if self._size > self._max_size:
                break
            episode = load_episode(eps_fn)
            self._episode_fns.append(eps_fn)
            self._episodes[eps_fn] = episode
            self._size += episode_len(episode)

    def _sample_episode(self):
        eps_fn = random.choice(self._episode_fns)
        return self._episodes[eps_fn]

    def _sample(self):
        episode = self._sample_episode()
        return process_episode(episode, self._sample_chuck_size)

    def __iter__(self):
        while True:
            yield self._sample()
