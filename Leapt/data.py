import datetime
import io
import random
import traceback
from collections import defaultdict
from copy import copy
from pathlib import Path

import numpy as np
import torch
from ml_collections import ConfigDict
from torch.utils.data import IterableDataset
import logging


"""
Various datasets' implementations for Brax simulation
"""

class Dataset(IterableDataset):
    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.fetch_every = 1
        config.max_size = int(1e42)
        config.n_worker = 2

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config


class EmptyDataset(IterableDataset):
    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.fetch_every = 1
        config.max_size = int(1e42)
        config.n_worker = 2

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, data_dir, sample_chuck_size):
        self.config = self.get_default_config(config)
        storage = DataStorage(data_dir)
        self._storage = storage
        self._episode_fns = []
        self._episodes = dict()
        self._size = 0

        self._n_worker = self.config.n_worker
        self._max_size = self.config.max_size / self._n_worker

        self._fetch_every = self.config.fetch_every
        self._samples_since_last_fetch = self.config.fetch_every

        self._sample_chuck_size = sample_chuck_size

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
            early_eps_fn.unlink(missing_ok=True)
        self._episode_fns.append(eps_fn)
        self._episodes[eps_fn] = episode
        self._size += eps_len
        return True

    def _try_fetch(self):
        if self._samples_since_last_fetch < self._fetch_every:
            return
        self._samples_since_last_fetch = 0
        try:
            worker_id = torch.utils.data.get_worker_info().id
        except:
            worker_id = 0
        fetched_size = 0
        for eps_fn in reversed(self._storage.all_eps_fns):
            eps_idx, eps_len = [int(x) for x in eps_fn.stem.split("_")[1:]]
            # if worker_id == 0:
            #     print(f"total number of eps is {len(self._storage.all_eps_fns)}")
            #     print(f"before store, eps_idx is {eps_idx}, worker_id is {worker_id}, number is {len(self._episode_fns)}")
            if eps_idx % self._n_worker != worker_id:
                continue
            if eps_fn in self._episodes.keys():
                break
            if fetched_size + eps_len > self._max_size:
                break
            fetched_size += eps_len
            if not self._store_episode(eps_fn):
                break
            # if worker_id == 0:
            #     print(f"after store, eps_idx is {eps_idx}, worker_id is {worker_id}, number is {len(self._episode_fns)}")

    def _sample(self):
        try:
            self._try_fetch()
        except:
            traceback.print_exc()
        self._samples_since_last_fetch += 1
        episode = self._sample_episode()
        return process_episode(episode, self._sample_chuck_size)

    def __iter__(self):
        while True:
            yield self._sample()


def process_episode(episode, num_sample):
    # episode = episode["episode"]
    idx = np.random.randint(0, episode_len(episode), size=num_sample)
    return episode[:, idx, :]


def save_episode(episode, fn):
    with io.BytesIO() as f1:
        np.save(f1, episode)
        f1.seek(0)
        with fn.open('wb') as f2:
            f2.write(f1.read())
    # with fn.open("wb") as f:
    #     np.save(f, episode)


def load_episode(fn):
    with fn.open("rb") as f:
        episode = np.load(f)
        # episode = {k: episode[k] for k in episode.keys()}
        return episode

def episode_len(episode):
    if isinstance(episode, dict):
        return list(episode.values())[0].shape[0]
    return episode.shape[1]


class DataStorage:
    """
    On disk data storage for Brax simulator
    """

    def __init__(self, data_dir):
        self._count_ep_len = 0
        self._data_dir = data_dir
        data_dir.mkdir(exist_ok=True)
        self._episode_fns = []
        self._preload()

    def __len__(self):
        return self._num_transitions

    def _preload(self):
        self._num_episodes = 0
        self._num_transitions = 0
        for fn in self._data_dir.glob("*.npy"):
            _, _, eps_len = fn.stem.split("_")
            self._num_episodes += 1
            self._num_transitions += int(eps_len)

    def add(self, episode):
        eps_fn = self._store_episode(episode)
        self._episode_fns.append(eps_fn)

    def _store_episode(self, episode):
        eps_idx = self._num_episodes
        eps_len = episode_len(episode)
        self._num_episodes += 1
        self._num_transitions += eps_len
        ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        eps_fn = f"{ts}_{eps_idx}_{eps_len}.npy"
        save_episode(episode, self._data_dir / eps_fn)
        return self._data_dir / eps_fn

    @property
    def all_eps_fns(self):
        return self._episode_fns


class LoadDataset(IterableDataset):
    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.fetch_every = 1
        config.max_size = int(1e42)
        config.n_worker = 2

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, data_dir, sample_chuck_size):
        self.config = self.get_default_config(config)
        self._episode_fns = []
        self._episodes = dict()
        self._size = 0
        self._data_dir = data_dir
        self._storage = None

        self._n_worker = self.config.n_worker
        self._max_size = self.config.max_size / self._n_worker

        self._fetch_every = self.config.fetch_every
        self._samples_since_last_fetch = self.config.fetch_every

        self._sample_chuck_size = sample_chuck_size

        self._load()

    def _load(self):
        logging.info(f"Labeling data... from {self._data_dir}")
        try:
            worker_id = torch.utils.data.get_worker_info().id
        except:
            worker_id = 0
        eps_fns = sorted(self._data_dir.glob("*.npy"))
        for eps_fn in eps_fns:
            if self._size > self._max_size:
                break
            eps_idx, eps_len = [int(x) for x in eps_fn.stem.split("_")[1:]]
            if eps_idx % self._n_worker != worker_id:
                continue
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
