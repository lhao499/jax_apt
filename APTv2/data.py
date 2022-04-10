import datetime
import io
import random
import traceback
from collections import defaultdict
from copy import copy
from pathlib import Path
from socket import gethostname

import numpy as np
import torch
from absl import logging
from ml_collections import ConfigDict
from torch.utils.data import IterableDataset

"""
Various datasets' implementations for Brax simulation
"""


class UnlabelDataset(IterableDataset):
    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.fetch_every = 1000
        config.max_size = int(1e42)
        config.discount = 0.99
        config.n_step = 1

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, data_dir, n_worker, episode_length):
        self.config = self.get_default_config(config)
        storage = DataStorage(episode_length, data_dir)
        self._storage = storage
        self._episode_fns = []
        self._episodes = dict()
        self._size = 0

        self._n_worker = n_worker
        self._max_size = self.config.max_size / self._n_worker

        self._discount = self.config.discount
        self._nstep = self.config.n_step

        self._fetch_every = self.config.fetch_every
        self._samples_since_last_fetch = self.config.fetch_every

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
            if eps_idx % self._n_worker != worker_id:
                continue
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
            traceback.print_exc()
        self._samples_since_last_fetch += 1
        episode = self._sample_episode()
        return process_episode(episode, nstep=self._nstep, discounting=self._discount)

    def __iter__(self):
        while True:
            yield self._sample()


class RelabelDataset(IterableDataset):
    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.max_size = int(1e10)
        config.discount = 0.99
        config.n_step = 1

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, env, n_worker, data_dir):
        self.config = self.get_default_config(config)
        self._episode_fns = []
        self._episodes = dict()
        self._size = 0
        self._env = env
        self._data_dir = Path(data_dir)

        self._n_worker = n_worker
        self._max_size = self.config.max_size / self._n_worker

        self._discount = self.config.discount
        self._nstep = self.config.n_step

        self._load()

    def _load(self, relable=True):
        logging.info(f"Labeling data... from {self._data_dir}")
        try:
            worker_id = torch.utils.data.get_worker_info().id
        except:
            worker_id = 0
        eps_fns = sorted(self._data_dir.glob("*.npz"))
        for eps_fn in eps_fns:
            if self._size > self._max_size:
                break
            eps_idx, eps_len = [int(x) for x in eps_fn.stem.split("_")[1:]]
            if eps_idx % self._n_worker != worker_id:
                continue
            episode = load_episode(eps_fn)
            if relable:
                episode = self._relable_reward(episode)
            self._episode_fns.append(eps_fn)
            self._episodes[eps_fn] = episode
            self._size += episode_len(episode)

        returns = [np.sum(episode["reward"]) for episode in self._episodes.values()]
        logging.info(
            f"max {max(returns):.5f}. min {min(returns):.5f}. mean {np.mean(returns):.5f} median {np.median(returns):.5f}"
        )
        logging.info(
            f"max size {self._max_size}. loaded size {self._size}. worker number {self._n_worker}. episode number {len(returns)}"
        )

    def _sample_episode(self):
        eps_fn = random.choice(self._episode_fns)
        return self._episodes[eps_fn]

    def _relable_reward(self, episode):
        return relable_episode(self._env, episode)

    def _sample(self):
        episode = self._sample_episode()
        return process_episode(episode, nstep=self._nstep, discounting=self._discount)

    def __iter__(self):
        while True:
            yield self._sample()


class LearnLabelDataset(IterableDataset):
    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.fetch_every = 1000
        config.max_size = int(1e42)
        config.discount = 0.99
        config.n_step = 1

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, data_dir, n_worker, load_data_dir, episode_length):
        self.config = self.get_default_config(config)
        storage = DataStorage(episode_length, data_dir)
        self._storage = storage
        self._episode_fns = []
        self._episodes = dict()
        self._size = 0

        self._n_worker = n_worker
        self._max_size = self.config.max_size / self._n_worker

        self._discount = self.config.discount
        self._nstep = self.config.n_step

        self._fetch_every = self.config.fetch_every
        self._samples_since_last_fetch = self.config.fetch_every

        self._load_data_dir = Path(load_data_dir)

        self._load()

    def _load(self):
        logging.info(f"Loading data... from {self._load_data_dir}")
        try:
            worker_id = torch.utils.data.get_worker_info().id
        except:
            worker_id = 0
        eps_fns = sorted(self._load_data_dir.glob("*.npz"))
        for eps_fn in eps_fns:
            eps_idx, eps_len = [int(x) for x in eps_fn.stem.split("_")[1:]]
            if eps_idx % self._n_worker != worker_id:
                continue
            self._store_episode(eps_fn)
        logging.info(
            f"max size {self._max_size}. loaded size {self._size}. worker number {self._n_worker}. episode number {sum(1 for _ in self._episodes.values())}"
        )

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
            if eps_idx % self._n_worker != worker_id:
                continue
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
            traceback.print_exc()
        self._samples_since_last_fetch += 1
        episode = self._sample_episode()
        return process_episode(episode, nstep=self._nstep, discounting=self._discount)

    def __iter__(self):
        while True:
            yield self._sample()


def process_episode(episode, nstep=1, discounting=1):
    """
    Selecting utility to choose observation at idx and next observation at idx + nstep along second dim of (n, l, d)
    """
    idx = np.random.randint(0, episode_len(episode))
    """TODO: How about sample batch of environment?"""
    nth = np.random.randint(0, episode["observation"].shape[0])
    obs = episode["observation"][nth, idx]
    action = episode["action"][nth, idx]
    next_obs = episode["next_observation"][nth, idx]
    reward = np.zeros_like(episode["reward"][nth, idx])
    discount = np.ones_like(episode["reward"][nth, idx])
    for i in range(nstep):
        assert nstep == 1, NotImplementedError
        """NOTE: nstep must equal 1 since episode may not be complete"""
        step_reward = episode["reward"][nth, idx + i]
        reward += discount * step_reward
        discount *= episode["discount"][nth, idx + i] * discounting
    return dict(
        obs=obs, action=action, reward=reward, discount=discount, next_obs=next_obs
    )


# def flatten_batch_data(data):
#     """Utility function to flatten batch episode (n, l, d)"""
#     return data.reshape(-1, data.shape[-1])
# obs, action, reward, discount, next_obs = map(flatten_batch_data, [obs, action, reward, discount, next_obs])


def episode_len(episode):
    """
    Return episode length l from (n, l, d) data
    """
    return next(iter(episode.values())).shape[1]


def save_episode(episode, fn):
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **episode)
        bs.seek(0)
        with fn.open("wb") as f:
            f.write(bs.read())


def load_episode(fn):
    with fn.open("rb") as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}
        return episode


def relable_episode(env, episode):
    rewards = []
    reward_spec = env.reward_spec()
    states = episode["physics"]
    for i in range(states.shape[0]):
        with env.physics.reset_context():
            env.physics.set_state(states[i])
        reward = env.task.get_reward(env.physics)
        reward = np.full(reward_spec.shape, reward, reward_spec.dtype)
        rewards.append(reward)
    episode["reward"] = np.array(rewards, dtype=reward_spec.dtype)[..., None]
    return episode


class DataStorage:
    """
    On disk data storage for Brax simulator
    """

    def __init__(self, episode_length, data_dir):
        self._episode_length = episode_length
        self._count_ep_len = 0
        self._data_dir = data_dir
        data_dir.mkdir(exist_ok=True)
        self._current_episode = {}
        self._current_episode = defaultdict(list)
        self._episode_fns = []
        self._preload()

    def __len__(self):
        return self._num_transitions

    @property
    def _is_an_episode(self):
        self._count_ep_len += 1
        return self._count_ep_len % (self._episode_length - 1) == 0

    def add(self, obs, action, next_obs, reward, done, physics):
        self._current_episode["observation"].append(obs)
        self._current_episode["action"].append(action)
        self._current_episode["next_observation"].append(next_obs)
        self._current_episode["reward"].append(reward)
        self._current_episode["done"].append(done)
        self._current_episode["physics"].append(physics)
        if self._is_an_episode:
            episode = dict()
            for name in self._current_episode.keys():
                value = self._current_episode[name]
                """batch of episodes (n, l, d)"""
                episode[name] = np.array(value)
            self._current_episode = defaultdict(list)
            eps_fn = self._store_episode(episode)
            self._episode_fns.append(eps_fn)

    def _preload(self):
        self._num_episodes = 0
        self._num_transitions = 0
        for fn in self._data_dir.glob("*.npz"):
            _, _, eps_len = fn.stem.split("_")
            self._num_episodes += 1
            self._num_transitions += int(eps_len)

    def _store_episode(self, episode):
        eps_idx = self._num_episodes
        eps_len = episode_len(episode)
        self._num_episodes += 1
        self._num_transitions += eps_len
        ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        eps_fn = f"{ts}_{eps_idx}_{eps_len}.npz"
        save_episode(episode, self._data_dir / eps_fn)
        return self._data_dir / eps_fn

    @property
    def all_eps_fns(self):
        return self._episode_fns
