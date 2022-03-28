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


class UnlabelDataset(IterableDataset):
    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.fetch_every = 1000
        config.max_size = int(1e12)
        config.discount = 0.99
        config.n_step = 3

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, data_specs, phys_specs, replay_dir, n_worker):
        self.config = self.get_default_config(config)
        storage = ReplayBufferStorage(data_specs, phys_specs, replay_dir)
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
        config.n_step = 3

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, env, n_worker, replay_dir):
        self.config = self.get_default_config(config)
        self._episode_fns = []
        self._episodes = dict()
        self._size = 0
        self._env = env
        self._replay_dir = Path(replay_dir)

        self._n_worker = n_worker
        self._max_size = self.config.max_size / self._n_worker

        self._discount = self.config.discount
        self._nstep = self.config.n_step

        self._load()
        self._stat()

    def _stat(self):
        returns = [np.sum(episode["reward"]) for episode in self._episodes.values()]
        logging.info(f"max {max(returns):.5f}. min {min(returns):.5f}. mean {np.mean(returns):.5f} median {np.median(returns):.5f}")
        logging.info(f"max size {self._max_size}. size {self._size}. worker number {self._n_worker}. episode number {len(returns)}")

    def _load(self, relable=True):
        logging.info(f"Labeling data... from {self._replay_dir}")
        try:
            worker_id = torch.utils.data.get_worker_info().id
        except:
            worker_id = 0
        eps_fns = sorted(self._replay_dir.glob("*.npz"))
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


def process_episode(episode, nstep=1, discounting=1):
    # add +1 for the first dummy transition
    idx = np.random.randint(0, episode_len(episode) - nstep + 1) + 1
    obs = episode["observation"][idx - 1]
    action = episode["action"][idx]
    next_obs = episode["observation"][idx + nstep - 1]
    reward = np.zeros_like(episode["reward"][idx])
    discount = np.ones_like(episode["discount"][idx])
    for i in range(nstep):
        step_reward = episode["reward"][idx + i]
        reward += discount * step_reward
        discount *= episode["discount"][idx + i] * discounting
    return dict(
        obs=obs, action=action, reward=reward, discount=discount, next_obs=next_obs
    )


def episode_len(episode):
    # subtract -1 because the dummy first transition
    return next(iter(episode.values())).shape[0] - 1


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


class ReplayBufferStorage:
    def __init__(self, data_specs, phys_specs, replay_dir):
        self._data_specs = data_specs
        self._phys_specs = phys_specs
        self._replay_dir = replay_dir
        replay_dir.mkdir(exist_ok=True)
        self._current_episode = defaultdict(list)
        self._episode_fns = []
        self._preload()

    def __len__(self):
        return self._num_transitions

    def add(self, time_step, phys):
        for key, value in phys.items():
            self._current_episode[key].append(value)
        for spec in self._data_specs:
            value = time_step[spec.name]
            if np.isscalar(value):
                value = np.full(spec.shape, value, spec.dtype)
            assert spec.shape == value.shape and spec.dtype == value.dtype
            self._current_episode[spec.name].append(value)
        if time_step.last():
            episode = dict()
            for spec in self._data_specs:
                value = self._current_episode[spec.name]
                episode[spec.name] = np.array(value, spec.dtype)
            for spec in self._phys_specs:
                value = self._current_episode[spec.name]
                episode[spec.name] = np.array(value, spec.dtype)
            self._current_episode = defaultdict(list)
            eps_fn = self._store_episode(episode)
            self._episode_fns.append(eps_fn)

    def _preload(self):
        self._num_episodes = 0
        self._num_transitions = 0
        for fn in self._replay_dir.glob("*.npz"):
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
        save_episode(episode, self._replay_dir / eps_fn)
        return self._replay_dir / eps_fn

    @property
    def all_eps_fns(self):
        return self._episode_fns
