import datetime
import glob
import io
import os
import random
import traceback
from collections import defaultdict
from concurrent.futures import process
from functools import partial

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


def episode_len(episode):
    # subtract -1 because the dummy first transition
    return next(iter(episode.values())).shape[0] - 1


def save_episode(episode, fn):
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **episode)
        bs.seek(0)
        with fn.open("wb") as f:
            f.write(bs.read())


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
            self._store_episode(episode)

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


def load_episode(fn):
    with fn.open("rb") as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}
        return episode


def load_episode_vanila(fn):
    with open(fn, "rb") as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}
        return episode


class ReplayBuffer:
    def __init__(self, storage, max_size, nstep, discount, fetch_every, batch_size):
        self._storage = storage
        self._size = 0
        self._max_size = max_size
        self._episode_fns = []
        self._episodes = dict()
        self._nstep = nstep
        self._discount = discount
        self._fetch_every = fetch_every
        self._samples_since_last_fetch = fetch_every
        self._batch_size = batch_size
        self._load()
        self._dataset = None

    def _load(self):
        self._eps_fns = glob.glob(
            os.path.join(self._storage._replay_dir.as_posix(), "*.npz")
        )

    def dataset(self):
        if self._dataset is None:
            self._dataset = self._create_dataset(self._batch_size)
        return self._dataset

    def _create_dataset(self, batch_size):
        example = next(iter(self._sample_sequence()))
        dataset = tf.data.Dataset.from_generator(
            lambda: self._sample_sequence(),
            {k: v.dtype for k, v in example.items()},
            {k: v.shape for k, v in example.items()},
        )
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        dataset = dataset.as_numpy_iterator()
        return dataset

    def _sample_sequence(self):
        while True:
            ep_fn = random.choice(self._eps_fns)
            episode = load_episode_vanila(ep_fn)
            episode = process_episode(episode, self._nstep, self._discount)
            yield episode


def make_replay_loader(
    storage,
    max_size,
    batch_size,
    n_worker,
    save_snapshot,
    nstep,
    discount,
    downstream,
    env,
    replay_dir,
):
    if downstream:
        pass
    else:
        dataset = ReplayBuffer(storage, max_size, nstep, discount, 1000, batch_size)
    return dataset
