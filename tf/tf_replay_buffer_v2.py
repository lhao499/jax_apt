import datetime
import glob
import io
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
    print("type of episode", type(episode))
    # add +1 for the first dummy transition
    idx = np.random.randint(0, episode_len(episode) - nstep + 1) + 1
    obs = episode["observation"][idx - 1]
    action = episode["action"][idx]
    next_obs = episode["observation"][idx + nstep - 1]
    reward = tf.zeros_like(episode["reward"][idx])
    discount = tf.ones_like(episode["discount"][idx])
    for i in range(nstep):
        step_reward = episode["reward"][idx + i]
        reward += discount * step_reward
        discount *= episode["discount"][idx + i] * discounting
    return dict(
        obs=obs, action=action, reward=reward, discount=discount, next_obs=next_obs
    )


@tf.function
def load_episode(fn):
    import ipdb

    ipdb.set_trace()
    with fn.open("rb") as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}
        return episode


class ReplayBuffer:
    def __init__(self, storage, fetch_every, nstep, discount, batch_size):
        self._storage = storage
        self._fetch_every = fetch_every
        self._samples_since_last_fetch = fetch_every
        self._nstep = nstep
        self._discount = discount
        self._batch_size = batch_size
        self._load()

    def _create_dataset(self):
        if self._samples_since_last_fetch % self._fetch_every == 0:
            dataset = tf.data.Dataset.from_tensor_slices(self._eps_fns)
            dataset = dataset.shuffle(buffer_size=len(self._eps_fns))
            # dataset = dataset.map(tf.io.read_file, num_parallel_calls=tf.data.AUTOTUNE)
            # dataset = dataset.map(load_episode, num_parallel_calls=tf.data.AUTOTUNE)
            dataset = dataset.map(
                partial(process_episode, nstep=self._nstep, discounting=self._discount),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
            dataset = dataset.batch(self._batch_size)
            dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
            dataset = dataset.as_numpy_iterator()
            self.dataset = iter(dataset)
        self._samples_since_last_fetch += 1
        return next(self.dataset)

    def _load(self):
        self._eps_fns = glob.glob(str(self._storage._replay_dir) + "/*.npz")

    def __iter__(self):
        while True:
            yield self._create_dataset()


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
        dataset = iter(ReplayBuffer(storage, 1000, nstep, discount, batch_size))
    return dataset
