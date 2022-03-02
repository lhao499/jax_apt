import datetime
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


class ReplayBuffer:
    def __init__(self, storage, max_size, fetch_every):
        self._storage = storage
        self._size = 0
        self._max_size = max_size
        self._episode_fns = []
        self._episodes = dict()
        self._fetch_every = fetch_every
        self._samples_since_last_fetch = fetch_every

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
        self._episode_fns.sort()
        self._episodes[eps_fn] = episode
        self._size += eps_len
        return True

    def _try_fetch(self):
        if self._samples_since_last_fetch < self._fetch_every:
            return
        self._samples_since_last_fetch = 0
        eps_fns = sorted(self._storage._replay_dir.glob("*.npz"), reverse=True)
        fetched_size = 0
        while True:
            eps_fn = random.choice(eps_fns)
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
            traceback.print_exc()
        self._samples_since_last_fetch += 1
        episode = self._sample_episode()
        return episode

    def __iter__(self):
        while True:
            yield self._sample()


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
        iterable = ReplayBuffer(storage, max_size, 1000)

    example = next(iter(iterable))

    dataset = tf.data.Dataset.from_generator(
        lambda: iterable,
        {k: v.dtype for k, v in example.items()},
        {k: v.shape for k, v in example.items()},
    )

    def process_episode(episode, nstep=1, discounting=1):
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

    def wrapped_process_episode(episode):
        return process_episode(episode, nstep=nstep, discounting=discount)
        # return tf.function(process_episode(episode, nstep=nstep, discounting=discount), input_signature=example)

    dataset = dataset.map(
        wrapped_process_episode,
        num_parallel_calls=tf.data.AUTOTUNE if n_worker <= 0 else n_worker,
    )
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    # dataset = dataset.make_initializable_iterator()
    dataset = dataset.as_numpy_iterator()
    return iter(dataset)
