import jax


class RolloutStorage(object):
    def __init__(self, env, max_traj_length=1000, video_recorder=None):
        self.max_traj_length = max_traj_length
        self._env = env
        self._traj_steps = 0
        self._video_recorder = video_recorder
        self._current_time_step = self._env.reset()
        self._done = True

    def sample_traj(
        self,
        rng,
        policy,
        n_trajs,
        deterministic=False,
        replay_storage=None,
        random=False,
    ):
        r_traj = 0
        for _ in range(n_trajs):
            if self._done or self._traj_steps >= self.max_traj_length:
                self._traj_steps = 0

                time_step, self._done = self._env.reset(), False
                if self._video_recorder is not None:
                    self._video_recorder.init(self._env)
                    self._video_recorder.record(self._env)
                phys = dict(physics=self._env._env.physics.state())
                self._current_time_step = time_step

                if replay_storage is not None:
                    replay_storage.add(time_step, phys)

            while True:
                self._traj_steps += 1
                rng, split_rng = jax.random.split(rng)
                action = policy(
                    split_rng,
                    self._current_time_step["observation"],
                    deterministic=deterministic,
                    random=random,
                ).reshape(-1)
                time_step = self._env.step(action)
                if self._video_recorder is not None:
                    self._video_recorder.record(self._env)
                phys = dict(physics=self._env._env.physics.state())
                self._current_time_step = time_step
                r_traj += time_step["reward"]
                self._done = time_step.last()

                if replay_storage is not None:
                    replay_storage.add(time_step, phys)

                if self._done:
                    break

        data = dict(r_traj=r_traj / n_trajs)
        return data, rng

    def sample_step(
        self,
        rng,
        policy,
        n_steps,
        deterministic=False,
        replay_storage=None,
        random=False,
    ):
        for _ in range(n_steps):
            if self._done or self._traj_steps >= self.max_traj_length:
                self._traj_steps = 0

                time_step, self._done = self._env.reset(), False
                if self._video_recorder is not None:
                    self._video_recorder.init(self._env)
                    self._video_recorder.record(self._env)
                phys = dict(physics=self._env._env.physics.state())
                self._current_time_step = time_step

                if replay_storage is not None:
                    replay_storage.add(time_step, phys)

            self._traj_steps += 1
            rng, split_rng = jax.random.split(rng)
            action = policy(
                split_rng,
                self._current_time_step["observation"],
                deterministic=deterministic,
                random=random,
            ).reshape(-1)
            time_step = self._env.step(action)
            if self._video_recorder is not None:
                self._video_recorder.record(self._env)
            phys = dict(physics=self._env._env.physics.state())
            self._current_time_step = time_step
            self._done = time_step.last()

            if replay_storage is not None:
                replay_storage.add(time_step, phys)

        data = dict()
        return data, rng
