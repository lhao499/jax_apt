import jax

class RolloutStorage(object):
    def __init__(self, env, max_traj_length=1000, video_recorder=None):
        self.max_traj_length = max_traj_length
        self._env = env.environment
        self._traj_steps = 0
        self._video_recorder = video_recorder
        self._current_obs = self._env.reset()[0]
        self._done = True
        self._step_fn = jax.jit(self._env.step)
        self._reset_fn = jax.jit(jax.vmap(self._env.reset))

    def sample_traj(
        self,
        rng,
        policy,
        n_trajs,
        deterministic=False,
        data_storage=None,
        random=False,
    ):
        r_traj = 0
        for _ in range(n_trajs):
            if self._done or self._traj_steps >= self.max_traj_length:
                self._traj_steps = 0

                (obs, reward, done, info), self._done = self._reset_fn(), False

                if self._video_recorder is not None:
                    self._video_recorder.init(self._env)
                    self._video_recorder.record(self._env)

                self._current_phys = self._env._env._state.qp
                self._current_obs = obs

            while self._traj_steps < self.max_traj_length:
                self._traj_steps += 1
                rng, split_rng = jax.random.split(rng)
                action = policy(
                    split_rng,
                    self._current_obs,
                    deterministic=deterministic,
                    random=random,
                ).reshape(-1)
                obs, reward, done, info = self._env.step(action)
                r_traj += reward

                if self._video_recorder is not None:
                    self._video_recorder.record(self._env)

                if data_storage is not None:
                    data_storage.add(self._current_obs, action, obs, reward, done, self._current_phys)

                self._current_obs = obs
                self._current_phys = self._env._env._state.qp
                self._done = done

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
        data_storage=None,
        random=False,
    ):
        for _ in range(n_steps):
            if self._done or self._traj_steps >= self.max_traj_length:
                self._traj_steps = 0

                (obs, reward, done, info), self._done = self._reset_fn(), False

                if self._video_recorder is not None:
                    self._video_recorder.init(self._env)
                    self._video_recorder.record(self._env)

                self._current_phys = self._env._env._state.qp
                self._current_obs = obs

            self._traj_steps += 1
            rng, split_rng = jax.random.split(rng)
            action = policy(
                split_rng,
                self._current_obs,
                deterministic=deterministic,
                random=random,
            ).reshape(-1)
            obs, reward, done, info = self._env.step(action)

            if self._video_recorder is not None:
                self._video_recorder.record(self._env)

            if data_storage is not None:
                data_storage.add(self._current_obs, action, obs, reward, done, self._current_phys)

            self._current_obs = obs
            self._current_phys = self._env._env._state.qp
            self._done = done

        data = dict()
        return data, rng
