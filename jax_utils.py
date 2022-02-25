import jax
import jax.numpy as jnp


class JaxRNG(object):
    def __init__(self, seed):
        self.rng = jax.random.PRNGKey(seed)

    def __call__(self):
        self.rng, next_rng = jax.random.split(self.rng)
        return next_rng


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
    return {k: jax.device_put(v) for k, v in batch.items()}


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


class RMS(object):
    def __init__(self, epsilon=1e-4, shape=(1,)):
        self.M = jnp.zeros(shape)
        self.S = jnp.ones(shape)
        self.n = epsilon

    @jax.jit
    def __call__(self, x):
        bs = x.size(0)
        delta = jnp.mean(x, axis=0) - self.M
        new_M = self.M + delta * bs / (self.n + bs)
        new_S = (
            self.S * self.n
            + jnp.var(x, axis=0) * bs
            + jnp.square(delta) * self.n * bs / (self.n + bs)
        ) / (self.n + bs)

        self.M = new_M
        self.S = new_S
        self.n += bs

        return self.M, self.S


class PBE(object):
    def __init__(self, rms, knn_clip, knn_k, knn_avg, knn_rms):
        self.rms = rms
        self.knn_rms = knn_rms
        self.knn_k = knn_k
        self.knn_avg = knn_avg
        self.knn_clip = knn_clip

    def __call__(self, data):
        # (b, 1, c) - (1, b, c) -> (b1, b2)
        bs = data.shape[0]
        dist = jnp.linalg.norm(
            data[:, None, :].reshape((bs, 1, -1))
            - data[None, :, :].reshape((1, bs, -1)),
            axis=-1,
        )
        neg_dist, _ = jax.lax.top_k(-dist, self.knn_k)  # (b, k)
        dist = -neg_dist
        sort_dist = jax.lax.sort(dist, dimension=-1)
        # k-th nearest neighbor
        if not self.knn_avg:
            entropy = sort_dist[:, -1].reshape(-1, 1)  # (b, 1)
            entropy /= self.rms(entropy)[0] if self.knn_rms else 1.0
            if self.knn_clip >= 0.0:
                entropy = jnp.maximum(entropy - self.knn_clip, jnp.zeros_like(entropy))
        # average k nearest neighbors
        else:
            entropy = entropy.reshape(-1, 1)  # (b * k, 1)
            entropy /= self.rms(entropy)[0] if self.knn_rms else 1.0
            if self.knn_clip >= 0.0:
                entropy = jnp.maximum(entropy - self.knn_clip, jnp.zeros_like(entropy))
            entropy = entropy.reshape((bs, self.knn_k))  # (b, k)
            entropy = entropy.mean(dim=1, keepdim=True)  # (b, 1)
        reward = jnp.log(entropy + 1.0)
        return reward


def nonparametric_entropy(data, knn_k, knn_avg=False, knn_log=True, knn_clip=0.0):
    data = data / jnp.linalg.norm(data, axis=-1, keepdims=True)
    bs = data.shape[0]
    dist = jnp.linalg.norm(
        data[:, None, :].reshape((bs, 1, -1)) - data[None, :, :].reshape((1, bs, -1)),
        axis=-1,
    )
    neg_dist, _ = jax.lax.top_k(-dist, knn_k)  # (b, k)
    dist = -neg_dist
    sort_dist = jax.lax.sort(dist, dimension=-1)
    # k-th nearest neighbor
    if not knn_avg:
        entropy = sort_dist[:, -1].reshape(-1, 1)  # (b, 1)
        if knn_clip > 0.0:
            entropy = jnp.maximum(entropy - knn_clip, jnp.zeros_like(entropy))
    # average k nearest neighbors
    else:
        entropy = sort_dist.reshape(-1, 1)  # (b * k, 1)
        if knn_clip > 0.0:
            entropy = jnp.maximum(entropy - knn_clip, jnp.zeros_like(entropy))
        entropy = entropy.reshape((bs, knn_k))  # (b, k)
        entropy = entropy.mean(axis=1, keepdims=True)  # (b, 1)
    if knn_log:
        reward = jnp.log(entropy + 1.0)
    else:
        reward = entropy
    return reward
