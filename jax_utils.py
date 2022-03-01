import jax
import jax.numpy as jnp
from functools import partial

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
    return jax.tree_map(jax.device_put, batch)


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


def _dist_fn(x, y):
    return jnp.sum(jnp.square(x - y))


@jax.jit
def distance_fn(a, b, metric=_dist_fn):
    return jax.vmap(jax.vmap(metric, (None, 0)), (0, None))(a, b)


@partial(jax.jit, static_argnums=[1, 2, 3, 4])
def nonparametric_entropy(data, knn_k=512, knn_avg=True, knn_log=False, knn_clip=0.0):
    knn_k = min(knn_k, data.shape[0])
    neg_distance = -distance_fn(data, data)
    neg_distance, indices = jax.lax.top_k(neg_distance, k=knn_k)
    distance = -neg_distance
    if knn_avg: # averaging
        entropy = distance.reshape(-1, 1)  # (b * k, 1)
        if knn_clip > 0.0:
            entropy = jnp.maximum(entropy - knn_clip, jnp.zeros_like(entropy))
        entropy = entropy.reshape((data.shape[0], knn_k))  # (b, k)
        entropy = entropy.mean(axis=1, keepdims=True)  # (b, 1)
    else:
        distance = jax.lax.sort(distance, dimension=-1)
        entropy = distance[:, -1].reshape(-1, 1)  # (b, 1)
        if knn_clip > 0.0:
            entropy = jnp.maximum(entropy - knn_clip, jnp.zeros_like(entropy))

    if knn_log: # rescaling
        reward = jnp.log(entropy + 1.0)
    else:
        reward = entropy
    return reward
