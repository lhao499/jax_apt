import math

import jax
import jax.numpy as jnp
import numpy as np


def cubify(arr, newshape):
    """
    Breaks up an n-dimensional array of shape (D_1, D_2, ..., D_n) into N blocks of shape (d_1, d_2, ..., d_n).
    Each block dimension d_i must divide the corresponding array dimension D_i without remainder.
    Args:
        arr: A array of shape (D_1, D_2, ..., D_n).
        newshape: (d_1, d_2, ..., d_n) describing the shape of each of the N blocks.
    Returns:
        A new array of shape (N, d_1, d_2, ..., d_n).
    """
    oldshape = jnp.array(arr.shape)
    newshape = jnp.array(newshape)
    repeats = (oldshape / newshape).astype(int)
    tmpshape = jnp.column_stack([repeats, newshape]).ravel()
    order = jnp.arange(len(tmpshape))
    order = jnp.concatenate([order[::2], order[1::2]])
    # newshape must divide oldshape evenly or else ValueError will be raised
    return arr.reshape(tmpshape).transpose(order).reshape(-1, *newshape)


def decubify(arr, oldshape):
    """
    Reassembles N blocks of shape (d_1, d_2, ..., d_n) into their original array of shape (D_1, D_2, ..., D_n).
    Args:
        arr: A array of shape (N, d_1, d_2, ..., d_n).
        oldshape: A tuple (D_1, D_2, ..., D_n) describing the shape of the original array.
    Returns:
        The reconstructed array of shape (D_1, D_2, ..., D_n).
    """
    newshape = arr.shape[1:]
    oldshape = jnp.array(oldshape)
    repeats = (oldshape / newshape).astype(int)
    tmpshape = jnp.concatenate([repeats, newshape])
    order = jnp.arange(len(tmpshape)).reshape(2, -1).ravel(order="F")
    return arr.reshape(tmpshape).transpose(order).reshape(oldshape)


def zigzag_indices(n):
    """
    Computes the zig zag scan indices for an nxn matrix, useful for selecting "most important" DCT coefficients.
    Args:
        n: Matrix side length
    Returns:
        List of lists, containing index pairs in sorted zig zag scan order
    """

    def compare(xy):
        x, y = xy
        return (x + y, -y if (x + y) % 2 else y)

    xs = range(n)
    indices = [index for index in sorted(([x, y] for x in xs for y in xs), key=compare)]
    return jnp.asarray(indices).transpose(1, 0).tolist()


def zigzagify(X, indices):
    """
    Flattens the last two dimensions of a array in a zig zag scan manner.

    Args:
        X: Array of arbitrary shape, with last two dimensions having the same dimensions
        indices: Precomputed zig zag scan indices
    Returns:
        Array where last two dimensions were flattened into one
    """
    return X[..., indices[0], indices[1]]


def dezigzagify(X, indices):
    """
    Undoes the zig zag scan operation and reconstructs a square matrix in the last two dimensions.
    Args:
        X: Array or arbitrary shape, where the last dimension are zig zag scan flattened. Last dimension
            may be smaller than the square of the original side length. The remaining coefficients are
            replaced by zeros.
        indices: Precomputed zig zag scan indices
    Returns:
        Array where last two dimensions are reconstructed from zig zag flattened vector
    """
    img_size = int(math.sqrt(len(indices[0])))
    Y = jnp.zeros(list(X.shape[:-1]) + [img_size, img_size])
    pad_X = jnp.pad(X, ((0, 0), (0, 0), (0, len(indices[0]) - X.shape[-1])), "constant")
    Y.at[..., jnp.asarray(indices[0]), jnp.asarray(indices[1])].set(pad_X)
    return Y


def dct_patch(img, patch_size, n_keep):
    """
    Performs 2D Discrete Cosine Transform on a patch-wise and channel-wise basis over an entire image.
    """
    n_channel, img_size, img_size = img.shape  # C x H x W
    img_patch = cubify(img, (n_channel, patch_size, patch_size))  # (NxN) x C x P x P
    coeff = jax.scipy.fft.dctn(img_patch, type=2, axes=[-1, -2])  # (NxN) x C x P x P
    coeff = zigzagify(coeff, zigzag_indices(patch_size))[..., :n_keep]  # (NxN) x C x M
    n_patch = img_size // patch_size
    coeff = coeff.reshape(n_channel * n_keep, n_patch, n_patch)  # (CxM) x P x P
    return coeff


@jax.jit
def batch_dct_patch(img, patch_size, n_keep):
    return jax.vmap(dct_patch, (0, None, None))(img, patch_size, n_keep)


def idct_patch(coeff, patch_size, n_keep, n_channel, img_size):
    """
    Performs an inverse patch-wise 2D Discrete Cosine Transform.

    Args:
        coeff: DCT coefficients of shape (num_channels*num_components) x num_patches x num_patches
        n_keep: How many DCT coefficients were returned by DCT per patch and per channel
        n_channel: Number of color channels in the original image
        img_size: Side length of the original image
    Returns:
        Reconstructed image of shape num_channels x img_size x img_size
    """
    coeff = coeff.reshape(n_channel, n_keep, -1).transpose([2, 0, 1])  # (NxN) x C x M
    coeff = dezigzagify(coeff, zigzag_indices(patch_size))  # (NxN) x C x P x P
    reconst = jax.scipy.fft.dctn(coeff, type=3, axes=[-1, -2])  # (NxN) x C x P x P
    reconst = decubify(reconst, (n_channel, img_size, img_size))  # C x H x W
    return reconst


if __name__ == "__main__":
    img = np.random.normal(size=(3, 64, 64))
    img = jnp.asarray(img)
    new_img = idct_patch(dct_patch(img, 8, 5), 8, 5, 3, 64)
    res = jnp.allclose(new_img, img)
    import ipdb

    ipdb.set_trace()
