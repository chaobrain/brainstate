# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from functools import partial

import brainunit as u
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import jit, vmap
from jax import lax, dtypes

from brainstate import environ


def _categorical(key, p, shape):
    # this implementation is fast when event shape is small, and slow otherwise
    # Ref: https://stackoverflow.com/a/34190035
    shape = shape or p.shape[:-1]
    s = jnp.cumsum(p, axis=-1)
    r = jr.uniform(key, shape=shape + (1,))
    return jnp.sum(s < r, axis=-1)


@partial(jit, static_argnums=(3, 4))
def multinomial(
    key,
    p,
    n,
    n_max,
    shape=()
):
    if u.math.shape(n) != u.math.shape(p)[:-1]:
        broadcast_shape = lax.broadcast_shapes(u.math.shape(n), u.math.shape(p)[:-1])
        n = jnp.broadcast_to(n, broadcast_shape)
        p = jnp.broadcast_to(p, broadcast_shape + u.math.shape(p)[-1:])
    shape = shape or p.shape[:-1]
    if n_max == 0:
        return jnp.zeros(shape + p.shape[-1:], dtype=jnp.result_type(int))
    # get indices from categorical distribution then gather the result
    indices = _categorical(key, p, (n_max,) + shape)
    # mask out values when counts is heterogeneous
    if jnp.ndim(n) > 0:
        mask = _promote_shapes(jnp.arange(n_max) < jnp.expand_dims(n, -1), shape=shape + (n_max,))[0]
        mask = jnp.moveaxis(mask, -1, 0).astype(indices.dtype)
        excess = jnp.concatenate([jnp.expand_dims(n_max - n, -1),
                                  jnp.zeros(u.math.shape(n) + (p.shape[-1] - 1,))],
                                 -1)
    else:
        mask = 1
        excess = 0
    # NB: we transpose to move batch shape to the front
    indices_2D = (jnp.reshape(indices * mask, (n_max, -1))).T
    samples_2D = vmap(_scatter_add_one)(
        jnp.zeros((indices_2D.shape[0], p.shape[-1]), dtype=indices.dtype),
        jnp.expand_dims(indices_2D, axis=-1),
        jnp.ones(indices_2D.shape, dtype=indices.dtype)
    )
    return jnp.reshape(samples_2D, shape + p.shape[-1:]) - excess


@partial(jit, static_argnums=(2, 3), static_argnames=['shape', 'dtype'])
def von_mises_centered(
    key,
    concentration,
    shape,
    dtype=None
):
    """Compute centered von Mises samples using rejection sampling from [1]_ with wrapped Cauchy proposal.

    Returns
    -------
    out: array_like
       centered samples from von Mises

    References
    ----------
    .. [1] Luc Devroye "Non-Uniform Random Variate Generation", Springer-Verlag, 1986;
           Chapter 9, p. 473-476. http://www.nrbook.com/devroye/Devroye_files/chapter_nine.pdf

    """
    shape = shape or u.math.shape(concentration)
    dtype = dtype or environ.dftype()
    concentration = lax.convert_element_type(concentration, dtype)
    concentration = jnp.broadcast_to(concentration, shape)

    if dtype == jnp.float16:
        s_cutoff = 1.8e-1
    elif dtype == jnp.float32:
        s_cutoff = 2e-2
    elif dtype == jnp.float64:
        s_cutoff = 1.2e-4
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    r = 1.0 + jnp.sqrt(1.0 + 4.0 * concentration ** 2)
    rho = (r - jnp.sqrt(2.0 * r)) / (2.0 * concentration)
    s_exact = (1.0 + rho ** 2) / (2.0 * rho)

    s_approximate = 1.0 / concentration

    s = jnp.where(concentration > s_cutoff, s_exact, s_approximate)

    def cond_fn(
        *args
    ):
        """check if all are done or reached max number of iterations"""
        i, _, done, _, _ = args[0]
        return jnp.bitwise_and(i < 100, jnp.logical_not(jnp.all(done)))

    def body_fn(
        *args
    ):
        i, key, done, _, w = args[0]
        uni_ukey, uni_vkey, key = jr.split(key, 3)
        u_ = jr.uniform(
            key=uni_ukey,
            shape=shape,
            dtype=concentration.dtype,
            minval=-1.0,
            maxval=1.0,
        )
        z = jnp.cos(jnp.pi * u_)
        w = jnp.where(done, w, (1.0 + s * z) / (s + z))  # Update where not done
        y = concentration * (s - w)
        v = jr.uniform(key=uni_vkey, shape=shape, dtype=concentration.dtype)
        accept = (y * (2.0 - y) >= v) | (jnp.log(y / v) + 1.0 >= y)
        return i + 1, key, accept | done, u_, w

    init_done = jnp.zeros(shape, dtype=bool)
    init_u = jnp.zeros(shape)
    init_w = jnp.zeros(shape)

    _, _, done, uu, w = lax.while_loop(
        cond_fun=cond_fn,
        body_fun=body_fn,
        init_val=(jnp.array(0), key, init_done, init_u, init_w),
    )

    return jnp.sign(uu) * jnp.arccos(w)


def _scatter_add_one(
    operand,
    indices,
    updates
):
    return lax.scatter_add(
        operand,
        indices,
        updates,
        lax.ScatterDimensionNumbers(
            update_window_dims=(),
            inserted_window_dims=(0,),
            scatter_dims_to_operand_dims=(0,),
        ),
    )


def _reshape(x, shape):
    if isinstance(x, (int, float, np.ndarray, np.generic)):
        return np.reshape(x, shape)
    else:
        return jnp.reshape(x, shape)


def _promote_shapes(*args, shape=()):
    # adapted from lax.lax_numpy
    if len(args) < 2 and not shape:
        return args
    else:
        shapes = [u.math.shape(arg) for arg in args]
        num_dims = len(lax.broadcast_shapes(shape, *shapes))
        return [
            _reshape(arg, (1,) * (num_dims - len(s)) + s)
            if len(s) < num_dims else arg
            for arg, s in zip(args, shapes)
        ]


python_scalar_dtypes = {
    bool: np.dtype('bool'),
    int: np.dtype('int64'),
    float: np.dtype('float64'),
    complex: np.dtype('complex128'),
}


def _dtype(
    x,
    *,
    canonicalize: bool = False
):
    """Return the dtype object for a value or type, optionally canonicalized based on X64 mode."""
    if x is None:
        raise ValueError(f"Invalid argument to dtype: {x}.")
    elif isinstance(x, type) and x in python_scalar_dtypes:
        dt = python_scalar_dtypes[x]
    elif type(x) in python_scalar_dtypes:
        dt = python_scalar_dtypes[type(x)]
    elif hasattr(x, 'dtype'):
        dt = x.dtype
    else:
        dt = np.result_type(x)
    return dtypes.canonicalize_dtype(dt) if canonicalize else dt


def _is_python_scalar(x):
    if hasattr(x, 'aval'):
        return x.aval.weak_type
    elif np.ndim(x) == 0:
        return True
    elif isinstance(x, (bool, int, float, complex)):
        return True
    else:
        return False


def const(example, val):
    if _is_python_scalar(example):
        dtype = dtypes.canonicalize_dtype(type(example))
        val = dtypes.scalar_type_of(example)(val)
        return val if dtype == _dtype(val, canonicalize=True) else np.array(val, dtype)
    else:
        dtype = dtypes.canonicalize_dtype(example.dtype)
    return np.array(val, dtype)


# ---------------------------------------------------------------------------------------------------------------


def _formalize_key(key):
    if isinstance(key, int):
        return jr.PRNGKey(key) if use_prng_key else jr.key(key)
    elif isinstance(key, (jax.Array, np.ndarray)):
        if jnp.issubdtype(key.dtype, jax.dtypes.prng_key):
            return key
        if key.size == 1 and jnp.issubdtype(key.dtype, jnp.integer):
            return jr.PRNGKey(key) if use_prng_key else jr.key(key)

        if key.dtype != jnp.uint32:
            raise TypeError('key must be a int or an array with two uint32.')
        if key.size != 2:
            raise TypeError('key must be a int or an array with two uint32.')
        return u.math.asarray(key, dtype=jnp.uint32)
    else:
        raise TypeError('key must be a int or an array with two uint32.')


def _size2shape(size):
    if size is None:
        return ()
    elif isinstance(size, (tuple, list)):
        return tuple(size)
    else:
        return (size,)


def _check_shape(
    name,
    shape,
    *param_shapes
):
    if param_shapes:
        shape_ = lax.broadcast_shapes(shape, *param_shapes)
        if shape != shape_:
            msg = ("{} parameter shapes must be broadcast-compatible with shape "
                   "argument, and the result of broadcasting the shapes must equal "
                   "the shape argument, but got result {} for shape argument {}.")
            raise ValueError(msg.format(name, shape_, shape))


def _loc_scale(
    loc,
    scale,
    value
):
    if loc is None:
        if scale is None:
            return value
        else:
            return value * scale
    else:
        if scale is None:
            return value + loc
        else:
            return value * scale + loc


def _check_py_seq(seq):
    return u.math.asarray(seq) if isinstance(seq, (tuple, list)) else seq
