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
from jax.scipy import special as jsp

from brainstate import environ


def _categorical(key, p, shape):
    # this implementation is fast when event shape is small, and slow otherwise
    # Ref: https://stackoverflow.com/a/34190035
    shape = shape or p.shape[:-1]
    s = jnp.cumsum(p, axis=-1)
    r = jr.uniform(key, shape=shape + (1,))
    return jnp.sum(s < r, axis=-1)


@partial(jit, static_argnames=('n_max', 'shape'))
def multinomial(key, p, n, *, n_max, shape=()):
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
        excess = jnp.concatenate(
            [jnp.expand_dims(n_max - n, -1),
             jnp.zeros(u.math.shape(n) + (p.shape[-1] - 1,))],
            -1
        )
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

    def cond_fn(*args):
        """check if all are done or reached max number of iterations"""
        i, _, done, _, _ = args[0]
        return jnp.bitwise_and(i < 100, jnp.logical_not(jnp.all(done)))

    def body_fn(*args):
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


def _scatter_add_one(operand, indices, updates):
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


def _dtype(x, *, canonicalize: bool = False):
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


def formalize_key(key, use_prng_key=True):
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


def _check_shape(name, shape, *param_shapes):
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


@partial(jit, static_argnames=['shape', 'dtype'])
def f(
    key,
    dfnum,
    dfden,
    *,
    shape,
    dtype=None
):
    """Draw samples from the central F distribution."""
    dtype = dtype or environ.dftype()
    dfnum = lax.convert_element_type(dfnum, dtype)
    dfden = lax.convert_element_type(dfden, dtype)

    if shape is None:
        shape = lax.broadcast_shapes(u.math.shape(dfnum), u.math.shape(dfden))
    elif isinstance(shape, int):
        shape = (shape,)
    else:
        shape = tuple(shape)

    dfnum = jnp.broadcast_to(dfnum, shape)
    dfden = jnp.broadcast_to(dfden, shape)

    size = int(np.prod(shape)) if shape else 1
    if size == 0:
        return jnp.empty(shape, dtype=dtype)

    key_num, key_den = jr.split(key)
    chi2_num = 2.0 * jr.gamma(key_num, 0.5 * dfnum, shape=shape, dtype=dtype)
    chi2_den = 2.0 * jr.gamma(key_den, 0.5 * dfden, shape=shape, dtype=dtype)

    return (chi2_num / dfnum) / (chi2_den / dfden)


@partial(jit, static_argnames=['shape', 'dtype'])
def noncentral_f(
    key,
    dfnum,
    dfden,
    nonc,
    *,
    shape,
    dtype=None
):
    """
    Draw samples from the noncentral F distribution.

    The noncentral F distribution is a generalization of the F distribution.
    It is parameterized by dfnum (degrees of freedom of the numerator),
    dfden (degrees of freedom of the denominator), and nonc (noncentrality parameter).

    The implementation uses the relationship:
    If X ~ noncentral_chisquare(dfnum, nonc) and Y ~ chisquare(dfden), then
    F = (X / dfnum) / (Y / dfden) ~ noncentral_f(dfnum, dfden, nonc)

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key
    dfnum : float or array_like
        Degrees of freedom of the numerator, must be > 0
    dfden : float or array_like
        Degrees of freedom of the denominator, must be > 0
    nonc : float or array_like
        Noncentrality parameter, must be >= 0
    shape : tuple
        Output shape
    dtype : dtype, optional
        Data type of the output

    Returns
    -------
    out : array_like
        Samples from the noncentral F distribution
    """
    dtype = dtype or environ.dftype()
    dfnum = lax.convert_element_type(dfnum, dtype)
    dfden = lax.convert_element_type(dfden, dtype)
    nonc = lax.convert_element_type(nonc, dtype)

    # Split key for two random samples
    key1, key2 = jr.split(key)

    # Generate noncentral chi-square for numerator
    # noncentral_chisquare(df, nonc) = chi-square(df - 1) + (normal(0,1) + sqrt(nonc))^2
    # when df > 1, else chi-square(df + 2*poisson(nonc/2))
    keys_numer = jr.split(key1, 3)
    i = jr.poisson(keys_numer[0], 0.5 * nonc, shape=shape, dtype=environ.ditype())
    n = jr.normal(keys_numer[1], shape=shape, dtype=dtype) + jnp.sqrt(nonc)
    cond = jnp.greater(dfnum, 1.0)
    df_numerator = jnp.where(cond, dfnum - 1.0, dfnum + 2.0 * i)
    chi2_numerator = 2.0 * jr.gamma(keys_numer[2], 0.5 * df_numerator, shape=shape, dtype=dtype)
    numerator = jnp.where(cond, chi2_numerator + n * n, chi2_numerator)

    # Generate central chi-square for denominator
    # chi-square(df) = 2 * gamma(df/2, 1)
    chi2_denominator = 2.0 * jr.gamma(key2, 0.5 * dfden, shape=shape, dtype=dtype)

    # Compute F statistic: (numerator / dfnum) / (denominator / dfden)
    f_stat = (numerator / dfnum) / (chi2_denominator / dfden)

    return f_stat


@partial(jit, static_argnames=['shape', 'dtype'])
def logseries(
    key,
    p,
    *,
    shape,
    dtype=None
):
    """Draw samples from the logarithmic series distribution."""
    dtype = dtype or environ.ditype()
    float_dtype = dtypes.canonicalize_dtype(environ.dftype())
    calc_dtype = dtypes.canonicalize_dtype(jnp.promote_types(float_dtype, jnp.float64))

    p = lax.convert_element_type(p, float_dtype)

    if shape is None:
        shape = u.math.shape(p)
    elif isinstance(shape, int):
        shape = (shape,)
    else:
        shape = tuple(shape)

    p = jnp.broadcast_to(p, shape)

    size = int(np.prod(shape)) if shape else 1
    if size == 0:
        return jnp.empty(shape, dtype=dtype)

    p_flat = jnp.reshape(lax.convert_element_type(p, calc_dtype), (size,))
    keys = jr.split(key, size)

    tiny = jnp.array(np.finfo(calc_dtype).tiny, dtype=calc_dtype)
    one_minus_eps = jnp.nextafter(jnp.array(1.0, dtype=calc_dtype), jnp.array(0.0, dtype=calc_dtype))

    def _sample_one(single_key, p_scalar):
        p_scalar = lax.convert_element_type(p_scalar, calc_dtype)
        operand = (single_key, p_scalar)

        def _limit_case(_):
            return jnp.array(1.0, dtype=calc_dtype)

        def _positive_case(args):
            key_i, p_val = args
            p_val = jnp.clip(p_val, tiny, one_minus_eps)
            log_p = jnp.log(p_val)
            log_norm = jnp.log(-jnp.log1p(-p_val))
            log_prob = log_p - log_norm
            log_cdf = log_prob
            log_u = jnp.log(jr.uniform(key_i, shape=(), dtype=calc_dtype, minval=tiny, maxval=one_minus_eps))

            init_state = (jnp.array(1.0, dtype=calc_dtype), log_prob, log_cdf, log_u)

            def cond_fn(state):
                _, _, log_cdf_val, log_u_val = state
                return log_u_val > log_cdf_val

            def body_fn(state):
                k_val, log_prob_val, log_cdf_val, log_u_val = state
                k_next = k_val + 1.0
                log_prob_next = log_prob_val + log_p + jnp.log(k_val) - jnp.log(k_next)
                log_cdf_next = jnp.logaddexp(log_cdf_val, log_prob_next)
                return k_next, log_prob_next, log_cdf_next, log_u_val

            k_val, _, _, _ = lax.while_loop(cond_fn, body_fn, init_state)
            return k_val

        return lax.cond(p_scalar <= 0.0, _limit_case, _positive_case, operand)

    samples = vmap(_sample_one)(keys, p_flat)
    samples = lax.convert_element_type(samples, dtype)
    return jnp.reshape(samples, shape)


@partial(jit, static_argnames=['shape', 'dtype'])
def zipf(
    key,
    a,
    *,
    shape,
    dtype=None
):
    """Draw samples from the Zipf (zeta) distribution."""
    dtype = dtype or environ.ditype()
    float_dtype = dtypes.canonicalize_dtype(environ.dftype())
    calc_dtype = dtypes.canonicalize_dtype(jnp.promote_types(float_dtype, jnp.float64))

    a = lax.convert_element_type(a, calc_dtype)

    if shape is None:
        shape = u.math.shape(a)
    elif isinstance(shape, int):
        shape = (shape,)
    else:
        shape = tuple(shape)

    a = jnp.broadcast_to(a, shape)

    size = int(np.prod(shape)) if shape else 1
    if size == 0:
        return jnp.empty(shape, dtype=dtype)

    u_ = jr.uniform(
        key,
        shape=shape,
        dtype=calc_dtype,
        minval=jnp.finfo(calc_dtype).tiny,
        maxval=jnp.array(1.0, dtype=calc_dtype)
    )

    a_flat = jnp.reshape(a, (size,))
    u_flat = jnp.reshape(u_, (size,))

    max_iters = jnp.array(1000000, dtype=jnp.int32)

    def _sample_one(a_scalar, u_scalar):
        norm = jsp.zeta(a_scalar, jnp.array(1.0, dtype=calc_dtype))

        def cdf(k_val):
            return (
                jnp.array(1.0, dtype=calc_dtype) -
                jsp.zeta(a_scalar, k_val + jnp.array(1.0, dtype=calc_dtype)) / norm
            )

        initial = jnp.array(1.0, dtype=calc_dtype)
        cdf_prev = jnp.array(0.0, dtype=calc_dtype)
        cdf_curr = cdf(initial)

        state = (
            initial,
            cdf_prev,
            cdf_curr,
            jnp.array(0, dtype=jnp.int32)
        )

        def cond_fn(state):
            _, c_prev, c_curr, it = state
            not_ok = jnp.logical_or(u_scalar > c_curr, u_scalar <= c_prev)
            return jnp.logical_and(not_ok, it < max_iters)

        def body_fn(state):
            k_val, c_prev, c_curr, it = state
            need_increase = u_scalar > c_curr

            def inc(_):
                k_next = k_val + jnp.array(1.0, dtype=calc_dtype)
                c_prev_next = jnp.array(1.0, dtype=calc_dtype) - jsp.zeta(a_scalar, k_next) / norm
                c_curr_next = cdf(k_next)
                return k_next, c_prev_next, c_curr_next, it + 1

            def dec(_):
                k_next = jnp.maximum(jnp.array(1.0, dtype=calc_dtype), k_val - jnp.array(1.0, dtype=calc_dtype))
                c_prev_next = jnp.array(1.0, dtype=calc_dtype) - jsp.zeta(a_scalar, k_next) / norm
                c_curr_next = cdf(k_next)
                return k_next, c_prev_next, c_curr_next, it + 1

            return lax.cond(need_increase, inc, dec, operand=None)

        k_final, _, _, _ = lax.while_loop(cond_fn, body_fn, state)
        return lax.convert_element_type(k_final, dtype)

    samples_flat = jax.vmap(_sample_one)(a_flat, u_flat)
    samples = jnp.reshape(samples_flat, shape)
    return samples


@partial(jit, static_argnames=['shape', 'dtype'])
def power(
    key,
    a,
    *,
    shape,
    dtype=None
):
    """Draw samples from the power distribution."""
    dtype = dtype or environ.dftype()
    float_dtype = dtypes.canonicalize_dtype(dtype)

    a = lax.convert_element_type(a, float_dtype)

    if shape is None:
        shape = u.math.shape(a)
    elif isinstance(shape, int):
        shape = (shape,)
    else:
        shape = tuple(shape)

    a = jnp.broadcast_to(a, shape)

    size = int(np.prod(shape)) if shape else 1
    if size == 0:
        return jnp.empty(shape, dtype=float_dtype)

    eps = jnp.array(np.finfo(float_dtype).tiny, dtype=float_dtype)
    a_safe = jnp.maximum(a, eps)

    u_ = jr.uniform(key, shape=shape, dtype=float_dtype, minval=eps, maxval=1.0)
    samples = jnp.power(u_, jnp.reciprocal(a_safe))

    return lax.convert_element_type(samples, dtype)


@partial(jit, static_argnames=['shape', 'dtype'])
def hypergeometric(
    key,
    ngood,
    nbad,
    nsample,
    *,
    shape,
    dtype=None
):
    """Draw samples from the hypergeometric distribution."""
    dtype = dtype or environ.ditype()
    out_dtype = dtypes.canonicalize_dtype(dtype)
    float_dtype = dtypes.canonicalize_dtype(environ.dftype())
    calc_dtype = dtypes.canonicalize_dtype(jnp.promote_types(float_dtype, jnp.float64))

    ngood = lax.convert_element_type(ngood, out_dtype)
    nbad = lax.convert_element_type(nbad, out_dtype)
    nsample = lax.convert_element_type(nsample, out_dtype)

    if shape is None:
        shape = lax.broadcast_shapes(u.math.shape(ngood), u.math.shape(nbad), u.math.shape(nsample))
    elif isinstance(shape, int):
        shape = (shape,)
    else:
        shape = tuple(shape)

    ngood = jnp.broadcast_to(ngood, shape)
    nbad = jnp.broadcast_to(nbad, shape)
    nsample = jnp.broadcast_to(nsample, shape)

    size = int(np.prod(shape)) if shape else 1
    if size == 0:
        return jnp.empty(shape, dtype=out_dtype)

    flat_ngood = jnp.reshape(ngood, (size,))
    flat_nbad = jnp.reshape(nbad, (size,))
    flat_nsample = jnp.reshape(nsample, (size,))
    sample_keys = jr.split(key, size + 1)[1:]

    one = jnp.array(1, dtype=out_dtype)
    zero = jnp.array(0, dtype=out_dtype)

    def _sample_one(sample_key, good, bad, draws):
        good = jnp.maximum(good, zero)
        bad = jnp.maximum(bad, zero)
        draws = jnp.maximum(draws, zero)
        total = good + bad
        draws = jnp.minimum(draws, total)

        init_state = (zero, sample_key, good, bad, zero, draws)

        def cond_fn(state):
            i, _, good_i, bad_i, _, draws_i = state
            total_i = good_i + bad_i
            return jnp.logical_and(i < draws_i, total_i > zero)

        def body_fn(state):
            i, key_i, good_i, bad_i, succ_i, draws_i = state
            key_i, subkey = jr.split(key_i)
            total_i = good_i + bad_i
            prob = jnp.where(
                total_i > zero,
                lax.convert_element_type(good_i, calc_dtype) / lax.convert_element_type(total_i, calc_dtype),
                jnp.array(0.0, dtype=calc_dtype),
            )
            u = jr.uniform(subkey, shape=(), dtype=calc_dtype)
            success = (u < prob).astype(out_dtype)
            good_i = good_i - success
            bad_i = bad_i - jnp.where(total_i > zero, one - success, zero)
            succ_i = succ_i + success
            return (i + one, key_i, good_i, bad_i, succ_i, draws_i)

        _, _, _, _, successes, _ = lax.while_loop(cond_fn, body_fn, init_state)
        return successes

    samples = jax.vmap(_sample_one)(sample_keys, flat_ngood, flat_nbad, flat_nsample)
    samples = lax.convert_element_type(samples, out_dtype)
    return jnp.reshape(samples, shape)
