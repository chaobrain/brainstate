# Copyright 2024 BDP Ecosystem Limited. All Rights Reserved.
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

# -*- coding: utf-8 -*-

from functools import partial
from operator import index
from typing import Optional

import brainunit as u
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import jit, vmap
from jax import lax, core, dtypes

from brainstate import environ
from brainstate._state import State
from brainstate.compile._error_if import jit_error_if
from brainstate.typing import DTypeLike, Size, SeedOrKey
from ._random_for_unit import uniform_for_unit, permutation_for_unit

__all__ = ['RandomState', 'DEFAULT', ]

use_prng_key = True


class RandomState(State):
    """RandomState that track the random generator state. """

    # __slots__ = ('_backup', '_value')

    def __init__(self, seed_or_key: Optional[SeedOrKey] = None):
        """RandomState constructor.

        Parameters
        ----------
        seed_or_key: int, Array, optional
          It can be an integer for initial seed of the random number generator,
          or it can be a JAX's PRNKey, which is an array with two elements and `uint32` dtype.
        """
        with jax.ensure_compile_time_eval():
            if seed_or_key is None:
                seed_or_key = np.random.randint(0, 100000, 2, dtype=np.uint32)
        if isinstance(seed_or_key, int):
            key = jr.PRNGKey(seed_or_key) if use_prng_key else jr.key(seed_or_key)
        else:
            if jnp.issubdtype(seed_or_key.dtype, jax.dtypes.prng_key):
                key = seed_or_key
            else:
                if len(seed_or_key) != 2 and seed_or_key.dtype != np.uint32:
                    raise ValueError('key must be an array with dtype uint32. '
                                     f'But we got {seed_or_key}')
                key = seed_or_key
        super().__init__(key)

        self._backup = None

    def __repr__(self):
        return f'{self.__class__.__name__}({self.value})'

    def check_if_deleted(self):
        if not use_prng_key and isinstance(self._value, np.ndarray):
            self._value = jr.key(np.random.randint(0, 10000))

        if (
            isinstance(self._value, jax.Array) and
            not isinstance(self._value, jax.core.Tracer) and
            self._value.is_deleted()
        ):
            self.seed()

    # ------------------- #
    # seed and random key #
    # ------------------- #

    def backup_key(self):
        if self._backup is not None:
            raise ValueError('The random key has been backed up, and has not been restored.')
        self._backup = self.value

    def restore_key(self):
        if self._backup is None:
            raise ValueError('The random key has not been backed up.')
        self.value = self._backup
        self._backup = None

    def clone(self):
        return type(self)(self.split_key())

    def set_key(self, key: SeedOrKey):
        self.value = key

    def seed(self, seed_or_key: Optional[SeedOrKey] = None):
        """Sets a new random seed.

        Parameters
        ----------
        seed_or_key: int, ArrayLike, optional
          It can be an integer for initial seed of the random number generator,
          or it can be a JAX's PRNKey, which is an array with two elements and `uint32` dtype.
        """
        with jax.ensure_compile_time_eval():
            if seed_or_key is None:
                seed_or_key = np.random.randint(0, 100000, 2, dtype=np.uint32)
        if np.size(seed_or_key) == 1:
            if isinstance(seed_or_key, int):
                key = jr.PRNGKey(seed_or_key) if use_prng_key else jr.key(seed_or_key)
            elif jnp.issubdtype(seed_or_key.dtype, jax.dtypes.prng_key):
                key = seed_or_key
            elif isinstance(seed_or_key, (jnp.ndarray, np.ndarray)) and jnp.issubdtype(seed_or_key.dtype, jnp.integer):
                key = jr.PRNGKey(seed_or_key) if use_prng_key else jr.key(seed_or_key)
            else:
                raise ValueError(f'Invalid seed_or_key: {seed_or_key}')
        else:
            if len(seed_or_key) == 2 and seed_or_key.dtype == np.uint32:
                key = seed_or_key
            else:
                raise ValueError(f'Invalid seed_or_key: {seed_or_key}')
        self.value = key

    def split_key(self, n: Optional[int] = None, backup: bool = False) -> SeedOrKey:
        """
        Create a new seed from the current seed.

        Parameters
        ----------
        n: int, optional
          The number of seeds to generate.
        backup : bool, optional
          Whether to backup the current key.

        Returns
        -------
        key : SeedOrKey
          The new seed or a tuple of JAX random keys.
        """
        if n is not None:
            assert isinstance(n, int) and n >= 1, f'n should be an integer greater than 1, but we got {n}'

        if not isinstance(self.value, jax.Array):
            self.value = jnp.asarray(self.value, dtype=jnp.uint32)
        keys = jr.split(self.value, num=2 if n is None else n + 1)
        self.value = keys[0]
        if backup:
            self.backup_key()
        if n is None:
            return keys[1]
        else:
            return keys[1:]

    def self_assign_multi_keys(self, n: int, backup: bool = True):
        """
        Self-assign multiple keys to the current random state.
        """
        if backup:
            keys = jr.split(self.value, n + 1)
            self.value = keys[0]
            self.backup_key()
            self.value = keys[1:]
        else:
            self.value = jr.split(self.value, n)

    # ---------------- #
    # random functions #
    # ---------------- #

    def rand(self, *dn, key: Optional[SeedOrKey] = None, dtype: DTypeLike = None):
        key = self.split_key() if key is None else _formalize_key(key)
        dtype = dtype or environ.dftype()
        r = uniform_for_unit(key, shape=dn, minval=0., maxval=1., dtype=dtype)
        return r

    def randint(
        self,
        low,
        high=None,
        size: Optional[Size] = None,
        dtype: DTypeLike = None,
        key: Optional[SeedOrKey] = None
    ):
        if high is None:
            high = low
            low = 0
        high = _check_py_seq(high)
        low = _check_py_seq(low)
        if size is None:
            size = lax.broadcast_shapes(jnp.shape(low),
                                        jnp.shape(high))
        key = self.split_key() if key is None else _formalize_key(key)
        dtype = dtype or environ.ditype()
        r = jr.randint(key,
                       shape=_size2shape(size),
                       minval=low, maxval=high, dtype=dtype)
        return r

    def random_integers(
        self,
        low,
        high=None,
        size: Optional[Size] = None,
        key: Optional[SeedOrKey] = None,
        dtype: DTypeLike = None,
    ):
        low = _check_py_seq(low)
        high = _check_py_seq(high)
        if high is None:
            high = low
            low = 1
        high += 1
        if size is None:
            size = lax.broadcast_shapes(jnp.shape(low), jnp.shape(high))
        key = self.split_key() if key is None else _formalize_key(key)
        dtype = dtype or environ.ditype()
        r = jr.randint(key,
                       shape=_size2shape(size),
                       minval=low,
                       maxval=high,
                       dtype=dtype)
        return r

    def randn(self, *dn, key: Optional[SeedOrKey] = None, dtype: DTypeLike = None):
        key = self.split_key() if key is None else _formalize_key(key)
        dtype = dtype or environ.dftype()
        r = jr.normal(key, shape=dn, dtype=dtype)
        return r

    def random(self,
               size: Optional[Size] = None,
               key: Optional[SeedOrKey] = None,
               dtype: DTypeLike = None):
        dtype = dtype or environ.dftype()
        key = self.split_key() if key is None else _formalize_key(key)
        r = uniform_for_unit(key, shape=_size2shape(size), minval=0., maxval=1., dtype=dtype)
        return r

    def random_sample(self,
                      size: Optional[Size] = None,
                      key: Optional[SeedOrKey] = None,
                      dtype: DTypeLike = None):
        r = self.random(size=size, key=key, dtype=dtype)
        return r

    def ranf(self,
             size: Optional[Size] = None,
             key: Optional[SeedOrKey] = None,
             dtype: DTypeLike = None):
        r = self.random(size=size, key=key, dtype=dtype)
        return r

    def sample(self,
               size: Optional[Size] = None,
               key: Optional[SeedOrKey] = None,
               dtype: DTypeLike = None):
        r = self.random(size=size, key=key, dtype=dtype)
        return r

    def choice(self,
               a,
               size: Optional[Size] = None,
               replace=True,
               p=None,
               key: Optional[SeedOrKey] = None):
        a = _check_py_seq(a)
        p = _check_py_seq(p)
        key = self.split_key() if key is None else _formalize_key(key)
        r = jr.choice(key, a=a, shape=_size2shape(size), replace=replace, p=p)
        return r

    def permutation(self,
                    x,
                    axis: int = 0,
                    independent: bool = False,
                    key: Optional[SeedOrKey] = None):
        x = _check_py_seq(x)
        key = self.split_key() if key is None else _formalize_key(key)
        r = permutation_for_unit(key, x, axis=axis, independent=independent)
        return r

    def shuffle(self,
                x,
                axis=0,
                key: Optional[SeedOrKey] = None):
        key = self.split_key() if key is None else _formalize_key(key)
        x = permutation_for_unit(key, x, axis=axis)
        return x

    def beta(self,
             a,
             b,
             size: Optional[Size] = None,
             key: Optional[SeedOrKey] = None,
             dtype: DTypeLike = None):
        a = _check_py_seq(a)
        b = _check_py_seq(b)
        if size is None:
            size = lax.broadcast_shapes(jnp.shape(a), jnp.shape(b))
        key = self.split_key() if key is None else _formalize_key(key)
        dtype = dtype or environ.dftype()
        r = jr.beta(key, a=a, b=b, shape=_size2shape(size), dtype=dtype)
        return r

    def exponential(self,
                    scale=None,
                    size: Optional[Size] = None,
                    key: Optional[SeedOrKey] = None,
                    dtype: DTypeLike = None):
        if size is None:
            size = jnp.shape(scale)
        key = self.split_key() if key is None else _formalize_key(key)
        dtype = dtype or environ.dftype()
        scale = jnp.asarray(scale, dtype=dtype)
        r = jr.exponential(key, shape=_size2shape(size), dtype=dtype)
        if scale is not None:
            r = r / scale
        return r

    def gamma(self,
              shape,
              scale=None,
              size: Optional[Size] = None,
              key: Optional[SeedOrKey] = None,
              dtype: DTypeLike = None):
        shape = _check_py_seq(shape)
        scale = _check_py_seq(scale)
        if size is None:
            size = lax.broadcast_shapes(jnp.shape(shape), jnp.shape(scale))
        key = self.split_key() if key is None else _formalize_key(key)
        dtype = dtype or environ.dftype()
        r = jr.gamma(key, a=shape, shape=_size2shape(size), dtype=dtype)
        if scale is not None:
            r = r * scale
        return r

    def gumbel(self,
               loc=None,
               scale=None,
               size: Optional[Size] = None,
               key: Optional[SeedOrKey] = None,
               dtype: DTypeLike = None):
        loc = _check_py_seq(loc)
        scale = _check_py_seq(scale)
        if size is None:
            size = lax.broadcast_shapes(jnp.shape(loc), jnp.shape(scale))
        key = self.split_key() if key is None else _formalize_key(key)
        dtype = dtype or environ.dftype()
        r = _loc_scale(loc, scale, jr.gumbel(key, shape=_size2shape(size), dtype=dtype))
        return r

    def laplace(self,
                loc=None,
                scale=None,
                size: Optional[Size] = None,
                key: Optional[SeedOrKey] = None,
                dtype: DTypeLike = None):
        loc = _check_py_seq(loc)
        scale = _check_py_seq(scale)
        if size is None:
            size = lax.broadcast_shapes(jnp.shape(loc), jnp.shape(scale))
        key = self.split_key() if key is None else _formalize_key(key)
        dtype = dtype or environ.dftype()
        r = _loc_scale(loc, scale, jr.laplace(key, shape=_size2shape(size), dtype=dtype))
        return r

    def logistic(self,
                 loc=None,
                 scale=None,
                 size: Optional[Size] = None,
                 key: Optional[SeedOrKey] = None,
                 dtype: DTypeLike = None):
        loc = _check_py_seq(loc)
        scale = _check_py_seq(scale)
        if size is None:
            size = lax.broadcast_shapes(
                jnp.shape(loc) if loc is not None else (),
                jnp.shape(scale) if scale is not None else ()
            )
        key = self.split_key() if key is None else _formalize_key(key)
        dtype = dtype or environ.dftype()
        r = _loc_scale(loc, scale, jr.logistic(key, shape=_size2shape(size), dtype=dtype))
        return r

    def normal(self,
               loc=None,
               scale=None,
               size: Optional[Size] = None,
               key: Optional[SeedOrKey] = None,
               dtype: DTypeLike = None):
        loc = _check_py_seq(loc)
        scale = _check_py_seq(scale)
        if size is None:
            size = lax.broadcast_shapes(
                jnp.shape(scale) if scale is not None else (),
                jnp.shape(loc) if loc is not None else ()
            )
        key = self.split_key() if key is None else _formalize_key(key)
        dtype = dtype or environ.dftype()
        r = _loc_scale(loc, scale, jr.normal(key, shape=_size2shape(size), dtype=dtype))
        return r

    def pareto(self,
               a,
               size: Optional[Size] = None,
               key: Optional[SeedOrKey] = None,
               dtype: DTypeLike = None):
        if size is None:
            size = jnp.shape(a)
        key = self.split_key() if key is None else _formalize_key(key)
        dtype = dtype or environ.dftype()
        a = jnp.asarray(a, dtype=dtype)
        r = jr.pareto(key, b=a, shape=_size2shape(size), dtype=dtype)
        return r

    def poisson(self,
                lam=1.0,
                size: Optional[Size] = None,
                key: Optional[SeedOrKey] = None,
                dtype: DTypeLike = None):
        lam = _check_py_seq(lam)
        if size is None:
            size = jnp.shape(lam)
        key = self.split_key() if key is None else _formalize_key(key)
        dtype = dtype or environ.ditype()
        r = jr.poisson(key, lam=lam, shape=_size2shape(size), dtype=dtype)
        return r

    def standard_cauchy(self,
                        size: Optional[Size] = None,
                        key: Optional[SeedOrKey] = None,
                        dtype: DTypeLike = None):
        key = self.split_key() if key is None else _formalize_key(key)
        dtype = dtype or environ.dftype()
        r = jr.cauchy(key, shape=_size2shape(size), dtype=dtype)
        return r

    def standard_exponential(self,
                             size: Optional[Size] = None,
                             key: Optional[SeedOrKey] = None,
                             dtype: DTypeLike = None):
        key = self.split_key() if key is None else _formalize_key(key)
        dtype = dtype or environ.dftype()
        r = jr.exponential(key, shape=_size2shape(size), dtype=dtype)
        return r

    def standard_gamma(self,
                       shape,
                       size: Optional[Size] = None,
                       key: Optional[SeedOrKey] = None,
                       dtype: DTypeLike = None):
        shape = _check_py_seq(shape)
        if size is None:
            size = jnp.shape(shape) if shape is not None else ()
        key = self.split_key() if key is None else _formalize_key(key)
        dtype = dtype or environ.dftype()
        r = jr.gamma(key, a=shape, shape=_size2shape(size), dtype=dtype)
        return r

    def standard_normal(self,
                        size: Optional[Size] = None,
                        key: Optional[SeedOrKey] = None,
                        dtype: DTypeLike = None):
        key = self.split_key() if key is None else _formalize_key(key)
        dtype = dtype or environ.dftype()
        r = jr.normal(key, shape=_size2shape(size), dtype=dtype)
        return r

    def standard_t(self, df,
                   size: Optional[Size] = None,
                   key: Optional[SeedOrKey] = None,
                   dtype: DTypeLike = None):
        df = _check_py_seq(df)
        if size is None:
            size = jnp.shape(size) if size is not None else ()
        key = self.split_key() if key is None else _formalize_key(key)
        dtype = dtype or environ.dftype()
        r = jr.t(key, df=df, shape=_size2shape(size), dtype=dtype)
        return r

    def uniform(self,
                low=0.0,
                high=1.0,
                size: Optional[Size] = None,
                key: Optional[SeedOrKey] = None,
                dtype: DTypeLike = None):
        low = _check_py_seq(low)
        high = _check_py_seq(high)
        if size is None:
            size = lax.broadcast_shapes(jnp.shape(low), jnp.shape(high))
        key = self.split_key() if key is None else _formalize_key(key)
        dtype = dtype or environ.dftype()
        r = uniform_for_unit(key, shape=_size2shape(size), minval=low, maxval=high, dtype=dtype)
        return r

    def __norm_cdf(self, x, sqrt2, dtype):
        # Computes standard normal cumulative distribution function
        return (np.asarray(1., dtype) + lax.erf(x / sqrt2)) / np.asarray(2., dtype)

    def truncated_normal(
        self,
        lower,
        upper,
        size: Optional[Size] = None,
        loc=0.,
        scale=1.,
        key: Optional[SeedOrKey] = None,
        dtype: DTypeLike = None
    ):
        lower = _check_py_seq(lower)
        upper = _check_py_seq(upper)
        loc = _check_py_seq(loc)
        scale = _check_py_seq(scale)
        dtype = dtype or environ.dftype()

        lower = u.math.asarray(lower, dtype=dtype)
        upper = u.math.asarray(upper, dtype=dtype)
        loc = u.math.asarray(loc, dtype=dtype)
        scale = u.math.asarray(scale, dtype=dtype)
        unit = u.get_unit(lower)
        lower, upper, loc, scale = (
            lower.mantissa if isinstance(lower, u.Quantity) else lower,
            u.Quantity(upper).in_unit(unit).mantissa,
            u.Quantity(loc).in_unit(unit).mantissa,
            u.Quantity(scale).in_unit(unit).mantissa
        )

        jit_error_if(
            u.math.any(u.math.logical_or(loc < lower - 2 * scale, loc > upper + 2 * scale)),
            "mean is more than 2 std from [lower, upper] in truncated_normal. "
            "The distribution of values may be incorrect."
        )

        if size is None:
            size = u.math.broadcast_shapes(jnp.shape(lower),
                                           jnp.shape(upper),
                                           jnp.shape(loc),
                                           jnp.shape(scale))

        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        sqrt2 = np.array(np.sqrt(2), dtype=dtype)
        l = self.__norm_cdf((lower - loc) / scale, sqrt2, dtype)
        u_ = self.__norm_cdf((upper - loc) / scale, sqrt2, dtype)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        key = self.split_key() if key is None else _formalize_key(key)
        out = uniform_for_unit(
            key, size, dtype,
            minval=lax.nextafter(2 * l - 1, np.array(np.inf, dtype=dtype)),
            maxval=lax.nextafter(2 * u_ - 1, np.array(-np.inf, dtype=dtype))
        )

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        out = lax.erf_inv(out)

        # Transform to proper mean, std
        out = out * scale * sqrt2 + loc

        # Clamp to ensure it's in the proper range
        out = jnp.clip(
            out,
            lax.nextafter(lax.stop_gradient(lower), np.array(np.inf, dtype=dtype)),
            lax.nextafter(lax.stop_gradient(upper), np.array(-np.inf, dtype=dtype))
        )
        return out if unit.is_unitless else u.Quantity(out, unit=unit)

    def _check_p(self, *args, **kwargs):
        raise ValueError('Parameter p should be within [0, 1], but we got {p}')

    def bernoulli(self,
                  p,
                  size: Optional[Size] = None,
                  key: Optional[SeedOrKey] = None):
        p = _check_py_seq(p)
        jit_error_if(jnp.any(jnp.logical_or(p < 0, p > 1)), self._check_p, p=p)
        if size is None:
            size = jnp.shape(p)
        key = self.split_key() if key is None else _formalize_key(key)
        r = jr.bernoulli(key, p=p, shape=_size2shape(size))
        return r

    def lognormal(
        self,
        mean=None,
        sigma=None,
        size: Optional[Size] = None,
        key: Optional[SeedOrKey] = None,
        dtype: DTypeLike = None
    ):
        mean = _check_py_seq(mean)
        sigma = _check_py_seq(sigma)
        mean = u.math.asarray(mean, dtype=dtype)
        sigma = u.math.asarray(sigma, dtype=dtype)
        unit = mean.unit if isinstance(mean, u.Quantity) else u.Unit()
        mean = mean.mantissa if isinstance(mean, u.Quantity) else mean
        sigma = sigma.in_unit(unit).mantissa if isinstance(sigma, u.Quantity) else sigma

        if size is None:
            size = jnp.broadcast_shapes(
                jnp.shape(mean) if mean is not None else (),
                jnp.shape(sigma) if sigma is not None else ()
            )
        key = self.split_key() if key is None else _formalize_key(key)
        dtype = dtype or environ.dftype()
        samples = jr.normal(key, shape=_size2shape(size), dtype=dtype)
        samples = _loc_scale(mean, sigma, samples)
        samples = jnp.exp(samples)
        return samples if unit.is_unitless else u.Quantity(samples, unit=unit)

    def binomial(
        self,
        n,
        p,
        size: Optional[Size] = None,
        key: Optional[SeedOrKey] = None,
        dtype: DTypeLike = None,
        check_valid: bool = True,
    ):
        n = _check_py_seq(n)
        p = _check_py_seq(p)
        if check_valid:
            jit_error_if(
                jnp.any(jnp.logical_or(p < 0, p > 1)),
                'Parameter p should be within [0, 1], but we got {p}',
                p=p
            )
        if size is None:
            size = jnp.broadcast_shapes(jnp.shape(n), jnp.shape(p))
        key = self.split_key() if key is None else _formalize_key(key)
        r = jr.binomial(key, n, p, shape=_size2shape(size))
        dtype = dtype or environ.ditype()
        return jnp.asarray(r, dtype=dtype)

    def chisquare(self,
                  df,
                  size: Optional[Size] = None,
                  key: Optional[SeedOrKey] = None,
                  dtype: DTypeLike = None):
        df = _check_py_seq(df)
        key = self.split_key() if key is None else _formalize_key(key)
        dtype = dtype or environ.dftype()
        if size is None:
            if jnp.ndim(df) == 0:
                dist = jr.normal(key, (df,), dtype=dtype) ** 2
                dist = dist.sum()
            else:
                raise NotImplementedError('Do not support non-scale "df" when "size" is None')
        else:
            dist = jr.normal(key, (df,) + _size2shape(size), dtype=dtype) ** 2
            dist = dist.sum(axis=0)
        return dist

    def dirichlet(self,
                  alpha,
                  size: Optional[Size] = None,
                  key: Optional[SeedOrKey] = None,
                  dtype: DTypeLike = None):
        key = self.split_key() if key is None else _formalize_key(key)
        alpha = _check_py_seq(alpha)
        dtype = dtype or environ.dftype()
        r = jr.dirichlet(key, alpha=alpha, shape=_size2shape(size), dtype=dtype)
        return r

    def geometric(self,
                  p,
                  size: Optional[Size] = None,
                  key: Optional[SeedOrKey] = None,
                  dtype: DTypeLike = None):
        p = _check_py_seq(p)
        if size is None:
            size = jnp.shape(p)
        key = self.split_key() if key is None else _formalize_key(key)
        dtype = dtype or environ.dftype()
        u_ = uniform_for_unit(key, size, dtype=dtype)
        r = jnp.floor(jnp.log1p(-u_) / jnp.log1p(-p))
        return r

    def _check_p2(self, p):
        raise ValueError(f'We require `sum(pvals[:-1]) <= 1`. But we got {p}')

    def multinomial(self,
                    n,
                    pvals,
                    size: Optional[Size] = None,
                    key: Optional[SeedOrKey] = None,
                    dtype: DTypeLike = None):
        key = self.split_key() if key is None else _formalize_key(key)
        n = _check_py_seq(n)
        pvals = _check_py_seq(pvals)
        jit_error_if(jnp.sum(pvals[:-1]) > 1., self._check_p2, pvals)
        if isinstance(n, jax.core.Tracer):
            raise ValueError("The total count parameter `n` should not be a jax abstract array.")
        size = _size2shape(size)
        n_max = int(np.max(jax.device_get(n)))
        batch_shape = lax.broadcast_shapes(jnp.shape(pvals)[:-1], jnp.shape(n))
        r = _multinomial(key, pvals, n, n_max, batch_shape + size)
        dtype = dtype or environ.ditype()
        return jnp.asarray(r, dtype=dtype)

    def multivariate_normal(
        self,
        mean,
        cov,
        size: Optional[Size] = None,
        method: str = 'cholesky',
        key: Optional[SeedOrKey] = None,
        dtype: DTypeLike = None
    ):
        if method not in {'svd', 'eigh', 'cholesky'}:
            raise ValueError("method must be one of {'svd', 'eigh', 'cholesky'}")
        dtype = dtype or environ.dftype()
        mean = u.math.asarray(_check_py_seq(mean), dtype=dtype)
        cov = u.math.asarray(_check_py_seq(cov), dtype=dtype)
        if isinstance(mean, u.Quantity):
            assert isinstance(cov, u.Quantity)
            assert mean.unit ** 2 == cov.unit
        mean = mean.mantissa if isinstance(mean, u.Quantity) else mean
        cov = cov.mantissa if isinstance(cov, u.Quantity) else cov
        unit = mean.unit if isinstance(mean, u.Quantity) else u.Unit()

        key = self.split_key() if key is None else _formalize_key(key)
        if not jnp.ndim(mean) >= 1:
            raise ValueError(f"multivariate_normal requires mean.ndim >= 1, got mean.ndim == {jnp.ndim(mean)}")
        if not jnp.ndim(cov) >= 2:
            raise ValueError(f"multivariate_normal requires cov.ndim >= 2, got cov.ndim == {jnp.ndim(cov)}")
        n = mean.shape[-1]
        if jnp.shape(cov)[-2:] != (n, n):
            raise ValueError(f"multivariate_normal requires cov.shape == (..., n, n) for n={n}, "
                             f"but got cov.shape == {jnp.shape(cov)}.")
        if size is None:
            size = lax.broadcast_shapes(mean.shape[:-1], cov.shape[:-2])
        else:
            size = _size2shape(size)
            _check_shape("normal", size, mean.shape[:-1], cov.shape[:-2])

        if method == 'svd':
            (u_, s, _) = jnp.linalg.svd(cov)
            factor = u_ * jnp.sqrt(s[..., None, :])
        elif method == 'eigh':
            (w, v) = jnp.linalg.eigh(cov)
            factor = v * jnp.sqrt(w[..., None, :])
        else:  # 'cholesky'
            factor = jnp.linalg.cholesky(cov)
        normal_samples = jr.normal(key, size + mean.shape[-1:], dtype=dtype)
        r = mean + jnp.einsum('...ij,...j->...i', factor, normal_samples)
        return r if unit.is_unitless else u.Quantity(r, unit=unit)

    def rayleigh(self,
                 scale=1.0,
                 size: Optional[Size] = None,
                 key: Optional[SeedOrKey] = None,
                 dtype: DTypeLike = None):
        scale = _check_py_seq(scale)
        if size is None:
            size = jnp.shape(scale)
        key = self.split_key() if key is None else _formalize_key(key)
        dtype = dtype or environ.dftype()
        x = jnp.sqrt(-2. * jnp.log(uniform_for_unit(key, shape=_size2shape(size), minval=0, maxval=1, dtype=dtype)))
        r = x * scale
        return r

    def triangular(self,
                   size: Optional[Size] = None,
                   key: Optional[SeedOrKey] = None):
        key = self.split_key() if key is None else _formalize_key(key)
        bernoulli_samples = jr.bernoulli(key, p=0.5, shape=_size2shape(size))
        r = 2 * bernoulli_samples - 1
        return r

    def vonmises(self,
                 mu,
                 kappa,
                 size: Optional[Size] = None,
                 key: Optional[SeedOrKey] = None,
                 dtype: DTypeLike = None):
        key = self.split_key() if key is None else _formalize_key(key)
        dtype = dtype or environ.dftype()
        mu = jnp.asarray(_check_py_seq(mu), dtype=dtype)
        kappa = jnp.asarray(_check_py_seq(kappa), dtype=dtype)
        if size is None:
            size = lax.broadcast_shapes(jnp.shape(mu), jnp.shape(kappa))
        size = _size2shape(size)
        samples = _von_mises_centered(key, kappa, size, dtype=dtype)
        samples = samples + mu
        samples = (samples + jnp.pi) % (2.0 * jnp.pi) - jnp.pi
        return samples

    def weibull(self,
                a,
                size: Optional[Size] = None,
                key: Optional[SeedOrKey] = None,
                dtype: DTypeLike = None):
        key = self.split_key() if key is None else _formalize_key(key)
        a = _check_py_seq(a)
        if size is None:
            size = jnp.shape(a)
        else:
            if jnp.size(a) > 1:
                raise ValueError(f'"a" should be a scalar when "size" is provided. But we got {a}')
        size = _size2shape(size)
        dtype = dtype or environ.dftype()
        random_uniform = uniform_for_unit(key=key, shape=size, minval=0, maxval=1, dtype=dtype)
        r = jnp.power(-jnp.log1p(-random_uniform), 1.0 / a)
        return r

    def weibull_min(self,
                    a,
                    scale=None,
                    size: Optional[Size] = None,
                    key: Optional[SeedOrKey] = None,
                    dtype: DTypeLike = None):
        key = self.split_key() if key is None else _formalize_key(key)
        a = _check_py_seq(a)
        scale = _check_py_seq(scale)
        if size is None:
            size = jnp.broadcast_shapes(jnp.shape(a), jnp.shape(scale) if scale is not None else ())
        else:
            if jnp.size(a) > 1:
                raise ValueError(f'"a" should be a scalar when "size" is provided. But we got {a}')
        size = _size2shape(size)
        dtype = dtype or environ.dftype()
        random_uniform = uniform_for_unit(key=key, shape=size, minval=0, maxval=1, dtype=dtype)
        r = jnp.power(-jnp.log1p(-random_uniform), 1.0 / a)
        if scale is not None:
            r /= scale
        return r

    def maxwell(self,
                size: Optional[Size] = None,
                key: Optional[SeedOrKey] = None,
                dtype: DTypeLike = None):
        key = self.split_key() if key is None else _formalize_key(key)
        shape = _size2shape(size) + (3,)
        dtype = dtype or environ.dftype()
        norm_rvs = jr.normal(key=key, shape=shape, dtype=dtype)
        r = jnp.linalg.norm(norm_rvs, axis=-1)
        return r

    def negative_binomial(self,
                          n,
                          p,
                          size: Optional[Size] = None,
                          key: Optional[SeedOrKey] = None,
                          dtype: DTypeLike = None):
        n = _check_py_seq(n)
        p = _check_py_seq(p)
        if size is None:
            size = lax.broadcast_shapes(jnp.shape(n), jnp.shape(p))
        size = _size2shape(size)
        logits = jnp.log(p) - jnp.log1p(-p)
        if key is None:
            keys = self.split_key(2)
        else:
            keys = jr.split(_formalize_key(key), 2)
        rate = self.gamma(shape=n, scale=jnp.exp(-logits), size=size, key=keys[0], dtype=environ.dftype())
        r = self.poisson(lam=rate, key=keys[1], dtype=dtype or environ.ditype())
        return r

    def wald(self,
             mean,
             scale,
             size: Optional[Size] = None,
             key: Optional[SeedOrKey] = None,
             dtype: DTypeLike = None):
        dtype = dtype or environ.dftype()
        key = self.split_key() if key is None else _formalize_key(key)
        mean = jnp.asarray(_check_py_seq(mean), dtype=dtype)
        scale = jnp.asarray(_check_py_seq(scale), dtype=dtype)
        if size is None:
            size = lax.broadcast_shapes(jnp.shape(mean), jnp.shape(scale))
        size = _size2shape(size)
        sampled_chi2 = jnp.square(self.randn(*size))
        sampled_uniform = self.uniform(size=size, key=key, dtype=dtype)
        # Wikipedia defines an intermediate x with the formula
        #   x = loc + loc ** 2 * y / (2 * conc) - loc / (2 * conc) * sqrt(4 * loc * conc * y + loc ** 2 * y ** 2)
        # where y ~ N(0, 1)**2 (sampled_chi2 above) and conc is the concentration.
        # Let us write
        #   w = loc * y / (2 * conc)
        # Then we can extract the common factor in the last two terms to obtain
        #   x = loc + loc * w * (1 - sqrt(2 / w + 1))
        # Now we see that the Wikipedia formula suffers from catastrphic
        # cancellation for large w (e.g., if conc << loc).
        #
        # Fortunately, we can fix this by multiplying both sides
        # by 1 + sqrt(2 / w + 1).  We get
        #   x * (1 + sqrt(2 / w + 1)) =
        #     = loc * (1 + sqrt(2 / w + 1)) + loc * w * (1 - (2 / w + 1))
        #     = loc * (sqrt(2 / w + 1) - 1)
        # The term sqrt(2 / w + 1) + 1 no longer presents numerical
        # difficulties for large w, and sqrt(2 / w + 1) - 1 is just
        # sqrt1pm1(2 / w), which we know how to compute accurately.
        # This just leaves the matter of small w, where 2 / w may
        # overflow.  In the limit a w -> 0, x -> loc, so we just mask
        # that case.
        sqrt1pm1_arg = 4 * scale / (mean * sampled_chi2)  # 2 / w above
        safe_sqrt1pm1_arg = jnp.where(sqrt1pm1_arg < np.inf, sqrt1pm1_arg, 1.0)
        denominator = 1.0 + jnp.sqrt(safe_sqrt1pm1_arg + 1.0)
        ratio = jnp.expm1(0.5 * jnp.log1p(safe_sqrt1pm1_arg)) / denominator
        sampled = mean * jnp.where(sqrt1pm1_arg < np.inf, ratio, 1.0)  # x above
        res = jnp.where(sampled_uniform <= mean / (mean + sampled),
                        sampled,
                        jnp.square(mean) / sampled)
        return res

    def t(self,
          df,
          size: Optional[Size] = None,
          key: Optional[SeedOrKey] = None,
          dtype: DTypeLike = None):
        dtype = dtype or environ.dftype()
        df = jnp.asarray(_check_py_seq(df), dtype=dtype)
        if size is None:
            size = np.shape(df)
        else:
            size = _size2shape(size)
            _check_shape("t", size, np.shape(df))
        if key is None:
            keys = self.split_key(2)
        else:
            keys = jr.split(_formalize_key(key), 2)
        n = jr.normal(keys[0], size, dtype=dtype)
        two = _const(n, 2)
        half_df = lax.div(df, two)
        g = jr.gamma(keys[1], half_df, size, dtype=dtype)
        r = n * jnp.sqrt(half_df / g)
        return r

    def orthogonal(self,
                   n: int,
                   size: Optional[Size] = None,
                   key: Optional[SeedOrKey] = None,
                   dtype: DTypeLike = None):
        dtype = dtype or environ.dftype()
        key = self.split_key() if key is None else _formalize_key(key)
        size = _size2shape(size)
        _check_shape("orthogonal", size)
        n = core.concrete_or_error(index, n, "The error occurred in jax.random.orthogonal()")
        z = jr.normal(key, size + (n, n), dtype=dtype)
        q, r = jnp.linalg.qr(z)
        d = jnp.diagonal(r, 0, -2, -1)
        r = q * jnp.expand_dims(d / abs(d), -2)
        return r

    def noncentral_chisquare(self,
                             df,
                             nonc,
                             size: Optional[Size] = None,
                             key: Optional[SeedOrKey] = None,
                             dtype: DTypeLike = None):
        dtype = dtype or environ.dftype()
        df = jnp.asarray(_check_py_seq(df), dtype=dtype)
        nonc = jnp.asarray(_check_py_seq(nonc), dtype=dtype)
        if size is None:
            size = lax.broadcast_shapes(jnp.shape(df), jnp.shape(nonc))
        size = _size2shape(size)
        if key is None:
            keys = self.split_key(3)
        else:
            keys = jr.split(_formalize_key(key), 3)
        i = jr.poisson(keys[0], 0.5 * nonc, shape=size, dtype=environ.ditype())
        n = jr.normal(keys[1], shape=size, dtype=dtype) + jnp.sqrt(nonc)
        cond = jnp.greater(df, 1.0)
        df2 = jnp.where(cond, df - 1.0, df + 2.0 * i)
        chi2 = 2.0 * jr.gamma(keys[2], 0.5 * df2, shape=size, dtype=dtype)
        r = jnp.where(cond, chi2 + n * n, chi2)
        return r

    def loggamma(self,
                 a,
                 size: Optional[Size] = None,
                 key: Optional[SeedOrKey] = None,
                 dtype: DTypeLike = None):
        dtype = dtype or environ.dftype()
        key = self.split_key() if key is None else _formalize_key(key)
        a = _check_py_seq(a)
        if size is None:
            size = jnp.shape(a)
        r = jr.loggamma(key, a, shape=_size2shape(size), dtype=dtype)
        return r

    def categorical(self,
                    logits,
                    axis: int = -1,
                    size: Optional[Size] = None,
                    key: Optional[SeedOrKey] = None):
        key = self.split_key() if key is None else _formalize_key(key)
        logits = _check_py_seq(logits)
        if size is None:
            size = list(jnp.shape(logits))
            size.pop(axis)
        r = jr.categorical(key, logits, axis=axis, shape=_size2shape(size))
        return r

    def zipf(self,
             a,
             size: Optional[Size] = None,
             key: Optional[SeedOrKey] = None,
             dtype: DTypeLike = None):
        a = _check_py_seq(a)
        if size is None:
            size = jnp.shape(a)
        dtype = dtype or environ.ditype()
        r = jax.pure_callback(lambda x: np.random.zipf(x, size).astype(dtype),
                              jax.ShapeDtypeStruct(size, dtype),
                              a)
        return r

    def power(self,
              a,
              size: Optional[Size] = None,
              key: Optional[SeedOrKey] = None,
              dtype: DTypeLike = None):
        a = _check_py_seq(a)
        if size is None:
            size = jnp.shape(a)
        size = _size2shape(size)
        dtype = dtype or environ.dftype()
        r = jax.pure_callback(lambda a: np.random.power(a=a, size=size).astype(dtype),
                              jax.ShapeDtypeStruct(size, dtype),
                              a)
        return r

    def f(self,
          dfnum,
          dfden,
          size: Optional[Size] = None,
          key: Optional[SeedOrKey] = None,
          dtype: DTypeLike = None):
        dfnum = _check_py_seq(dfnum)
        dfden = _check_py_seq(dfden)
        if size is None:
            size = jnp.broadcast_shapes(jnp.shape(dfnum), jnp.shape(dfden))
        size = _size2shape(size)
        d = {'dfnum': dfnum, 'dfden': dfden}
        dtype = dtype or environ.dftype()
        r = jax.pure_callback(lambda dfnum_, dfden_: np.random.f(dfnum=dfnum_,
                                                                 dfden=dfden_,
                                                                 size=size).astype(dtype),
                              jax.ShapeDtypeStruct(size, dtype),
                              dfnum, dfden)
        return r

    def hypergeometric(
        self,
        ngood,
        nbad,
        nsample,
        size: Optional[Size] = None,
        key: Optional[SeedOrKey] = None,
        dtype: DTypeLike = None
    ):
        ngood = _check_py_seq(ngood)
        nbad = _check_py_seq(nbad)
        nsample = _check_py_seq(nsample)

        if size is None:
            size = lax.broadcast_shapes(jnp.shape(ngood),
                                        jnp.shape(nbad),
                                        jnp.shape(nsample))
        size = _size2shape(size)
        dtype = dtype or environ.ditype()
        d = {'ngood': ngood, 'nbad': nbad, 'nsample': nsample}
        r = jax.pure_callback(lambda d: np.random.hypergeometric(ngood=d['ngood'],
                                                                 nbad=d['nbad'],
                                                                 nsample=d['nsample'],
                                                                 size=size).astype(dtype),
                              jax.ShapeDtypeStruct(size, dtype),
                              d)
        return r

    def logseries(self,
                  p,
                  size: Optional[Size] = None,
                  key: Optional[SeedOrKey] = None,
                  dtype: DTypeLike = None):
        p = _check_py_seq(p)
        if size is None:
            size = jnp.shape(p)
        size = _size2shape(size)
        dtype = dtype or environ.ditype()
        r = jax.pure_callback(lambda p: np.random.logseries(p=p, size=size).astype(dtype),
                              jax.ShapeDtypeStruct(size, dtype),
                              p)
        return r

    def noncentral_f(self,
                     dfnum,
                     dfden,
                     nonc,
                     size: Optional[Size] = None,
                     key: Optional[SeedOrKey] = None,
                     dtype: DTypeLike = None):
        dfnum = _check_py_seq(dfnum)
        dfden = _check_py_seq(dfden)
        nonc = _check_py_seq(nonc)
        if size is None:
            size = lax.broadcast_shapes(jnp.shape(dfnum),
                                        jnp.shape(dfden),
                                        jnp.shape(nonc))
        size = _size2shape(size)
        d = {'dfnum': dfnum, 'dfden': dfden, 'nonc': nonc}
        dtype = dtype or environ.dftype()
        r = jax.pure_callback(lambda x: np.random.noncentral_f(dfnum=x['dfnum'],
                                                               dfden=x['dfden'],
                                                               nonc=x['nonc'],
                                                               size=size).astype(dtype),
                              jax.ShapeDtypeStruct(size, dtype),
                              d)
        return r

    # PyTorch compatibility #
    # --------------------- #

    def rand_like(self, input, *, dtype=None, key: Optional[SeedOrKey] = None):
        """Returns a tensor with the same size as input that is filled with random
        numbers from a uniform distribution on the interval ``[0, 1)``.

        Args:
          input:  the ``size`` of input will determine size of the output tensor.
          dtype:  the desired data type of returned Tensor. Default: if ``None``, defaults to the dtype of input.
          key: the seed or key for the random.

        Returns:
          The random data.
        """
        return self.random(jnp.shape(input), key=key).astype(dtype)

    def randn_like(self, input, *, dtype=None, key: Optional[SeedOrKey] = None):
        """Returns a tensor with the same size as ``input`` that is filled with
        random numbers from a normal distribution with mean 0 and variance 1.

        Args:
          input:  the ``size`` of input will determine size of the output tensor.
          dtype:  the desired data type of returned Tensor. Default: if ``None``, defaults to the dtype of input.
          key: the seed or key for the random.

        Returns:
          The random data.
        """
        return self.randn(*jnp.shape(input), key=key).astype(dtype)

    def randint_like(self, input, low=0, high=None, *, dtype=None, key: Optional[SeedOrKey] = None):
        if high is None:
            high = max(input)
        return self.randint(low, high=high, size=jnp.shape(input), dtype=dtype, key=key)


# default random generator
DEFAULT = RandomState(np.random.randint(0, 10000, size=2, dtype=np.uint32))


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
        return jnp.asarray(key, dtype=jnp.uint32)
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


def _is_python_scalar(x):
    if hasattr(x, 'aval'):
        return x.aval.weak_type
    elif np.ndim(x) == 0:
        return True
    elif isinstance(x, (bool, int, float, complex)):
        return True
    else:
        return False


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


def _const(example, val):
    if _is_python_scalar(example):
        dtype = dtypes.canonicalize_dtype(type(example))
        val = dtypes.scalar_type_of(example)(val)
        return val if dtype == _dtype(val, canonicalize=True) else np.array(val, dtype)
    else:
        dtype = dtypes.canonicalize_dtype(example.dtype)
    return np.array(val, dtype)


@partial(jit, static_argnums=(2,))
def _categorical(key, p, shape):
    # this implementation is fast when event shape is small, and slow otherwise
    # Ref: https://stackoverflow.com/a/34190035
    shape = shape or p.shape[:-1]
    s = jnp.cumsum(p, axis=-1)
    r = jr.uniform(key, shape=shape + (1,))
    return jnp.sum(s < r, axis=-1)


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
        shapes = [jnp.shape(arg) for arg in args]
        num_dims = len(lax.broadcast_shapes(shape, *shapes))
        return [
            _reshape(arg, (1,) * (num_dims - len(s)) + s) if len(s) < num_dims else arg
            for arg, s in zip(args, shapes)
        ]


@partial(jit, static_argnums=(3, 4))
def _multinomial(key, p, n, n_max, shape=()):
    if jnp.shape(n) != jnp.shape(p)[:-1]:
        broadcast_shape = lax.broadcast_shapes(jnp.shape(n), jnp.shape(p)[:-1])
        n = jnp.broadcast_to(n, broadcast_shape)
        p = jnp.broadcast_to(p, broadcast_shape + jnp.shape(p)[-1:])
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
                                  jnp.zeros(jnp.shape(n) + (p.shape[-1] - 1,))],
                                 -1)
    else:
        mask = 1
        excess = 0
    # NB: we transpose to move batch shape to the front
    indices_2D = (jnp.reshape(indices * mask, (n_max, -1))).T
    samples_2D = vmap(_scatter_add_one)(jnp.zeros((indices_2D.shape[0], p.shape[-1]), dtype=indices.dtype),
                                        jnp.expand_dims(indices_2D, axis=-1),
                                        jnp.ones(indices_2D.shape, dtype=indices.dtype))
    return jnp.reshape(samples_2D, shape + p.shape[-1:]) - excess


@partial(jit, static_argnums=(2, 3), static_argnames=['shape', 'dtype'])
def _von_mises_centered(key, concentration, shape, dtype=None):
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
    shape = shape or jnp.shape(concentration)
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
        u = jr.uniform(
            key=uni_ukey,
            shape=shape,
            dtype=concentration.dtype,
            minval=-1.0,
            maxval=1.0,
        )
        z = jnp.cos(jnp.pi * u)
        w = jnp.where(done, w, (1.0 + s * z) / (s + z))  # Update where not done
        y = concentration * (s - w)
        v = jr.uniform(key=uni_vkey, shape=shape, dtype=concentration.dtype)
        accept = (y * (2.0 - y) >= v) | (jnp.log(y / v) + 1.0 >= y)
        return i + 1, key, accept | done, u, w

    init_done = jnp.zeros(shape, dtype=bool)
    init_u = jnp.zeros(shape)
    init_w = jnp.zeros(shape)

    _, _, done, u, w = lax.while_loop(
        cond_fun=cond_fn,
        body_fun=body_fn,
        init_val=(jnp.array(0), key, init_done, init_u, init_w),
    )

    return jnp.sign(u) * jnp.arccos(w)


def _loc_scale(loc, scale, value):
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
    return jnp.asarray(seq) if isinstance(seq, (tuple, list)) else seq
