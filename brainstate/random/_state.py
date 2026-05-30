# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
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

from operator import index
from typing import Optional

import brainunit as u
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import lax

from brainstate import environ
from brainstate._compatible_import import Tracer, concrete_or_error
from brainstate._state import State
from brainstate.transform._named_scope import named_scope
from brainstate.typing import DTypeLike, Size, SeedOrKey
from ._impl import (
    multinomial,
    von_mises_centered,
    const,
    formalize_key,
    _format_key,
    _is_typed_key,
    _validate_raw_key_data,
    _loc_scale,
    _loc_scale_unit,
    _scale_unit,
    _remove_unit_param,
    _size2shape,
    _check_py_seq,
    _check_shape,
    noncentral_f,
    logseries,
    hypergeometric,
    f,
    power,
    zipf,
)

__all__ = [
    'RandomState',
    'DEFAULT',
]


def _randn_static_argnums(self, *dn, **kwargs):
    return tuple(range(len(dn) + 1))


class RandomState(State):
    """RandomState that track the random generator state. """

    __module__ = 'brainstate.random'

    def __init__(
        self,
        seed_or_key: Optional[SeedOrKey] = None
    ):
        """RandomState constructor.

        Parameters
        ----------
        seed_or_key: int, Array, optional
          An integer seed, a JAX typed PRNG key (``jax.random.key``), or a legacy
          ``uint32[2]`` key array (auto-wrapped into a typed key). If ``None``, a
          random seed is drawn.

        Notes
        -----
        When constructed from a raw ``uint32[2]`` numpy array (as the module-level
        ``DEFAULT`` is), the raw data is stored as a lazy placeholder and only
        materialized into a typed JAX key on first use. This keeps ``import
        brainstate`` free of JAX device-array creation / backend initialization.
        """
        if seed_or_key is None:
            # Lazy numpy placeholder: holding a numpy array never touches JAX, so
            # constructing the module-level DEFAULT does not initialize a backend.
            seed_or_key = np.random.randint(0, 2 ** 32, size=2, dtype=np.uint32)
        if (
            isinstance(seed_or_key, np.ndarray)
            and not _is_typed_key(seed_or_key)
            and seed_or_key.dtype == jnp.uint32
            and seed_or_key.size == 2
        ):
            # A legacy raw ``uint32[2]`` key: store it lazily and materialize into a
            # typed key on first read. This keeps ``import brainstate`` free of JAX
            # device-array creation while preserving the exact key bits.
            key = seed_or_key
        else:
            # int / typed key / size-1 integer seed array -> typed key (may call JAX).
            key = _format_key(seed_or_key)
        super().__init__(key)

        self._backup = None

    def __repr__(self):
        return f'{self.__class__.__name__}({self.value})'

    def check_if_deleted(self):
        # Lazily wrap a stored raw-key placeholder into a typed key on first read.
        # This preserves the seed the state was constructed with (no fresh
        # randomness), which is required for reproducibility.
        if isinstance(self._value, np.ndarray):
            self._value = _format_key(self._value)

        if (
            isinstance(self._value, jax.Array) and
            not isinstance(self._value, Tracer) and
            self._value.is_deleted()
        ):
            self.seed()

    def _numpy_keys(self, batch_size):
        return np.random.randint(0, 10000, (batch_size, 2), dtype=np.uint32)

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
        self.value = _format_key(key)

    def seed(
        self,
        seed_or_key: Optional[SeedOrKey] = None
    ):
        """Sets a new random seed.

        Parameters
        ----------
        seed_or_key: int, ArrayLike, optional
          An integer seed, a JAX typed PRNG key, or a legacy ``uint32[2]`` key
          array (auto-wrapped into a typed key). If ``None``, a random seed is drawn.
        """
        if seed_or_key is None:
            # fresh lazy placeholder, materialized on first use
            self.value = np.random.randint(0, 2 ** 32, size=2, dtype=np.uint32)
        else:
            self.value = _format_key(seed_or_key)

    def split_key(
        self,
        n: Optional[int] = None,
        backup: bool = False
    ) -> SeedOrKey:
        """
        Create a new seed from the current seed.

        Parameters
        ----------
        n: int, optional
          The number of seeds to generate.
        backup : bool, optional
          Whether to back up the current key.

        Returns
        -------
        key : SeedOrKey
          A single typed JAX key (``n`` is ``None``), or a typed-key array of shape
          ``(n,)``.
        """
        if n is not None:
            assert isinstance(n, int) and n >= 1, f'n should be an integer greater than 1, but we got {n}'

        # Reading ``self.value`` materializes a lazy placeholder into a typed key.
        keys = jr.split(self.value, num=2 if n is None else n + 1)
        self.value = keys[0]
        if backup:
            self.backup_key()
        if n is None:
            return keys[1]
        else:
            return keys[1:]

    def self_assign_multi_keys(
        self,
        n: int,
        backup: bool = True
    ):
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

    def __get_key(self, key):
        return self.split_key() if key is None else _format_key(key)

    # ---------------- #
    # random functions #
    # ---------------- #

    @named_scope('brainstate/random', static_argnums=_randn_static_argnums, static_argnames=['dtype'])
    def rand(
        self,
        *dn,
        key: Optional[SeedOrKey] = None,
        dtype: DTypeLike = None
    ):
        key = self.__get_key(key)
        dtype = dtype or environ.dftype()
        r = jr.uniform(key, dn, dtype)
        return r

    @named_scope(
        'brainstate/random',
        static_argnums=(0, 3, 4),
        static_argnames=['dtype', 'size']
    )
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
        high = _remove_unit_param('high', _check_py_seq(high))
        low = _remove_unit_param('low', _check_py_seq(low))
        if size is None:
            size = lax.broadcast_shapes(u.math.shape(low), u.math.shape(high))
        key = self.__get_key(key)
        dtype = dtype or environ.ditype()
        r = jr.randint(
            key,
            shape=_size2shape(size),
            minval=low,
            maxval=high,
            dtype=dtype
        )
        return r

    @named_scope(
        'brainstate/random',
        static_argnums=(0, 3, 5),
        static_argnames=['dtype', 'size']
    )
    def random_integers(
        self,
        low,
        high=None,
        size: Optional[Size] = None,
        key: Optional[SeedOrKey] = None,
        dtype: DTypeLike = None
    ):
        low = _remove_unit_param('low', _check_py_seq(low))
        high = _remove_unit_param('high', _check_py_seq(high))
        if high is None:
            high = low
            low = 1
        high += 1
        if size is None:
            size = lax.broadcast_shapes(u.math.shape(low), u.math.shape(high))
        key = self.__get_key(key)
        dtype = dtype or environ.ditype()
        r = jr.randint(
            key,
            shape=_size2shape(size),
            minval=low,
            maxval=high,
            dtype=dtype
        )
        return r

    @named_scope(
        'brainstate/random',
        static_argnums=_randn_static_argnums,
        static_argnames=['dtype'],
    )
    def randn(
        self,
        *dn,
        key: Optional[SeedOrKey] = None,
        dtype: DTypeLike = None
    ):
        key = self.__get_key(key)
        r = jr.normal(key, shape=dn, dtype=dtype or environ.dftype())
        return r

    @named_scope(
        'brainstate/random',
        static_argnums=(0, 1, 3),
        static_argnames=['dtype', 'size']
    )
    def random(
        self,
        size: Optional[Size] = None,
        key: Optional[SeedOrKey] = None,
        dtype: DTypeLike = None
    ):
        key = self.__get_key(key)
        r = jr.uniform(key, _size2shape(size), dtype=dtype or environ.dftype())
        return r

    @named_scope(
        'brainstate/random',
        static_argnums=(0, 1, 3),
        static_argnames=['dtype', 'size']
    )
    def random_sample(
        self,
        size: Optional[Size] = None,
        key: Optional[SeedOrKey] = None,
        dtype: DTypeLike = None
    ):
        r = self.random(size=size, key=key, dtype=dtype or environ.dftype())
        return r

    @named_scope(
        'brainstate/random',
        static_argnums=(0, 1, 3),
        static_argnames=['dtype', 'size']
    )
    def ranf(
        self,
        size: Optional[Size] = None,
        key: Optional[SeedOrKey] = None,
        dtype: DTypeLike = None
    ):
        r = self.random(size=size, key=key, dtype=dtype or environ.dftype())
        return r

    @named_scope('brainstate/random', static_argnums=(0, 1, 3), static_argnames=['dtype', 'size'])
    def sample(
        self,
        size: Optional[Size] = None,
        key: Optional[SeedOrKey] = None,
        dtype: DTypeLike = None
    ):
        r = self.random(size=size, key=key, dtype=dtype or environ.dftype())
        return r

    @named_scope(
        'brainstate/random',
        static_argnums=lambda self, a, *args, **kwargs: (0, 1, 2, 3) if isinstance(a, int) else (0, 2, 3),
        static_argnames=['size', 'replace']
    )
    def choice(
        self,
        a,
        size: Optional[Size] = None,
        replace=True,
        p=None,
        key: Optional[SeedOrKey] = None
    ):
        a = _check_py_seq(a)
        a, unit = u.split_mantissa_unit(a)
        p = _check_py_seq(p)
        key = self.__get_key(key)
        r = jr.choice(key, a=a, shape=_size2shape(size), replace=replace, p=p)
        return u.maybe_decimal(r * unit)

    @named_scope(
        'brainstate/random',
        static_argnums=(0, 2, 3),
        static_argnames=['axis', 'independent']
    )
    def permutation(
        self,
        x,
        axis: int = 0,
        independent: bool = False,
        key: Optional[SeedOrKey] = None
    ):
        x = _check_py_seq(x)
        x, unit = u.split_mantissa_unit(x)
        key = self.__get_key(key)
        r = jr.permutation(key, x, axis, independent=independent)
        return u.maybe_decimal(r * unit)

    @named_scope('brainstate/random', static_argnums=(0, 2), static_argnames=['axis'])
    def shuffle(
        self,
        x,
        axis=0,
        key: Optional[SeedOrKey] = None
    ):
        return self.permutation(x, axis=axis, key=key, independent=False)

    @named_scope('brainstate/random', static_argnums=(0, 3, 5), static_argnames=['dtype', 'size'])
    def beta(
        self,
        a,
        b,
        size: Optional[Size] = None,
        key: Optional[SeedOrKey] = None,
        dtype: DTypeLike = None
    ):
        a = _remove_unit_param('a', _check_py_seq(a))
        b = _remove_unit_param('b', _check_py_seq(b))
        if size is None:
            size = lax.broadcast_shapes(u.math.shape(a), u.math.shape(b))
        key = self.__get_key(key)
        r = jr.beta(key, a=a, b=b, shape=_size2shape(size), dtype=dtype or environ.dftype())
        return r

    @named_scope('brainstate/random', static_argnums=(0, 2, 4), static_argnames=['dtype', 'size'])
    def exponential(
        self,
        scale=None,
        size: Optional[Size] = None,
        key: Optional[SeedOrKey] = None,
        dtype: DTypeLike = None
    ):
        scale_m, unit = _scale_unit(_check_py_seq(scale))
        if size is None:
            size = u.math.shape(scale_m) if scale_m is not None else ()
        key = self.__get_key(key)
        dtype = dtype or environ.dftype()
        r = jr.exponential(key, shape=_size2shape(size), dtype=dtype)
        if scale_m is not None:
            # ``scale`` is the numpy-compatible scale parameter beta = 1 / lambda,
            # i.e. the distribution mean. A standard exponential has mean 1, so the
            # draw is multiplied (not divided) by ``scale`` to reach mean ``scale``.
            r = r * u.math.asarray(scale_m, dtype=dtype)
        return u.maybe_decimal(r * unit)

    @named_scope('brainstate/random', static_argnums=(0, 3, 5), static_argnames=['dtype', 'size'])
    def gamma(
        self,
        shape,
        scale=None,
        size: Optional[Size] = None,
        key: Optional[SeedOrKey] = None,
        dtype: DTypeLike = None
    ):
        shape = _remove_unit_param('shape', _check_py_seq(shape))
        scale_m, unit = _scale_unit(_check_py_seq(scale))
        if size is None:
            size = lax.broadcast_shapes(
                u.math.shape(shape),
                u.math.shape(scale_m) if scale_m is not None else ()
            )
        key = self.__get_key(key)
        r = jr.gamma(key, a=shape, shape=_size2shape(size), dtype=dtype or environ.dftype())
        if scale_m is not None:
            r = r * scale_m
        return u.maybe_decimal(r * unit)

    @named_scope('brainstate/random', static_argnums=(0, 3, 5), static_argnames=['dtype', 'size'])
    def gumbel(
        self,
        loc=None,
        scale=None,
        size: Optional[Size] = None,
        key: Optional[SeedOrKey] = None,
        dtype: DTypeLike = None
    ):
        loc, scale, unit = _loc_scale_unit(_check_py_seq(loc), _check_py_seq(scale))
        if size is None:
            size = lax.broadcast_shapes(
                u.math.shape(loc) if loc is not None else (),
                u.math.shape(scale) if scale is not None else ()
            )
        key = self.__get_key(key)
        r = _loc_scale(loc, scale, jr.gumbel(key, shape=_size2shape(size), dtype=dtype or environ.dftype()))
        return u.maybe_decimal(r * unit)

    @named_scope('brainstate/random', static_argnums=(0, 3, 5), static_argnames=['dtype', 'size'])
    def laplace(
        self,
        loc=None,
        scale=None,
        size: Optional[Size] = None,
        key: Optional[SeedOrKey] = None,
        dtype: DTypeLike = None
    ):
        loc, scale, unit = _loc_scale_unit(_check_py_seq(loc), _check_py_seq(scale))
        if size is None:
            size = lax.broadcast_shapes(
                u.math.shape(loc) if loc is not None else (),
                u.math.shape(scale) if scale is not None else ()
            )
        key = self.__get_key(key)
        r = _loc_scale(loc, scale, jr.laplace(key, shape=_size2shape(size), dtype=dtype or environ.dftype()))
        return u.maybe_decimal(r * unit)

    @named_scope('brainstate/random', static_argnums=(0, 3, 5), static_argnames=['dtype', 'size'])
    def logistic(
        self,
        loc=None,
        scale=None,
        size: Optional[Size] = None,
        key: Optional[SeedOrKey] = None,
        dtype: DTypeLike = None
    ):
        loc, scale, unit = _loc_scale_unit(_check_py_seq(loc), _check_py_seq(scale))
        if size is None:
            size = lax.broadcast_shapes(
                u.math.shape(loc) if loc is not None else (),
                u.math.shape(scale) if scale is not None else ()
            )
        key = self.__get_key(key)
        r = _loc_scale(loc, scale, jr.logistic(key, shape=_size2shape(size), dtype=dtype or environ.dftype()))
        return u.maybe_decimal(r * unit)

    @named_scope('brainstate/random', static_argnums=(0, 3, 5), static_argnames=['dtype', 'size'])
    def normal(
        self,
        loc=None,
        scale=None,
        size: Optional[Size] = None,
        key: Optional[SeedOrKey] = None,
        dtype: DTypeLike = None
    ):
        loc, scale, unit = _loc_scale_unit(_check_py_seq(loc), _check_py_seq(scale))
        if size is None:
            size = lax.broadcast_shapes(
                u.math.shape(scale) if scale is not None else (),
                u.math.shape(loc) if loc is not None else ()
            )
        key = self.__get_key(key)
        dtype = dtype or environ.dftype()
        r = _loc_scale(loc, scale, jr.normal(key, shape=_size2shape(size), dtype=dtype))
        return u.maybe_decimal(r * unit)

    @named_scope('brainstate/random', static_argnums=(0, 2, 4), static_argnames=['dtype', 'size'])
    def pareto(
        self,
        a,
        size: Optional[Size] = None,
        key: Optional[SeedOrKey] = None,
        dtype: DTypeLike = None
    ):
        a = _remove_unit_param('a', a)
        if size is None:
            size = u.math.shape(a)
        key = self.__get_key(key)
        dtype = dtype or environ.dftype()
        a = u.math.asarray(a, dtype=dtype)
        r = jr.pareto(key, b=a, shape=_size2shape(size), dtype=dtype)
        return r

    @named_scope('brainstate/random', static_argnums=(0, 2, 4), static_argnames=['dtype', 'size'])
    def poisson(
        self,
        lam=1.0,
        size: Optional[Size] = None,
        key: Optional[SeedOrKey] = None,
        dtype: DTypeLike = None
    ):
        lam = _remove_unit_param('lam', _check_py_seq(lam))
        if size is None:
            size = u.math.shape(lam)
        key = self.__get_key(key)
        dtype = dtype or environ.ditype()
        r = jr.poisson(key, lam=lam, shape=_size2shape(size), dtype=dtype)
        return r

    @named_scope('brainstate/random', static_argnums=(0, 1, 3), static_argnames=['dtype', 'size'])
    def standard_cauchy(
        self,
        size: Optional[Size] = None,
        key: Optional[SeedOrKey] = None,
        dtype: DTypeLike = None
    ):
        key = self.__get_key(key)
        dtype = dtype or environ.dftype()
        r = jr.cauchy(key, shape=_size2shape(size), dtype=dtype)
        return r

    @named_scope('brainstate/random', static_argnums=(0, 1, 3), static_argnames=['dtype', 'size'])
    def standard_exponential(
        self,
        size: Optional[Size] = None,
        key: Optional[SeedOrKey] = None,
        dtype: DTypeLike = None
    ):
        key = self.__get_key(key)
        dtype = dtype or environ.dftype()
        r = jr.exponential(key, shape=_size2shape(size), dtype=dtype)
        return r

    @named_scope('brainstate/random', static_argnums=(0, 2, 4), static_argnames=['dtype', 'size'])
    def standard_gamma(
        self,
        shape,
        size: Optional[Size] = None,
        key: Optional[SeedOrKey] = None,
        dtype: DTypeLike = None
    ):
        shape = _remove_unit_param('shape', _check_py_seq(shape))
        if size is None:
            size = u.math.shape(shape) if shape is not None else ()
        key = self.__get_key(key)
        dtype = dtype or environ.dftype()
        r = jr.gamma(key, a=shape, shape=_size2shape(size), dtype=dtype)
        return r

    @named_scope('brainstate/random', static_argnums=(0, 1, 3), static_argnames=['dtype', 'size'])
    def standard_normal(
        self,
        size: Optional[Size] = None,
        key: Optional[SeedOrKey] = None,
        dtype: DTypeLike = None
    ):
        key = self.__get_key(key)
        dtype = dtype or environ.dftype()
        r = jr.normal(key, shape=_size2shape(size), dtype=dtype)
        return r

    @named_scope('brainstate/random', static_argnums=(0, 2, 4), static_argnames=['dtype', 'size'])
    def standard_t(
        self,
        df,
        size: Optional[Size] = None,
        key: Optional[SeedOrKey] = None,
        dtype: DTypeLike = None
    ):
        df = _remove_unit_param('df', _check_py_seq(df))
        if size is None:
            size = u.math.shape(size) if size is not None else ()
        key = self.__get_key(key)
        dtype = dtype or environ.dftype()
        r = jr.t(key, df=df, shape=_size2shape(size), dtype=dtype)
        return r

    @named_scope(
        'brainstate/random',
        static_argnums=(0, 3, 5),
        static_argnames=['dtype', 'size']
    )
    def uniform(
        self,
        low=0.0,
        high=1.0,
        size: Optional[Size] = None,
        key: Optional[SeedOrKey] = None,
        dtype: DTypeLike = None
    ):
        # ``low``/``high`` share one physical unit, inferred from whichever bound
        # carries one (a plain bound is then interpreted in that shared unit). A
        # compatible-but-different unit is converted; an incompatible one raises.
        low, high, unit = _loc_scale_unit(_check_py_seq(low), _check_py_seq(high))
        if size is None:
            size = lax.broadcast_shapes(u.math.shape(low), u.math.shape(high))
        key = self.__get_key(key)
        dtype = dtype or environ.dftype()
        r = jr.uniform(key, _size2shape(size), dtype=dtype, minval=low, maxval=high)
        return u.maybe_decimal(r * unit)

    def __norm_cdf(self, x, sqrt2, dtype):
        # Computes standard normal cumulative distribution function
        return (np.asarray(1., dtype) + lax.erf(x / sqrt2)) / np.asarray(2., dtype)

    @named_scope(
        'brainstate/random',
        static_argnums=(0, 3, 7, 8),
        static_argnames=['dtype', 'size', 'check_valid'],
    )
    def truncated_normal(
        self,
        lower,
        upper,
        size: Optional[Size] = None,
        loc=0.0,
        scale=1.0,
        key: Optional[SeedOrKey] = None,
        dtype: DTypeLike = None,
        check_valid: bool = True
    ):
        dtype = dtype or environ.dftype()
        # ``lower``/``upper``/``loc``/``scale`` share a single physical unit. Infer it
        # from whichever parameter carries one; a plain value is then interpreted as
        # already being expressed in that shared unit. A compatible-but-different unit
        # (e.g. ``volt`` against ``mV``) is converted; an incompatible one raises.
        values = [
            u.math.asarray(_check_py_seq(v), dtype=dtype)
            for v in (lower, upper, loc, scale)
        ]
        unit = u.UNITLESS
        for v in values:
            q = u.Quantity(v)
            if not q.is_unitless:
                unit = q.unit
                break

        def _to_shared_unit(v):
            q = u.Quantity(v)
            return q.mantissa if q.is_unitless else q.in_unit(unit).mantissa

        lower, upper, loc, scale = (_to_shared_unit(v) for v in values)

        if check_valid:
            from brainstate.transform._error_if import jit_error_if
            jit_error_if(
                u.math.any(u.math.logical_or(loc < lower - 2 * scale, loc > upper + 2 * scale)),
                "mean is more than 2 std from [lower, upper] in truncated_normal. "
                "The distribution of values may be incorrect."
            )

        if size is None:
            size = u.math.broadcast_shapes(
                u.math.shape(lower),
                u.math.shape(upper),
                u.math.shape(loc),
                u.math.shape(scale)
            )

        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        sqrt2 = np.array(np.sqrt(2), dtype=dtype)
        l = self.__norm_cdf((lower - loc) / scale, sqrt2, dtype)
        u_ = self.__norm_cdf((upper - loc) / scale, sqrt2, dtype)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        key = self.__get_key(key)
        out = jr.uniform(
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
        return u.maybe_decimal(out * unit)

    def _check_p(self, *args, **kwargs):
        raise ValueError('Parameter p should be within [0, 1], but we got {p}')

    @named_scope(
        'brainstate/random',
        static_argnums=(0, 2, 4),
        static_argnames=['size', 'check_valid']
    )
    def bernoulli(
        self,
        p,
        size: Optional[Size] = None,
        key: Optional[SeedOrKey] = None,
        check_valid: bool = True
    ):
        p = _remove_unit_param('p', _check_py_seq(p))
        if check_valid:
            from brainstate.transform._error_if import jit_error_if
            jit_error_if(jnp.any(jnp.logical_or(p < 0, p > 1)), self._check_p, p=p)
        if size is None:
            size = u.math.shape(p)
        key = self.__get_key(key)
        r = jr.bernoulli(key, p=p, shape=_size2shape(size))
        return r

    @named_scope(
        'brainstate/random',
        static_argnums=(0, 3, 5),
        static_argnames=['dtype', 'size']
    )
    def lognormal(
        self,
        mean=None,
        sigma=None,
        size: Optional[Size] = None,
        key: Optional[SeedOrKey] = None,
        dtype: DTypeLike = None
    ):
        dtype = dtype or environ.dftype()
        # A lognormal sample ``X`` satisfies ``ln(X) ~ normal(mean, sigma)``; the
        # logarithm is only defined for a dimensionless argument, so ``mean`` and
        # ``sigma`` must be dimensionless and the output is dimensionless.
        mean = _remove_unit_param('mean', _check_py_seq(mean), dtype=dtype)
        sigma = _remove_unit_param('sigma', _check_py_seq(sigma), dtype=dtype)

        if size is None:
            size = jnp.broadcast_shapes(
                u.math.shape(mean) if mean is not None else (),
                u.math.shape(sigma) if sigma is not None else ()
            )
        key = self.__get_key(key)
        samples = jr.normal(key, shape=_size2shape(size), dtype=dtype)
        samples = _loc_scale(mean, sigma, samples)
        return jnp.exp(samples)

    @named_scope(
        'brainstate/random',
        static_argnums=(0, 3, 5, 6),
        static_argnames=['dtype', 'size', 'check_valid']
    )
    def binomial(
        self,
        n,
        p,
        size: Optional[Size] = None,
        key: Optional[SeedOrKey] = None,
        dtype: DTypeLike = None,
        check_valid: bool = True
    ):
        n = _remove_unit_param('n', _check_py_seq(n))
        p = _remove_unit_param('p', _check_py_seq(p))
        if check_valid:
            from brainstate.transform._error_if import jit_error_if
            jit_error_if(
                jnp.any(jnp.logical_or(p < 0, p > 1)),
                'Parameter p should be within [0, 1], but we got {p}',
                p=p
            )
        if size is None:
            size = jnp.broadcast_shapes(u.math.shape(n), u.math.shape(p))
        key = self.__get_key(key)
        r = jr.binomial(key, n, p, shape=_size2shape(size))
        dtype = dtype or environ.ditype()
        return u.math.asarray(r, dtype=dtype)

    @named_scope(
        'brainstate/random',
        static_argnums=(0, 2, 4),
        static_argnames=['dtype', 'size']
    )
    def chisquare(
        self,
        df,
        size: Optional[Size] = None,
        key: Optional[SeedOrKey] = None,
        dtype: DTypeLike = None
    ):
        df = _remove_unit_param('df', _check_py_seq(df))
        key = self.__get_key(key)
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

    @named_scope(
        'brainstate/random',
        static_argnums=(0, 2, 4),
        static_argnames=['dtype', 'size']
    )
    def dirichlet(
        self,
        alpha,
        size: Optional[Size] = None,
        key: Optional[SeedOrKey] = None,
        dtype: DTypeLike = None
    ):
        key = self.__get_key(key)
        alpha = _remove_unit_param('alpha', _check_py_seq(alpha))
        dtype = dtype or environ.dftype()
        r = jr.dirichlet(key, alpha=alpha, shape=_size2shape(size), dtype=dtype)
        return r

    @named_scope(
        'brainstate/random',
        static_argnums=(0, 2, 4),
        static_argnames=['dtype', 'size']
    )
    def geometric(
        self,
        p,
        size: Optional[Size] = None,
        key: Optional[SeedOrKey] = None,
        dtype: DTypeLike = None
    ):
        p = _remove_unit_param('p', _check_py_seq(p))
        if size is None:
            size = u.math.shape(p)
        key = self.__get_key(key)
        dtype = dtype or environ.dftype()
        u_ = jr.uniform(key, size, dtype)
        r = jnp.floor(jnp.log1p(-u_) / jnp.log1p(-p))
        return r

    def _check_p2(self, p):
        raise ValueError(f'We require `sum(pvals[:-1]) <= 1`. But we got {p}')

    @named_scope(
        'brainstate/random',
        static_argnums=(0, 3, 5, 6),
        static_argnames=['dtype', 'size', 'check_valid']
    )
    def multinomial(
        self,
        n,
        pvals,
        size: Optional[Size] = None,
        key: Optional[SeedOrKey] = None,
        dtype: DTypeLike = None,
        check_valid: bool = True
    ):
        key = self.__get_key(key)
        n = _remove_unit_param('n', _check_py_seq(n))
        pvals = _remove_unit_param('pvals', _check_py_seq(pvals))
        if check_valid:
            from brainstate.transform._error_if import jit_error_if
            jit_error_if(jnp.sum(pvals[:-1]) > 1., self._check_p2, pvals)
        if isinstance(n, Tracer):
            raise ValueError("The total count parameter `n` should not be a jax abstract array.")
        size = _size2shape(size)
        n_max = int(np.max(jax.device_get(n)))
        batch_shape = lax.broadcast_shapes(u.math.shape(pvals)[:-1], u.math.shape(n))
        r = multinomial(key, pvals, n, n_max=n_max, shape=batch_shape + size)
        dtype = dtype or environ.ditype()
        return u.math.asarray(r, dtype=dtype)

    @named_scope(
        'brainstate/random',
        static_argnums=(0, 3, 4, 6),
        static_argnames=['dtype', 'size', 'method']
    )
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
            assert isinstance(cov, u.Quantity), 'cov must carry a unit when mean does.'
            assert mean.unit ** 2 == cov.unit, 'cov unit must equal mean unit squared.'
        # Capture the output unit *before* stripping the mantissa from ``mean``.
        unit = mean.unit if isinstance(mean, u.Quantity) else u.UNITLESS
        mean = mean.mantissa if isinstance(mean, u.Quantity) else mean
        cov = cov.mantissa if isinstance(cov, u.Quantity) else cov

        key = self.__get_key(key)
        if not jnp.ndim(mean) >= 1:
            raise ValueError(f"multivariate_normal requires mean.ndim >= 1, got mean.ndim == {jnp.ndim(mean)}")
        if not jnp.ndim(cov) >= 2:
            raise ValueError(f"multivariate_normal requires cov.ndim >= 2, got cov.ndim == {jnp.ndim(cov)}")
        n = mean.shape[-1]
        if u.math.shape(cov)[-2:] != (n, n):
            raise ValueError(f"multivariate_normal requires cov.shape == (..., n, n) for n={n}, "
                             f"but got cov.shape == {u.math.shape(cov)}.")
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
        return u.maybe_decimal(r * unit)

    @named_scope(
        'brainstate/random',
        static_argnums=(0, 2, 4),
        static_argnames=['dtype', 'size']
    )
    def rayleigh(
        self,
        scale=1.0,
        size: Optional[Size] = None,
        key: Optional[SeedOrKey] = None,
        dtype: DTypeLike = None
    ):
        scale_m, unit = _scale_unit(_check_py_seq(scale))
        if size is None:
            size = u.math.shape(scale_m) if scale_m is not None else ()
        key = self.__get_key(key)
        dtype = dtype or environ.dftype()
        x = jnp.sqrt(-2. * jnp.log(jr.uniform(key, shape=_size2shape(size), dtype=dtype)))
        r = x * scale_m
        return u.maybe_decimal(r * unit)

    @named_scope(
        'brainstate/random',
        static_argnums=(0, 1),
        static_argnames=['size']
    )
    def triangular(
        self,
        size: Optional[Size] = None,
        key: Optional[SeedOrKey] = None
    ):
        key = self.__get_key(key)
        bernoulli_samples = jr.bernoulli(key, p=0.5, shape=_size2shape(size))
        r = 2 * bernoulli_samples - 1
        return r

    @named_scope(
        'brainstate/random',
        static_argnums=(0, 3, 5),
        static_argnames=['dtype', 'size']
    )
    def vonmises(
        self,
        mu,
        kappa,
        size: Optional[Size] = None,
        key: Optional[SeedOrKey] = None,
        dtype: DTypeLike = None
    ):
        key = self.__get_key(key)
        dtype = dtype or environ.dftype()
        # ``mu`` is an angle and may carry an angular unit (e.g. radian); the output
        # is an angle in ``[-pi, pi]`` carrying that same unit. ``kappa`` is a
        # dimensionless concentration.
        mu_m, unit = _scale_unit(_check_py_seq(mu))
        mu_m = u.math.asarray(mu_m, dtype=dtype)
        kappa = u.math.asarray(_remove_unit_param('kappa', _check_py_seq(kappa)), dtype=dtype)
        if size is None:
            size = lax.broadcast_shapes(u.math.shape(mu_m), u.math.shape(kappa))
        size = _size2shape(size)
        samples = von_mises_centered(key, kappa, size, dtype=dtype)
        samples = samples + mu_m
        samples = (samples + jnp.pi) % (2.0 * jnp.pi) - jnp.pi
        return u.maybe_decimal(samples * unit)

    @named_scope(
        'brainstate/random',
        static_argnums=(0, 2, 4),
        static_argnames=['dtype', 'size']
    )
    def weibull(
        self,
        a,
        size: Optional[Size] = None,
        key: Optional[SeedOrKey] = None,
        dtype: DTypeLike = None
    ):
        key = self.__get_key(key)
        a = _remove_unit_param('a', _check_py_seq(a))
        if size is None:
            size = u.math.shape(a)
        else:
            if jnp.size(a) > 1:
                raise ValueError(f'"a" should be a scalar when "size" is provided. But we got {a}')
        size = _size2shape(size)
        dtype = dtype or environ.dftype()
        random_uniform = jr.uniform(key=key, shape=size, dtype=dtype)
        r = jnp.power(-jnp.log1p(-random_uniform), 1.0 / a)
        return r

    @named_scope(
        'brainstate/random',
        static_argnums=(0, 3, 5),
        static_argnames=['dtype', 'size']
    )
    def weibull_min(
        self,
        a,
        scale=None,
        size: Optional[Size] = None,
        key: Optional[SeedOrKey] = None,
        dtype: DTypeLike = None
    ):
        key = self.__get_key(key)
        a = _remove_unit_param('a', _check_py_seq(a))
        scale_m, unit = _scale_unit(_check_py_seq(scale))
        if size is None:
            size = jnp.broadcast_shapes(u.math.shape(a), u.math.shape(scale_m) if scale_m is not None else ())
        else:
            if jnp.size(a) > 1:
                raise ValueError(f'"a" should be a scalar when "size" is provided. But we got {a}')
        size = _size2shape(size)
        dtype = dtype or environ.dftype()
        random_uniform = jr.uniform(key=key, shape=size, dtype=dtype)
        r = jnp.power(-jnp.log1p(-random_uniform), 1.0 / a)
        if scale_m is not None:
            r = r / scale_m
        return u.maybe_decimal(r * unit)

    @named_scope(
        'brainstate/random',
        static_argnums=(0, 1, 3),
        static_argnames=['dtype', 'size']
    )
    def maxwell(
        self,
        size: Optional[Size] = None,
        key: Optional[SeedOrKey] = None,
        dtype: DTypeLike = None
    ):
        key = self.__get_key(key)
        shape = _size2shape(size) + (3,)
        dtype = dtype or environ.dftype()
        norm_rvs = jr.normal(key=key, shape=shape, dtype=dtype)
        r = jnp.linalg.norm(norm_rvs, axis=-1)
        return r

    @named_scope(
        'brainstate/random',
        static_argnums=(0, 3, 5),
        static_argnames=['dtype', 'size']
    )
    def negative_binomial(
        self,
        n,
        p,
        size: Optional[Size] = None,
        key: Optional[SeedOrKey] = None,
        dtype: DTypeLike = None
    ):
        n = _remove_unit_param('n', _check_py_seq(n))
        p = _remove_unit_param('p', _check_py_seq(p))
        if size is None:
            size = lax.broadcast_shapes(u.math.shape(n), u.math.shape(p))
        size = _size2shape(size)
        logits = jnp.log(p) - jnp.log1p(-p)
        if key is None:
            keys = self.split_key(2)
        else:
            keys = jr.split(_format_key(key), 2)
        rate = self.gamma(shape=n, scale=jnp.exp(-logits), size=size, key=keys[0], dtype=environ.dftype())
        r = self.poisson(lam=rate, key=keys[1], dtype=dtype or environ.ditype())
        return r

    @named_scope(
        'brainstate/random',
        static_argnums=(0, 3, 5),
        static_argnames=['dtype', 'size']
    )
    def wald(
        self,
        mean,
        scale,
        size: Optional[Size] = None,
        key: Optional[SeedOrKey] = None,
        dtype: DTypeLike = None
    ):
        dtype = dtype or environ.dftype()
        key = self.__get_key(key)
        mean, scale, unit = _loc_scale_unit(_check_py_seq(mean), _check_py_seq(scale))
        mean = u.math.asarray(mean, dtype=dtype)
        scale = u.math.asarray(scale, dtype=dtype)
        if size is None:
            size = lax.broadcast_shapes(u.math.shape(mean), u.math.shape(scale))
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
        return u.maybe_decimal(res * unit)

    @named_scope('brainstate/random', static_argnums=(0, 2, 4), static_argnames=['dtype', 'size'])
    def t(
        self,
        df,
        size: Optional[Size] = None,
        key: Optional[SeedOrKey] = None,
        dtype: DTypeLike = None
    ):
        dtype = dtype or environ.dftype()
        df = u.math.asarray(_remove_unit_param('df', _check_py_seq(df)), dtype=dtype)
        if size is None:
            size = np.shape(df)
        else:
            size = _size2shape(size)
            _check_shape("t", size, np.shape(df))
        if key is None:
            keys = self.split_key(2)
        else:
            keys = jr.split(_format_key(key), 2)
        n = jr.normal(keys[0], size, dtype=dtype)
        two = const(n, 2)
        half_df = lax.div(df, two)
        g = jr.gamma(keys[1], half_df, size, dtype=dtype)
        r = n * jnp.sqrt(half_df / g)
        return r

    @named_scope('brainstate/random', static_argnums=(0, 1, 2, 4), static_argnames=['n', 'dtype', 'size'])
    def orthogonal(
        self,
        n: int,
        size: Optional[Size] = None,
        key: Optional[SeedOrKey] = None,
        dtype: DTypeLike = None
    ):
        dtype = dtype or environ.dftype()
        key = self.__get_key(key)
        size = _size2shape(size)
        _check_shape("orthogonal", size)
        n = concrete_or_error(index, n, "The error occurred in jax.random.orthogonal()")
        z = jr.normal(key, size + (n, n), dtype=dtype)
        q, r = jnp.linalg.qr(z)
        d = jnp.diagonal(r, 0, -2, -1)
        r = q * jnp.expand_dims(d / abs(d), -2)
        return r

    @named_scope('brainstate/random', static_argnums=(0, 3, 5), static_argnames=['dtype', 'size'])
    def noncentral_chisquare(
        self,
        df,
        nonc,
        size: Optional[Size] = None,
        key: Optional[SeedOrKey] = None,
        dtype: DTypeLike = None
    ):
        dtype = dtype or environ.dftype()
        df = u.math.asarray(_remove_unit_param('df', _check_py_seq(df)), dtype=dtype)
        nonc = u.math.asarray(_remove_unit_param('nonc', _check_py_seq(nonc)), dtype=dtype)
        if size is None:
            size = lax.broadcast_shapes(u.math.shape(df), u.math.shape(nonc))
        size = _size2shape(size)
        if key is None:
            keys = self.split_key(3)
        else:
            keys = jr.split(_format_key(key), 3)
        i = jr.poisson(keys[0], 0.5 * nonc, shape=size, dtype=environ.ditype())
        n = jr.normal(keys[1], shape=size, dtype=dtype) + jnp.sqrt(nonc)
        cond = jnp.greater(df, 1.0)
        df2 = jnp.where(cond, df - 1.0, df + 2.0 * i)
        chi2 = 2.0 * jr.gamma(keys[2], 0.5 * df2, shape=size, dtype=dtype)
        r = jnp.where(cond, chi2 + n * n, chi2)
        return r

    @named_scope('brainstate/random', static_argnums=(0, 2, 4), static_argnames=['dtype', 'size'])
    def loggamma(
        self,
        a,
        size: Optional[Size] = None,
        key: Optional[SeedOrKey] = None,
        dtype: DTypeLike = None
    ):
        dtype = dtype or environ.dftype()
        key = self.__get_key(key)
        a = _remove_unit_param('a', _check_py_seq(a))
        if size is None:
            size = u.math.shape(a)
        r = jr.loggamma(key, a, shape=_size2shape(size), dtype=dtype)
        return r

    @named_scope('brainstate/random', static_argnums=(0, 2, 3), static_argnames=['dtype', 'size', 'axis'])
    def categorical(
        self,
        logits,
        axis: int = -1,
        size: Optional[Size] = None,
        key: Optional[SeedOrKey] = None
    ):
        key = self.__get_key(key)
        logits = _remove_unit_param('logits', _check_py_seq(logits))
        if size is None:
            size = list(u.math.shape(logits))
            size.pop(axis)
        r = jr.categorical(key, logits, axis=axis, shape=_size2shape(size))
        return r

    @named_scope('brainstate/random', static_argnums=(0, 2, 4), static_argnames=['dtype', 'size'])
    def zipf(
        self,
        a,
        size: Optional[Size] = None,
        key: Optional[SeedOrKey] = None,
        dtype: DTypeLike = None
    ):
        a = _remove_unit_param('a', _check_py_seq(a))
        if size is None:
            size = u.math.shape(a)
        r = zipf(
            self.__get_key(key),
            a,
            shape=size,
            dtype=dtype or environ.ditype()
        )
        return r

    @named_scope('brainstate/random', static_argnums=(0, 2, 4), static_argnames=['dtype', 'size'])
    def power(
        self,
        a,
        size: Optional[Size] = None,
        key: Optional[SeedOrKey] = None,
        dtype: DTypeLike = None
    ):
        a = _remove_unit_param('a', _check_py_seq(a))
        if size is None:
            size = u.math.shape(a)
        size = _size2shape(size)
        r = power(
            self.__get_key(key),
            a,
            shape=size,
            dtype=dtype or environ.dftype(),
        )
        return r

    @named_scope('brainstate/random', static_argnums=(0, 3, 5), static_argnames=['dtype', 'size'])
    def f(
        self,
        dfnum,
        dfden,
        size: Optional[Size] = None,
        key: Optional[SeedOrKey] = None,
        dtype: DTypeLike = None
    ):
        dfnum = _remove_unit_param('dfnum', _check_py_seq(dfnum))
        dfden = _remove_unit_param('dfden', _check_py_seq(dfden))
        if size is None:
            size = jnp.broadcast_shapes(u.math.shape(dfnum), u.math.shape(dfden))
        size = _size2shape(size)
        r = f(
            self.__get_key(key),
            dfnum,
            dfden,
            shape=size,
            dtype=dtype or environ.dftype(),
        )
        return r

    @named_scope('brainstate/random', static_argnums=(0, 4, 6), static_argnames=['dtype', 'size'])
    def hypergeometric(
        self,
        ngood,
        nbad,
        nsample,
        size: Optional[Size] = None,
        key: Optional[SeedOrKey] = None,
        dtype: DTypeLike = None
    ):
        ngood = _remove_unit_param('ngood', _check_py_seq(ngood))
        nbad = _remove_unit_param('nbad', _check_py_seq(nbad))
        nsample = _remove_unit_param('nsample', _check_py_seq(nsample))
        if size is None:
            size = lax.broadcast_shapes(
                u.math.shape(ngood),
                u.math.shape(nbad),
                u.math.shape(nsample)
            )
        size = _size2shape(size)
        r = hypergeometric(
            self.__get_key(key),
            ngood,
            nbad,
            nsample,
            shape=size,
            dtype=dtype or environ.ditype(),
        )
        return r

    @named_scope('brainstate/random', static_argnums=(0, 2, 4), static_argnames=['dtype', 'size'])
    def logseries(
        self,
        p,
        size: Optional[Size] = None,
        key: Optional[SeedOrKey] = None,
        dtype: DTypeLike = None
    ):
        p = _remove_unit_param('p', _check_py_seq(p))
        if size is None:
            size = u.math.shape(p)
        size = _size2shape(size)
        r = logseries(
            self.__get_key(key),
            p,
            shape=size,
            dtype=dtype or environ.ditype()
        )
        return r

    @named_scope('brainstate/random', static_argnums=(0, 4, 6), static_argnames=['dtype', 'size'])
    def noncentral_f(
        self,
        dfnum,
        dfden,
        nonc,
        size: Optional[Size] = None,
        key: Optional[SeedOrKey] = None,
        dtype: DTypeLike = None
    ):
        dfnum = _remove_unit_param('dfnum', _check_py_seq(dfnum))
        dfden = _remove_unit_param('dfden', _check_py_seq(dfden))
        nonc = _remove_unit_param('nonc', _check_py_seq(nonc))
        if size is None:
            size = lax.broadcast_shapes(u.math.shape(dfnum),
                                        u.math.shape(dfden),
                                        u.math.shape(nonc))
        size = _size2shape(size)
        r = noncentral_f(
            self.__get_key(key),
            dfnum,
            dfden,
            nonc,
            shape=size,
            dtype=dtype or environ.dftype(),
        )
        return r

    # PyTorch compatibility #
    # --------------------- #

    @named_scope('brainstate/random', static_argnums=(0, 2), static_argnames=['dtype'])
    def rand_like(
        self,
        input,
        dtype: DTypeLike = None,
        key: Optional[SeedOrKey] = None
    ):
        """Returns a tensor with the same size as input that is filled with random
        numbers from a uniform distribution on the interval ``[0, 1)``.

        Args:
          input:  the ``size`` of input will determine size of the output tensor.
          dtype:  the desired data type of returned Tensor. Default: if ``None``, defaults to the dtype of input.
          key: the seed or key for the random.

        Returns:
          The random data.
        """
        return self.random(u.math.shape(input), key=key).astype(dtype)

    @named_scope('brainstate/random', static_argnums=(0, 2), static_argnames=['dtype'])
    def randn_like(
        self,
        input,
        dtype: DTypeLike = None,
        key: Optional[SeedOrKey] = None
    ):
        """Returns a tensor with the same size as ``input`` that is filled with
        random numbers from a normal distribution with mean 0 and variance 1.

        Args:
          input:  the ``size`` of input will determine size of the output tensor.
          dtype:  the desired data type of returned Tensor. Default: if ``None``, defaults to the dtype of input.
          key: the seed or key for the random.

        Returns:
          The random data.
        """
        return self.randn(*u.math.shape(input), key=key).astype(dtype)

    @named_scope('brainstate/random', static_argnums=(0, 4), static_argnames=['dtype'])
    def randint_like(
        self,
        input,
        low=0,
        high=None,
        dtype: DTypeLike = None,
        key: Optional[SeedOrKey] = None
    ):
        if high is None:
            high = max(input)
        return self.randint(low, high, size=u.math.shape(input), dtype=dtype, key=key)


# default random generator
DEFAULT = RandomState(np.random.randint(0, 2 ** 32, size=2, dtype=np.uint32))
