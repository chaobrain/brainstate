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
from __future__ import annotations

from typing import Union, Callable, Optional

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from brainstate._state import ParamState
from brainstate._utils import set_module_as
from brainstate.compile import for_loop
from brainstate.init import param
from brainstate.nn._module import Module
from brainstate.random import RandomState
from brainstate.typing import ArrayLike
from ._misc import FloatScalar, IntScalar

__all__ = [
    'FixedProb',
]


class FixedProb(Module):
    """
    The FixedProb module implements a fixed probability connection with CSR sparse data structure.

    Parameters
    ----------
    n_pre : int
        Number of pre-synaptic neurons.
    n_post : int
        Number of post-synaptic neurons.
    prob : float
        Probability of connection.
    weight : float or callable or jax.Array or brainunit.Quantity
        Maximum synaptic conductance.
    allow_multi_conn : bool, optional
        Whether multiple connections are allowed from a single pre-synaptic neuron.
        Default is True, meaning that a value of ``a`` can be selected multiple times.
    prob : float
        Probability of connection.
    name : str, optional
        Name of the module.
    """

    __module__ = 'brainstate.event'

    def __init__(
        self,
        n_pre: IntScalar,
        n_post: IntScalar,
        prob: FloatScalar,
        weight: Union[Callable, ArrayLike],
        allow_multi_conn: bool = True,
        seed: Optional[int] = None,
        name: Optional[str] = None,
        grad_mode: str = 'vjp'
    ):
        super().__init__(name=name)
        self.n_pre = n_pre
        self.n_post = n_post
        self.in_size = n_pre
        self.out_size = n_post

        self.n_conn = int(n_post * prob)
        if self.n_conn < 1:
            raise ValueError(
                f"The number of connections must be at least 1. Got: int({n_post} * {prob}) = {self.n_conn}")

        assert grad_mode in ['vjp', 'jvp'], f"Unsupported grad_mode: {grad_mode}"
        self.grad_mode = grad_mode

        # indices of post connected neurons
        if allow_multi_conn:
            self.indices = np.random.RandomState(seed).randint(0, n_post, size=(self.n_pre, self.n_conn))
        else:
            rng = RandomState(seed)
            self.indices = for_loop(lambda i: rng.choice(n_post, size=(self.n_conn,), replace=False), np.arange(n_pre))
        self.indices = u.math.asarray(self.indices)

        # maximum synaptic conductance
        weight = param(weight, (self.n_pre, self.n_conn), allow_none=False)
        self.weight = ParamState(weight)

    def update(self, spk: jax.Array) -> Union[jax.Array, u.Quantity]:
        device_kind = jax.devices()[0].platform  # spk.device.device_kind
        if device_kind == 'cpu':
            return cpu_fixed_prob(self.indices,
                                  u.math.asarray(self.weight.value),
                                  u.math.asarray(spk),
                                  n_post=self.n_post,
                                  grad_mode=self.grad_mode)
        elif device_kind in ['gpu', 'tpu']:
            raise NotImplementedError()
        else:
            raise ValueError(f"Unsupported device: {device_kind}")


@set_module_as('brainstate.event')
def cpu_fixed_prob(
    indices: jax.Array,
    weight: Union[u.Quantity, jax.Array],
    spk: jax.Array,
    *,
    n_post: int,
    grad_mode: str = 'vjp'
) -> Union[u.Quantity, jax.Array]:
    """
    The FixedProb module implements a fixed probability connection with CSR sparse data structure.

    Parameters
    ----------
    n_post : int
        Number of post-synaptic neurons.
    weight : brainunit.Quantity or jax.Array
        Maximum synaptic conductance.
    spk : jax.Array
        Spike events.
    indices : jax.Array
        Indices of post connected neurons.
    grad_mode : str, optional
        Gradient mode. Default is 'vjp'. Can be 'vjp' or 'jvp'.

    Returns
    -------
    post_data : brainunit.Quantity or jax.Array
        Post synaptic data.
    """
    unit = u.get_unit(weight)
    weight = u.get_mantissa(weight)
    indices = jnp.asarray(indices)
    spk = jnp.asarray(spk)

    def mv(spk_vector):
        assert spk_vector.ndim == 1, f"spk must be 1D. Got: {spk.ndim}"
        if grad_mode == 'vjp':
            post_data = _cpu_event_fixed_prob_mv_vjp(indices, weight, spk_vector, n_post)
        elif grad_mode == 'jvp':
            post_data = _cpu_event_fixed_prob_mv_jvp(indices, weight, spk_vector, n_post)
        else:
            raise ValueError(f"Unsupported grad_mode: {grad_mode}")
        return post_data

    assert spk.ndim >= 1, f"spk must be at least 1D. Got: {spk.ndim}"
    assert weight.ndim in [2, 0], f"weight must be 2D or 0D. Got: {weight.ndim}"
    assert indices.ndim == 2, f"indices must be 2D. Got: {indices.ndim}"

    if spk.ndim == 1:
        post_data = mv(spk)
    else:
        shape = spk.shape[:-1]
        post_data = jax.vmap(mv)(u.math.reshape(spk, (-1, spk.shape[-1])))
        post_data = u.math.reshape(post_data, shape + post_data.shape[-1:])
    return u.maybe_decimal(u.Quantity(post_data, unit=unit))


# -------------------
# CPU Implementation
# -------------------


def _cpu_event_fixed_prob_mv(indices, g_max, spk, n_post: int) -> jax.Array:
    def scan_fn(post, i):
        w = g_max if jnp.size(g_max) == 1 else g_max[i]
        ids = indices[i]
        sp = spk[i]
        if spk.dtype == jnp.bool_:
            post = jax.lax.cond(sp, lambda: post.at[ids].add(w), lambda: post)
        else:
            post = jax.lax.cond(sp == 0., lambda: post, lambda: post.at[ids].add(w * sp))
        return post, None

    return jax.lax.scan(scan_fn, jnp.zeros((n_post,), dtype=g_max.dtype), np.arange(len(spk)))[0]


# --------------
# VJP
# --------------

def _cpu_event_fixed_prob_mv_fwd(indices, g_max, spk, n_post):
    return _cpu_event_fixed_prob_mv(indices, g_max, spk, n_post=n_post), (g_max, spk)


def _cpu_event_fixed_prob_mv_bwd(indices, n_post, res, ct):
    weight, spk = res

    # ∂L/∂spk = ∂L/∂y * ∂y/∂spk
    homo = jnp.size(weight) == 1
    if homo:  # homogeneous weight
        ct_spk = jax.vmap(lambda idx: jnp.sum(ct[idx] * weight))(indices)
    else:  # heterogeneous weight
        ct_spk = jax.vmap(lambda idx, w: jnp.inner(ct[idx], w))(indices, weight)

    # ∂L/∂w = ∂L/∂y * ∂y/∂w
    if homo:  # scalar
        ct_gmax = _cpu_event_fixed_prob_mv(indices, jnp.asarray(1.), spk, n_post=n_post)
        ct_gmax = jnp.inner(ct, ct_gmax)
    else:
        def scan_fn(d_gmax, i):
            if spk.dtype == jnp.bool_:
                d_gmax = jax.lax.cond(spk[i], lambda: d_gmax.at[i].add(ct[indices[i]]), lambda: d_gmax)
            else:
                d_gmax = jax.lax.cond(spk[i] == 0., lambda: d_gmax, lambda: d_gmax.at[i].add(ct[indices[i]] * spk[i]))
            return d_gmax, None

        ct_gmax = jax.lax.scan(scan_fn, jnp.zeros_like(weight), np.arange(len(spk)))[0]
    return ct_gmax, ct_spk


_cpu_event_fixed_prob_mv_vjp = jax.custom_vjp(_cpu_event_fixed_prob_mv, nondiff_argnums=(0, 3))
_cpu_event_fixed_prob_mv_vjp.defvjp(_cpu_event_fixed_prob_mv_fwd, _cpu_event_fixed_prob_mv_bwd)


# --------------
# JVP
# --------------


def _cpu_event_fixed_prob_mv_jvp_rule(indices, n_post, primals, tangents):
    # forward pass
    weight, spk = primals
    y = _cpu_event_fixed_prob_mv(indices, weight, spk, n_post=n_post)

    # forward gradients
    gmax_dot, spk_dot = tangents

    # ∂y/∂gmax
    dgmax = _cpu_event_fixed_prob_mv(indices, gmax_dot, spk, n_post=n_post)

    def scan_fn(post, i):
        ids = indices[i]
        w = weight if jnp.size(weight) == 1 else weight[i]
        post = post.at[ids].add(w * spk_dot[i])
        return post, None

    # ∂y/∂gspk
    dspk = jax.lax.scan(scan_fn, jnp.zeros((n_post,), dtype=weight.dtype), np.arange(len(spk)))[0]
    return y, dgmax + dspk


_cpu_event_fixed_prob_mv_jvp = jax.custom_jvp(_cpu_event_fixed_prob_mv, nondiff_argnums=(0, 3))
_cpu_event_fixed_prob_mv_jvp.defjvp(_cpu_event_fixed_prob_mv_jvp_rule)


def _gpu_event_fixed_prob_mv(indices, g_max, spk, n_post: int) -> jax.Array:
    def scan_fn(post, i):
        w = g_max if jnp.size(g_max) == 1 else g_max[i]
        ids = indices[i]
        sp = spk[i]
        if spk.dtype == jnp.bool_:
            post = jax.lax.cond(sp, lambda: post.at[ids].add(w), lambda: post)
        else:
            post = jax.lax.cond(sp == 0., lambda: post, lambda: post.at[ids].add(w * sp))
        return post, None

    return jax.lax.scan(scan_fn, jnp.zeros((n_post,), dtype=g_max.dtype), np.arange(len(spk)))[0]
