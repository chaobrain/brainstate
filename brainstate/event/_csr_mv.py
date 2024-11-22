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
from brainstate.init import param
from brainstate.nn._module import Module
from brainstate.typing import ArrayLike, Size

__all__ = [
    'CSRLinear',
]


class CSRLinear(Module):
    """
    The CSRLinear module implements a fixed probability connection with CSR sparse data structure.

    Parameters
    ----------
    in_size : Size
        Number of pre-synaptic neurons, i.e., input size.
    out_size : Size
        Number of post-synaptic neurons, i.e., output size.
    weight : float or callable or jax.Array or brainunit.Quantity
        Maximum synaptic conductance or a function that returns the maximum synaptic conductance.
    name : str, optional
        Name of the module.
    """

    __module__ = 'brainstate.event'

    def __init__(
        self,
        in_size: Size,
        out_size: Size,
        indptr: ArrayLike,
        indices: ArrayLike,
        weight: Union[Callable, ArrayLike],
        name: Optional[str] = None,
        grad_mode: str = 'vjp'
    ):
        super().__init__(name=name)

        # network size
        self.in_size = in_size
        self.out_size = out_size
        self.n_pre = self.in_size[-1]
        self.n_post = self.out_size[-1]

        # gradient mode
        assert grad_mode in ['vjp', 'jvp'], f"Unsupported grad_mode: {grad_mode}"
        self.grad_mode = grad_mode

        # CSR data structure
        indptr = jnp.asarray(indptr)
        indices = jnp.asarray(indices)
        assert indptr.ndim == 1, f"indptr must be 1D. Got: {indptr.ndim}"
        assert indices.ndim == 1, f"indices must be 1D. Got: {indices.ndim}"
        assert indptr.size == self.n_pre + 1, f"indptr must have size {self.n_pre + 1}. Got: {indptr.size}"
        with jax.ensure_compile_time_eval():
            self.indptr = u.math.asarray(indptr)
            self.indices = u.math.asarray(indices)

        # maximum synaptic conductance
        weight = param(weight, (len(indices),), allow_none=False)
        if u.math.size(weight) != 1 and u.math.size(weight) != len(self.indices):
            raise ValueError(f"weight must be 1D or 2D with size {len(self.indices)}. Got: {u.math.size(weight)}")
        self.weight = ParamState(weight)

    def update(self, spk: jax.Array) -> Union[jax.Array, u.Quantity]:
        weight = self.weight.value

        # return zero if no pre-synaptic neurons
        if len(self.indices) == 0:
            r = u.math.zeros(spk.shape[:-1] + (self.n_post,),
                             dtype=weight.dtype,
                             unit=u.get_unit(weight) * u.get_unit(spk))
            return u.maybe_decimal(r)

        device_kind = jax.devices()[0].platform  # spk.device.device_kind

        # CPU implementation
        if device_kind == 'cpu':
            return cpu_event_csr(
                u.math.asarray(spk),
                self.indptr,
                self.indices,
                u.math.asarray(weight),
                n_post=self.n_post, grad_mode=self.grad_mode
            )

        # GPU/TPU implementation
        elif device_kind in ['gpu', 'tpu']:
            raise NotImplementedError()

        else:
            raise ValueError(f"Unsupported device: {device_kind}")


@set_module_as('brainstate.event')
def cpu_event_csr(
    spk: jax.Array,
    indptr: jax.Array,
    indices: jax.Array,
    weight: Union[u.Quantity, jax.Array],
    *,
    n_post: int,
    grad_mode: str = 'vjp'
) -> Union[u.Quantity, jax.Array]:
    """
    The CSRLinear module implements a fixed probability connection with CSR sparse data structure.

    Parameters
    ----------
    spk : jax.Array
        Spike events.
    indptr : jax.Array
        Index pointer of post connected neurons.
    indices : jax.Array
        Indices of post connected neurons.
    weight : brainunit.Quantity or jax.Array
        Maximum synaptic conductance.
    n_post : int
        Number of post-synaptic neurons.
    grad_mode : str, optional
        Gradient mode. Default is 'vjp'. Can be 'vjp' or 'jvp'.

    Returns
    -------
    post_data : brainunit.Quantity or jax.Array
        Post synaptic data.
    """
    unit = u.get_unit(weight)
    weight = u.get_mantissa(weight)

    def mv(spk_vector):
        assert spk_vector.ndim == 1, f"spk must be 1D. Got: {spk.ndim}"
        if grad_mode == 'vjp':
            post_data = _cpu_event_csr_mv_vjp(spk_vector, indptr, indices, weight, n_post)
        elif grad_mode == 'jvp':
            post_data = _cpu_event_csr_mv_jvp(spk_vector, indptr, indices, weight, n_post)
        else:
            raise ValueError(f"Unsupported grad_mode: {grad_mode}")
        return post_data

    assert spk.ndim >= 1, f"spk must be at least 1D. Got: {spk.ndim}"
    assert weight.ndim in [1, 0], f"g_max must be 1D or 0D. Got: {weight.ndim}"
    assert indices.ndim == 1, f"indices must be 1D. Got: {indices.ndim}"

    if spk.ndim == 1:
        post_data = mv(spk)
    else:
        shape = spk.shape[:-1]
        post_data = jax.vmap(mv)(u.math.reshape(spk, (-1, spk.shape[-1])))
        post_data = u.math.reshape(post_data, shape + post_data.shape[-1:])
    return u.maybe_decimal(u.Quantity(post_data, unit=unit))


# --------------
# Implementation
# --------------


def _cpu_event_csr_mv(
    spk: jax.Array,
    indptr: jax.Array,
    indices: jax.Array,
    weight: Union[u.Quantity, jax.Array],
    n_post: int
) -> jax.Array:
    bool_x = spk.dtype == jnp.bool_
    homo_w = jnp.size(weight) == 1

    def add_fn(post_val, i_start, i_end, sp):
        def body_fn(x):
            post, i = x
            i_post = indices[i]
            w = weight if homo_w else weight[i]
            w = w if bool_x else w * sp
            post = post.at[i_post].add(w)
            return post, i + 1

        return jax.lax.while_loop(lambda x: x[1] < i_end, body_fn, (post_val, i_start))[0]

    def scan_fn(post, i):
        sp = spk[i]  # pre-synaptic spike event
        if bool_x:
            post = jax.lax.cond(sp, lambda: add_fn(post, indptr[i], indptr[i + 1], sp), lambda: post)
        else:
            post = jax.lax.cond(sp == 0., lambda: post, lambda: add_fn(post, indptr[i], indptr[i + 1], sp))
        return post, None

    return jax.lax.scan(scan_fn, jnp.zeros((n_post,), dtype=weight.dtype), np.arange(len(spk)))[0]


# --------------
# VJP
# --------------

def _cpu_event_csr_mv_fwd(
    spk: jax.Array,
    indptr: jax.Array,
    indices: jax.Array,
    weight: Union[u.Quantity, jax.Array],
    n_post: int
):
    return _cpu_event_csr_mv(spk, indptr, indices, weight, n_post=n_post), (spk, weight)


def _cpu_event_csr_mv_bwd(indptr, indices, n_post, res, ct):
    spk, weight = res
    homo = jnp.size(weight) == 1
    bool_spk = spk.dtype == jnp.bool_

    # ∂L/∂spk = ∂L/∂y * ∂y/∂spk
    def fn_spk(i_pre):
        def body_fn(x):
            r, i = x
            i_post = indices[i]
            r = r + (ct[i_post] if homo else ct[i_post] * weight[i])
            return r, i + 1

        p = jax.lax.while_loop(lambda x: x[1] < indptr[i_pre + 1], body_fn, (0., indptr[i_pre]))[0]
        p = p * weight if homo else p
        return p

    ct_spk = jax.vmap(fn_spk)(np.arange(len(spk)))

    # ∂L/∂w = ∂L/∂y * ∂y/∂w
    if homo:  # scalar
        ct_gmax = _cpu_event_csr_mv(spk, indptr, indices, jnp.asarray(1.), n_post=n_post)
        ct_gmax = jnp.inner(ct, ct_gmax)
    else:
        def single_post(dw, i_pre):
            def body_fn(x):
                dw, i = x
                i_post = indices[i]
                dw = dw.at[i].add(ct[i_post] if bool_spk else ct[i_post] * spk[i_pre])
                return dw, i + 1

            return jax.lax.while_loop(lambda x: x[1] < indptr[i_pre + 1], body_fn, (dw, indptr[i_pre]))[0]

        def fn_w(dw, i_pre):
            sp = spk[i_pre]
            if bool_spk:
                return jax.lax.cond(sp, lambda: single_post(dw, i_pre), lambda: dw), None
            else:
                return jax.lax.cond(sp == 0., lambda: dw, lambda: single_post(dw, i_pre)), None

        ct_gmax = jax.lax.scan(fn_w, jnp.zeros_like(weight), np.arange(len(spk)))[0]
    return ct_spk, ct_gmax


_cpu_event_csr_mv_vjp = jax.custom_vjp(_cpu_event_csr_mv, nondiff_argnums=(1, 2, 4))
_cpu_event_csr_mv_vjp.defvjp(_cpu_event_csr_mv_fwd, _cpu_event_csr_mv_bwd)


# --------------
# JVP
# --------------


def _cpu_event_csr_mv_jvp_rule(indptr, indices, n_post, primals, tangents):
    # forward pass
    spk, weight = primals
    y = _cpu_event_csr_mv(spk, indptr, indices, weight, n_post=n_post)

    # forward gradients
    spk_dot, weight_dot = tangents
    homo_w = jnp.size(weight) == 1

    # ∂y/∂gmax
    dweight = _cpu_event_csr_mv(spk, indptr, indices, weight_dot, n_post=n_post)

    # ∂y/∂gspk
    def scan_fn(post, i_pre):
        def while_fn(x):
            p, i, sp = x
            i_post = indices[i]
            p = p.at[i_post].add(sp if homo_w else sp * weight[i])
            return p, i + 1, sp

        post = jax.lax.while_loop(lambda x: x[1] < indptr[i_pre + 1],
                                  while_fn,
                                  (post, indptr[i_pre], spk_dot[i_pre]))[0]

        return post, None

    dspk = jax.lax.scan(scan_fn, jnp.zeros((n_post,), dtype=weight.dtype), np.arange(len(spk)))[0]
    dspk = (dspk * weight) if homo_w else dspk
    return y, dweight + dspk


_cpu_event_csr_mv_jvp = jax.custom_jvp(_cpu_event_csr_mv, nondiff_argnums=(1, 2, 4))
_cpu_event_csr_mv_jvp.defjvp(_cpu_event_csr_mv_jvp_rule)
