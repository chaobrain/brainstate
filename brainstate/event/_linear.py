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
import jax.experimental.pallas as pl
import jax.numpy as jnp
import numpy as np
from jax.interpreters import ad

from brainstate._state import ParamState, State
from brainstate.init import param
from brainstate.nn._module import Module
from brainstate.typing import ArrayLike, Size
from ._xla_custom_op import XLACustomOp

__all__ = [
    'Linear',
]


class Linear(Module):
    """
    The FixedProb module implements a fixed probability connection with CSR sparse data structure.

    Parameters
    ----------
    in_size : Size
        Number of pre-synaptic neurons, i.e., input size.
    out_size : Size
        Number of post-synaptic neurons, i.e., output size.
    weight : float or callable or jax.Array or brainunit.Quantity
        Maximum synaptic conductance.
    block_size : int, optional
        Block size for parallel computation.
    float_as_event : bool, optional
        Whether to treat float as event.
    name : str, optional
        Name of the module.
    """

    __module__ = 'brainstate.event'

    def __init__(
        self,
        in_size: Size,
        out_size: Size,
        weight: Union[Callable, ArrayLike],
        float_as_event: bool = True,
        block_size: int = 64,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)

        # network parameters
        self.in_size = in_size
        self.out_size = out_size
        self.float_as_event = float_as_event
        self.block_size = block_size

        # maximum synaptic conductance
        weight = param(weight, (self.in_size[-1], self.out_size[-1]), allow_none=False)
        self.weight = ParamState(weight)

    def update(self, spk: jax.Array) -> Union[jax.Array, u.Quantity]:
        weight = self.weight.value if isinstance(self.weight, State) else self.weight
        if u.math.size(weight) == 1:
            return u.math.ones(self.out_size) * (u.math.sum(spk) * weight)

        return event_linear(spk, weight, block_size=self.block_size, float_as_event=self.float_as_event)


def event_linear(spk, weight, *, block_size, float_as_event) -> jax.Array | u.Quantity:
    """
    The event-driven linear computation.

    Parameters
    ----------
    weight : brainunit.Quantity or jax.Array
        Maximum synaptic conductance.
    spk : jax.Array
        Spike events.
    block_size : int
        Block size for parallel computation.
    float_as_event : bool
        Whether to treat float as event.

    Returns
    -------
    post_data : brainunit.Quantity or jax.Array
        Post synaptic data.
    """
    with jax.ensure_compile_time_eval():
        weight = u.math.asarray(weight)
        unit = u.get_unit(weight)
        weight = u.get_mantissa(weight)
        spk = jnp.asarray(spk)

    def mv(spk_vector):
        assert spk_vector.ndim == 1, f"spk must be 1D. Got: {spk.ndim}"
        return event_linear_p(
            spk, weight,
            outs=[jax.ShapeDtypeStruct([weight.shape[1]], weight.dtype)],
            block_size=block_size,
            float_as_event=float_as_event,
            spk_info=jax.ShapeDtypeStruct(spk.shape, spk.dtype),
            weight_info=jax.ShapeDtypeStruct(weight.shape, weight.dtype),
        )

    assert spk.ndim >= 1, f"spk must be at least 1D. Got: {spk.ndim}"
    assert weight.ndim in [2, 0], f"weight must be 2D or 0D. Got: {weight.ndim}"

    if spk.ndim == 1:
        [post_data] = mv(spk)
    else:
        [post_data] = jax.vmap(mv)(u.math.reshape(spk, (-1, spk.shape[-1])))
        post_data = u.math.reshape(post_data, spk.shape[:-1] + post_data.shape[-1:])
    return u.maybe_decimal(u.Quantity(post_data, unit=unit))


Kernel = Callable


def cpu_kernel_generator(
    float_as_event: bool,
    spk_info: jax.ShapeDtypeStruct,
    **kwargs
) -> Kernel:
    import numba  # pylint: disable=import-outside-toplevel

    if spk_info.dtype == jnp.bool_:

        @numba.njit
        def _kernel(spikes, weights, posts):
            r = np.zeros((weights.shape[1],), dtype=weights.dtype)
            for i in range(spikes.shape[0]):
                if spikes[i]:
                    r = r + weights[i]
            posts[:] = r

    elif float_as_event:
        @numba.njit
        def _kernel(spikes, weights, posts):
            r = np.zeros((weights.shape[1],), dtype=weights.dtype)
            for i in range(spikes.shape[0]):
                if spikes[i] != 0.:
                    r = r + weights[i]
            posts[:] = r

    else:
        @numba.njit
        def _kernel(spikes, weights, posts):
            r = np.zeros((weights.shape[1],), dtype=weights.dtype)
            for i in range(spikes.shape[0]):
                sp = spikes[i]
                if sp != 0.:
                    r = r + weights[i] * sp
            posts[:] = r

    return _kernel


def gpu_kernel_generator(
    block_size: int,
    float_as_event: bool,
    weight_info: jax.ShapeDtypeStruct,
    **kwargs
) -> Kernel:
    def _mv_kernel(sp_ref, w_ref, post_ref):
        # 每个block处理一个[block_size,]的post
        # 每个block处理一个[n_pre]的pre
        # 每个block处理一个[n_pre, block_size]的w

        pid = pl.program_id(0)

        def scan_fn(i, post_):
            if sp_ref.dtype == jnp.bool_:
                post_ = jax.lax.cond(
                    sp_ref[i],
                    lambda: post_ + w_ref[i, ...],
                    lambda: post_
                )
            else:
                if float_as_event:
                    post_ = jax.lax.cond(
                        sp_ref[i] != 0.,
                        lambda: post_ + w_ref[i, ...],
                        lambda: post_
                    )
                else:
                    sp = sp_ref[i]
                    post_ = jax.lax.cond(
                        sp != 0.,
                        lambda: post_ + w_ref[i, ...] * sp,
                        lambda: post_
                    )
            return post_

        post = jax.lax.fori_loop(0, n_pre, scan_fn, jnp.zeros(post_ref.shape, dtype=post_ref.dtype))
        mask = jnp.arange(block_size) + pid * block_size < n_pre
        pl.store(post_ref, pl.dslice(None, None), post, mask=mask)

    n_pre = weight_info.shape[0]
    n_post = weight_info.shape[1]
    kernel = pl.pallas_call(
        _mv_kernel,
        out_shape=[
            jax.ShapeDtypeStruct([weight_info.shape[1]], weight_info.dtype),
        ],
        out_specs=[
            pl.BlockSpec((block_size,), lambda i: i),
        ],
        in_specs=[
            pl.BlockSpec((n_pre,), lambda i: 0),
            pl.BlockSpec((n_pre, block_size), lambda i: (0, i)),
        ],
        grid=(
            pl.cdiv(n_post, block_size),
        ),
        interpret=False,
    )
    return kernel


def jvp_spikes(spk_dot, spikes, weights, **kwargs):
    return [spk_dot @ weights]


def jvp_weights(w_dot, spikes, weights, *, float_as_event, block_size, **kwargs):
    return event_linear_p(
        spikes,
        w_dot,
        outs=[jax.ShapeDtypeStruct([w_dot.shape[1]], w_dot.dtype)],
        block_size=block_size,
        float_as_event=float_as_event,
        spk_info=jax.ShapeDtypeStruct(spikes.shape, spikes.dtype),
        weight_info=jax.ShapeDtypeStruct(w_dot.shape, w_dot.dtype),
    )


def transpose_rule(ct, spikes, weights, *, float_as_event, **kwargs):
    if ad.is_undefined_primal(spikes):
        ct_events = jnp.matmul(weights, ct[0])
        return (ad.Zero(spikes) if type(ct[0]) is ad.Zero else ct_events), weights

    else:
        def map_fn(sp):
            if spikes.dtype == jnp.bool_:
                d_gmax = jnp.where(sp, ct[0], jnp.zeros_like(ct[0]))
            else:
                if float_as_event:
                    d_gmax = jnp.where(sp == 0., jnp.zeros_like(ct[0]), ct[0])
                else:
                    d_gmax = jnp.where(sp == 0., jnp.zeros_like(ct[0]), ct[0] * sp)
                    # d_gmax = jax.lax.cond(sp == 0., lambda: jnp.zeros_like(ct[0]), lambda: ct[0] * sp)
            return d_gmax

        ct_weights = jax.vmap(map_fn)(spikes)
        return spikes, (ad.Zero(weights) if type(ct[0]) is ad.Zero else ct_weights)


event_linear_p = XLACustomOp(
    'event_linear',
    cpu_kernel_generator=cpu_kernel_generator,
    gpu_kernel_generator=gpu_kernel_generator,
)
event_linear_p.defjvp(jvp_spikes, jvp_weights)
event_linear_p.def_transpose_rule(transpose_rule)
