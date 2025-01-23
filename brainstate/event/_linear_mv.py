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
from ._xla_custom_op import XLACustomKernel, PallasKernelGenerator
from ._xla_custom_op_numba import numba_environ, NumbaKernelGenerator

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
        return event_liner_p_call(
            spk,
            weight,
            block_size=block_size,
            float_as_event=float_as_event,
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
    # max difference: 5.9604645e-08
    # n_pre: 1000, n_post: 1000, spike probability: 0.01, Linear: 0.0006995201110839844 s
    # n_pre: 1000, n_post: 1000, spike probability: 0.01, Matmul: 0.008831977844238281 s
    # Acceleration ratio: 11.625766871165645
    #
    # max difference: 5.9604645e-07
    # n_pre: 1000, n_post: 1000, spike probability: 0.1, Linear: 0.0016047954559326172 s
    # n_pre: 1000, n_post: 1000, spike probability: 0.1, Matmul: 0.010058879852294922 s
    # Acceleration ratio: 5.268013668102808
    #
    #
    #
    # max difference: 1.1920929e-07
    # n_pre: 1000, n_post: 10000, spike probability: 0.01, Linear: 0.002207040786743164 s
    # n_pre: 1000, n_post: 10000, spike probability: 0.01, Matmul: 0.1253821849822998 s
    # Acceleration ratio: 55.8100896618775
    #
    # max difference: 5.9604645e-07
    # n_pre: 1000, n_post: 10000, spike probability: 0.1, Linear: 0.013149738311767578 s
    # n_pre: 1000, n_post: 10000, spike probability: 0.1, Matmul: 0.12587380409240723 s
    # Acceleration ratio: 8.57234289444102
    #
    #
    #
    # max difference: 2.0861626e-07
    # n_pre: 10000, n_post: 10000, spike probability: 0.01, Linear: 0.015156984329223633 s
    # n_pre: 10000, n_post: 10000, spike probability: 0.01, Matmul: 0.4421505928039551 s
    # Acceleration ratio: 28.17140924606358
    #
    # max difference: 1.9073486e-06
    # n_pre: 10000, n_post: 10000, spike probability: 0.1, Linear: 0.19855880737304688 s
    # n_pre: 10000, n_post: 10000, spike probability: 0.1, Matmul: 1.1657319068908691 s
    # Acceleration ratio: 4.870965495379532
    #
    #
    #
    # max difference: 1.4901161e-07
    # n_pre: 10000, n_post: 1000, spike probability: 0.01, Linear: 0.0022020339965820312 s
    # n_pre: 10000, n_post: 1000, spike probability: 0.01, Matmul: 0.12016820907592773 s
    # Acceleration ratio: 53.57145950627977
    #
    # max difference: 2.1457672e-06
    # n_pre: 10000, n_post: 1000, spike probability: 0.1, Linear: 0.01901721954345703 s
    # n_pre: 10000, n_post: 1000, spike probability: 0.1, Matmul: 0.12867045402526855 s
    # Acceleration ratio: 5.765997191715561
    #
    #
    #
    # max difference: 2.5331974e-07
    # n_pre: 20000, n_post: 10000, spike probability: 0.01, Linear: 0.030600309371948242 s
    # n_pre: 20000, n_post: 10000, spike probability: 0.01, Matmul: 2.2922556400299072 s
    # Acceleration ratio: 73.90955768346747
    #
    # max difference: 2.8014183e-06
    # n_pre: 20000, n_post: 10000, spike probability: 0.1, Linear: 0.43077588081359863 s
    # n_pre: 20000, n_post: 10000, spike probability: 0.1, Matmul: 2.2729294300079346 s
    # Acceleration ratio: 4.276361865281533
    #
    #
    #
    # max difference: 2.9802322e-07
    # n_pre: 20000, n_post: 20000, spike probability: 0.01, Linear: 0.05977320671081543 s
    # n_pre: 20000, n_post: 20000, spike probability: 0.01, Matmul: 4.704617500305176 s
    # Acceleration ratio: 77.70779834627673
    #
    # max difference: 2.503395e-06
    # n_pre: 20000, n_post: 20000, spike probability: 0.1, Linear: 0.8139562606811523 s
    # n_pre: 20000, n_post: 20000, spike probability: 0.1, Matmul: 4.603509187698364 s
    # Acceleration ratio: 4.6557205959027295
    #
    #

    import numba  # pylint: disable=import-outside-toplevel

    if spk_info.dtype == jnp.bool_:

        @numba.njit(**numba_environ.numba_setting)
        def _kernel(spikes, weights, posts):
            r = np.zeros((weights.shape[1],), dtype=weights.dtype)
            for i in range(spikes.shape[0]):
                if spikes[i]:
                    r = r + weights[i]
            posts[:] = r

    elif float_as_event:
        @numba.njit(**numba_environ.numba_setting)
        def _kernel(spikes, weights, posts):
            r = np.zeros((weights.shape[1],), dtype=weights.dtype)
            for i in range(spikes.shape[0]):
                if spikes[i] != 0.:
                    r = r + weights[i]
            posts[:] = r

    else:
        @numba.njit(**numba_environ.numba_setting)
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
    def version1():
        # max difference: 0.99948376
        # n_pre: 1000, n_post: 1000, spike probability: 0.01, Linear: 0.013459444046020508 s
        # n_pre: 1000, n_post: 1000, spike probability: 0.01, Matmul: 0.012884378433227539 s
        # Acceleration ratio: -0.042725807308734653
        #
        # max difference: 0.9998415
        # n_pre: 1000, n_post: 1000, spike probability: 0.1, Linear: 0.010735034942626953 s
        # n_pre: 1000, n_post: 1000, spike probability: 0.1, Matmul: 0.011649131774902344 s
        # Acceleration ratio: 0.08515080175898371
        #
        #
        #
        # max difference: 0.99967647
        # n_pre: 1000, n_post: 10000, spike probability: 0.01, Linear: 0.01203775405883789 s
        # n_pre: 1000, n_post: 10000, spike probability: 0.01, Matmul: 0.10982584953308105 s
        # Acceleration ratio: 8.12345018815607
        #
        # max difference: 0.9999729
        # n_pre: 1000, n_post: 10000, spike probability: 0.1, Linear: 0.013982772827148438 s
        # n_pre: 1000, n_post: 10000, spike probability: 0.1, Matmul: 0.11064291000366211 s
        # Acceleration ratio: 6.912801800572909
        #
        #
        #
        # max difference: 0.999918
        # n_pre: 10000, n_post: 10000, spike probability: 0.01, Linear: 0.13564801216125488 s
        # n_pre: 10000, n_post: 10000, spike probability: 0.01, Matmul: 0.3635435104370117 s
        # Acceleration ratio: 1.6800504087361081
        #
        # max difference: 0.99988335
        # n_pre: 10000, n_post: 10000, spike probability: 0.1, Linear: 0.3616366386413574 s
        # n_pre: 10000, n_post: 10000, spike probability: 0.1, Matmul: 0.13324809074401855 s
        # Acceleration ratio: -0.6315415073964243
        #
        #
        #
        # max difference: 1.0430813e-07
        # n_pre: 10000, n_post: 1000, spike probability: 0.01, Linear: 0.12608551979064941 s
        # n_pre: 10000, n_post: 1000, spike probability: 0.01, Matmul: 0.1295468807220459 s
        # Acceleration ratio: 0.027452485718769903
        #
        # max difference: 4.7683716e-07
        # n_pre: 10000, n_post: 1000, spike probability: 0.1, Linear: 0.08305215835571289 s
        # n_pre: 10000, n_post: 1000, spike probability: 0.1, Matmul: 0.12557435035705566 s
        # Acceleration ratio: 0.5119938222342153
        #
        #
        #
        # max difference: 0.9996885
        # n_pre: 20000, n_post: 10000, spike probability: 0.01, Linear: 0.13745379447937012 s
        # n_pre: 20000, n_post: 10000, spike probability: 0.01, Matmul: 0.30812668800354004 s
        # Acceleration ratio: 1.2416746599875461
        #
        # max difference: 0.99990356
        # n_pre: 20000, n_post: 10000, spike probability: 0.1, Linear: 0.3163454532623291 s
        # n_pre: 20000, n_post: 10000, spike probability: 0.1, Matmul: 0.2500569820404053 s
        # Acceleration ratio: -0.2095445676184705
        #
        #
        #
        # max difference: 0.999979
        # n_pre: 20000, n_post: 20000, spike probability: 0.01, Linear: 0.3194081783294678 s
        # n_pre: 20000, n_post: 20000, spike probability: 0.01, Matmul: 0.6508703231811523 s
        # Acceleration ratio: 1.037738440465927
        #
        # max difference: 0.9999609
        # n_pre: 20000, n_post: 20000, spike probability: 0.1, Linear: 0.36977124214172363 s
        # n_pre: 20000, n_post: 20000, spike probability: 0.1, Matmul: 0.4229443073272705 s
        # Acceleration ratio: 0.14379989335451637
        #
        #
        #
        #

        # 每个block处理一个[block_size,]的post
        # 每个block处理一个[block_size]的pre
        # 每个block处理一个[block_size, block_size]的w
        def _mv_kernel(
            sp_ref,  # [block_size]
            w_ref,  # [block_size, block_size]
            post_ref,  # [block_size]
        ):

            r_pid = pl.program_id(0)
            c_start = pl.program_id(1) * block_size
            row_length = jnp.minimum(n_pre - r_pid * block_size, block_size)
            mask = jnp.arange(block_size) + c_start < weight_info.shape[1]

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

            post = jax.lax.fori_loop(0, row_length, scan_fn, jnp.zeros(post_ref.shape, dtype=post_ref.dtype))
            pl.atomic_add(post_ref, pl.dslice(None, None), post, mask=mask)

        n_pre = weight_info.shape[0]
        n_post = weight_info.shape[1]
        kernel = pl.pallas_call(
            _mv_kernel,
            out_shape=[
                jax.ShapeDtypeStruct([weight_info.shape[1]], weight_info.dtype),
            ],
            out_specs=[
                pl.BlockSpec((block_size,), lambda i, j: j),
            ],
            in_specs=[
                pl.BlockSpec((block_size,), lambda i, j: i),
                pl.BlockSpec((block_size, block_size), lambda i, j: (i, j)),
            ],
            grid=(
                pl.cdiv(n_pre, block_size),
                pl.cdiv(n_post, block_size),
            ),
            interpret=False,
        )
        return kernel

    def version2():
        def _mv_kernel(
            sp_ref,  # [block_size]
            w_ref,  # [block_size, block_size]
            post_ref,  # [block_size]
        ):

            r_pid = pl.program_id(0)
            c_start = pl.program_id(1) * block_size
            row_length = jnp.minimum(n_pre - r_pid * block_size, block_size)
            mask = jnp.arange(block_size) + c_start < weight_info.shape[1]

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

            post = jax.lax.fori_loop(0, row_length, scan_fn, jnp.zeros(post_ref.shape, dtype=post_ref.dtype))
            pl.atomic_add(post_ref, pl.dslice(None, None), post, mask=mask)

        n_pre = weight_info.shape[0]
        n_post = weight_info.shape[1]
        kernel = pl.pallas_call(
            _mv_kernel,
            out_shape=[
                jax.ShapeDtypeStruct([weight_info.shape[1]], weight_info.dtype),
            ],
            out_specs=[
                pl.BlockSpec((block_size,), lambda i, j: j),
            ],
            in_specs=[
                pl.BlockSpec((block_size,), lambda i, j: i),
                pl.BlockSpec((block_size, block_size), lambda i, j: (i, j)),
            ],
            grid=(
                pl.cdiv(n_pre, block_size),
                pl.cdiv(n_post, block_size),
            ),
            interpret=False,
        )
        return kernel

    return version1()


def jvp_spikes(spk_dot, spikes, weights, **kwargs):
    return [spk_dot @ weights]


def jvp_weights(w_dot, spikes, weights, *, float_as_event, block_size, **kwargs):
    return event_liner_p_call(
        spikes,
        w_dot,
        block_size=block_size,
        float_as_event=float_as_event,
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


event_linear_p = XLACustomKernel(
    'event_linear',
    cpu_kernel=NumbaKernelGenerator(cpu_kernel_generator),
    gpu_kernel=PallasKernelGenerator(gpu_kernel_generator),
)
event_linear_p.defjvp(jvp_spikes, jvp_weights)
event_linear_p.def_transpose_rule(transpose_rule)


def event_liner_p_call(
    spikes,
    weights,
    *,
    block_size,
    float_as_event,
):
    return event_linear_p(
        spikes,
        weights,
        outs=[jax.ShapeDtypeStruct([weights.shape[1]], weights.dtype)],
        block_size=block_size,
        float_as_event=float_as_event,
        spk_info=jax.ShapeDtypeStruct(spikes.shape, spikes.dtype),
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
    )
