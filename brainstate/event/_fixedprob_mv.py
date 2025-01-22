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

from brainstate import environ
from brainstate._state import ParamState
from brainstate.augment import vmap
from brainstate.init import param
from brainstate.nn._module import Module
from brainstate.random import RandomState
from brainstate.typing import ArrayLike, Size
from ._misc import FloatScalar, op_environ
from ._xla_custom_op import XLACustomOp, NumbaOpGenerator, PallasOpGenerator

__all__ = [
    'FixedProb',
]


class FixedProb(Module):
    """
    The FixedProb module implements a fixed probability connection with CSR sparse data structure.

    Parameters
    ----------
    in_size : Size
        Number of pre-synaptic neurons, i.e., input size.
    out_size : Size
        Number of post-synaptic neurons, i.e., output size.
    prob : float
        Probability of connection, i.e., connection probability.
    weight : float or callable or jax.Array or brainunit.Quantity
        Maximum synaptic conductance, i.e., synaptic weight.
    allow_multi_conn : bool, optional
        Whether multiple connections are allowed from a single pre-synaptic neuron.
        Default is True, meaning that a value of ``a`` can be selected multiple times.
    seed: int, optional
        Random seed. Default is None. If None, the default random seed will be used.
    float_as_event : bool, optional
        Whether to treat float as event. Default is True.
    block_size : int, optional
        Block size for parallel computation. Default is 64. This is only used for GPU.
    name : str, optional
        Name of the module.
    """

    __module__ = 'brainstate.event'

    def __init__(
        self,
        in_size: Size,
        out_size: Size,
        prob: FloatScalar,
        weight: Union[Callable, ArrayLike],
        allow_multi_conn: bool = True,
        seed: Optional[int] = None,
        float_as_event: bool = True,
        block_size: Optional[int] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)

        # network parameters
        self.in_size = in_size
        self.out_size = out_size
        self.n_conn = int(self.out_size[-1] * prob)
        self.float_as_event = float_as_event
        self.block_size = block_size

        if self.n_conn > 1:
            # indices of post connected neurons
            with jax.ensure_compile_time_eval():
                if allow_multi_conn:
                    rng = np.random.RandomState(seed)
                    self.indices = rng.randint(0, self.out_size[-1], size=(self.in_size[-1], self.n_conn))
                else:
                    rng = RandomState(seed)

                    @vmap(rngs=rng)
                    def rand_indices(key):
                        rng.set_key(key)
                        return rng.choice(self.out_size[-1], size=(self.n_conn,), replace=False)

                    self.indices = rand_indices(rng.split_key(self.in_size[-1]))
                self.indices = u.math.asarray(self.indices)

        # maximum synaptic conductance
        weight = param(weight, (self.in_size[-1], self.n_conn), allow_none=False)
        self.weight = ParamState(weight)

    def update(self, spk: jax.Array) -> Union[jax.Array, u.Quantity]:
        if self.n_conn > 1:
            r = event_fixed_prob(
                spk,
                self.weight.value,
                self.indices,
                n_post=self.out_size[-1],
                block_size=self.block_size,
                float_as_event=self.float_as_event
            )
        else:
            weight = self.weight.value
            unit = u.get_unit(weight)
            r = jnp.zeros(spk.shape[:-1] + (self.out_size[-1],), dtype=weight.dtype)
            r = u.maybe_decimal(u.Quantity(r, unit=unit))
        return u.math.asarray(r, dtype=environ.dftype())


def event_fixed_prob(
    spk, weight, indices,
    *,
    n_post, block_size, float_as_event
):
    """
    The FixedProb module implements a fixed probability connection with CSR sparse data structure.

    Parameters
    ----------
    weight : brainunit.Quantity or jax.Array
        Maximum synaptic conductance.
    spk : jax.Array
        Spike events.

    Returns
    -------
    post_data : brainunit.Quantity or jax.Array
        Post synaptic data.
    """
    with jax.ensure_compile_time_eval():
        weight = u.math.asarray(weight)
        unit = u.get_unit(weight)
        weight = u.get_mantissa(weight)
        indices = jnp.asarray(indices)
        spk = jnp.asarray(spk)

    def mv(spk_vector):
        assert spk_vector.ndim == 1, f"spk must be 1D. Got: {spk.ndim}"
        return event_ellmv_p_call(
            spk,
            weight,
            indices,
            n_post=n_post,
            block_size=block_size,
            float_as_event=float_as_event
        )

    assert spk.ndim >= 1, f"spk must be at least 1D. Got: {spk.ndim}"
    assert weight.ndim in [2, 0], f"weight must be 2D or 0D. Got: {weight.ndim}"
    assert indices.ndim == 2, f"indices must be 2D. Got: {indices.ndim}"

    if spk.ndim == 1:
        [post_data] = mv(spk)
    else:
        [post_data] = jax.vmap(mv)(u.math.reshape(spk, (-1, spk.shape[-1])))
        post_data = u.math.reshape(post_data, spk.shape[:-1] + post_data.shape[-1:])
    return u.maybe_decimal(u.Quantity(post_data, unit=unit))


Kernel = Callable


def cpu_kernel_generator(
    float_as_event: bool,
    weight_info: jax.ShapeDtypeStruct,
    spike_info: jax.ShapeDtypeStruct,
    **kwargs
):
    import numba  # pylint: disable=import-outside-toplevel

    def version1():
        # Intel i9-12900H
        #
        # n_pre: 1000, n_post: 1000, conn_prob: 0.01, spk_prob: 0.01, Linear: 0.004149198532104492 s
        # n_pre: 1000, n_post: 1000, conn_prob: 0.01, spk_prob: 0.01, Matmul: 0.08957552909851074 s
        # Acceleration ratio: 20.58863414353847
        #
        # n_pre: 1000, n_post: 10000, conn_prob: 0.01, spk_prob: 0.01, Linear: 0.006804466247558594 s
        # n_pre: 1000, n_post: 10000, conn_prob: 0.01, spk_prob: 0.01, Matmul: 1.2348318099975586 s
        # Acceleration ratio: 180.47372109320253
        #
        # n_pre: 10000, n_post: 10000, conn_prob: 0.01, spk_prob: 0.01, Linear: 0.03094005584716797 s
        # n_pre: 10000, n_post: 10000, conn_prob: 0.01, spk_prob: 0.01, Matmul: 11.536803245544434 s
        # Acceleration ratio: 371.8759670807262
        #
        # n_pre: 10000, n_post: 1000, conn_prob: 0.01, spk_prob: 0.01, Linear: 0.010653495788574219 s
        # n_pre: 10000, n_post: 1000, conn_prob: 0.01, spk_prob: 0.01, Matmul: 1.2019984722137451 s
        # Acceleration ratio: 111.82667173932504
        #
        # n_pre: 10000, n_post: 20000, conn_prob: 0.01, spk_prob: 0.01, Linear: 0.05920886993408203 s
        # n_pre: 10000, n_post: 20000, conn_prob: 0.01, spk_prob: 0.01, Matmul: 23.162949562072754 s
        # Acceleration ratio: 390.20742530401867
        #
        # n_pre: 20000, n_post: 10000, conn_prob: 0.01, spk_prob: 0.01, Linear: 0.06537938117980957 s
        # n_pre: 20000, n_post: 10000, conn_prob: 0.01, spk_prob: 0.01, Matmul: 21.971742630004883 s
        # Acceleration ratio: 335.0653195780046
        #
        # n_pre: 20000, n_post: 20000, conn_prob: 0.01, spk_prob: 0.01, Linear: 0.11733055114746094 s
        # n_pre: 20000, n_post: 20000, conn_prob: 0.01, spk_prob: 0.01, Matmul: 45.15763020515442 s
        # Acceleration ratio: 383.87529261155817
        #
        # n_pre: 20000, n_post: 30000, conn_prob: 0.01, spk_prob: 0.01, Linear: 0.16046690940856934 s
        # n_pre: 20000, n_post: 30000, conn_prob: 0.01, spk_prob: 0.01, Matmul: 68.04417014122009 s
        # Acceleration ratio: 423.0386406892832
        #
        # n_pre: 30000, n_post: 20000, conn_prob: 0.01, spk_prob: 0.01, Linear: 0.17695116996765137 s
        # n_pre: 30000, n_post: 20000, conn_prob: 0.01, spk_prob: 0.01, Matmul: 73.59888315200806 s
        # Acceleration ratio: 414.92764357230726
        #
        #
        if weight_info.size == 1:
            if spike_info.dtype == jnp.bool_:
                @numba.njit(**op_environ.numba_setting)
                def ell_mv(spikes, weights, indices, posts):
                    posts[:] = 0.
                    w = weights[()]
                    for i in range(spikes.shape[0]):
                        if spikes[i]:
                            for j in range(indices.shape[1]):
                                posts[indices[i, j]] += w

            elif float_as_event:
                @numba.njit(**op_environ.numba_setting)
                def ell_mv(spikes, weights, indices, posts):
                    posts[:] = 0.
                    w = weights[()]
                    for i in range(spikes.shape[0]):
                        if spikes[i] != 0.:
                            for j in range(indices.shape[1]):
                                posts[indices[i, j]] += w

            else:
                @numba.njit(**op_environ.numba_setting)
                def ell_mv(spikes, weights, indices, posts):
                    posts[:] = 0.
                    w = weights[()]
                    for i in range(spikes.shape[0]):
                        sp = spikes[i]
                        if sp != 0.:
                            wsp = w * sp
                            for j in range(indices.shape[1]):
                                posts[indices[i, j]] += wsp

        else:
            if spike_info.dtype == jnp.bool_:
                @numba.njit(**op_environ.numba_setting)
                def ell_mv(spikes, weights, indices, posts):
                    posts[:] = 0.
                    for i in range(spikes.shape[0]):
                        if spikes[i]:
                            for j in range(indices.shape[1]):
                                posts[indices[i, j]] += weights[i, j]

            elif float_as_event:
                @numba.njit(**op_environ.numba_setting)
                def ell_mv(spikes, weights, indices, posts):
                    posts[:] = 0.
                    for i in range(spikes.shape[0]):
                        if spikes[i] != 0.:
                            for j in range(indices.shape[1]):
                                posts[indices[i, j]] += weights[i, j]

            else:
                @numba.njit(**op_environ.numba_setting)
                def ell_mv(spikes, weights, indices, posts):
                    posts[:] = 0.
                    for i in range(spikes.shape[0]):
                        sp = spikes[i]
                        if sp != 0.:
                            for j in range(indices.shape[1]):
                                posts[indices[i, j]] += weights[i, j] * sp

        return ell_mv

    return version1()


def gpu_kernel_generator(
    n_pre: int,
    n_conn: int,
    n_post: int,
    block_size: int,
    float_as_event: bool,
    weight_info: jax.ShapeDtypeStruct,
    **kwargs
):
    def version1():
        # [NVIDIA GeForce RTX 3080 Ti Laptop GPU]
        #
        # n_pre: 1000, n_post: 1000, conn_prob: 0.01, spk_prob: 0.01, Linear: 0.09133028984069824 s
        # n_pre: 1000, n_post: 1000, conn_prob: 0.01, spk_prob: 0.01, Matmul: 0.09012126922607422 s
        # Acceleration ratio: -0.01323789311008261
        #
        # n_pre: 1000, n_post: 10000, conn_prob: 0.01, spk_prob: 0.01, Linear: 0.12473106384277344 s
        # n_pre: 1000, n_post: 10000, conn_prob: 0.01, spk_prob: 0.01, Matmul: 1.2428414821624756 s
        # Acceleration ratio: 8.96416966128909
        #
        # n_pre: 10000, n_post: 10000, conn_prob: 0.01, spk_prob: 0.01, Linear: 0.13015508651733398 s
        # n_pre: 10000, n_post: 10000, conn_prob: 0.01, spk_prob: 0.01, Matmul: 1.5407586097717285 s
        # Acceleration ratio: 10.837867047681852
        #
        # n_pre: 10000, n_post: 1000, conn_prob: 0.01, spk_prob: 0.01, Linear: 0.14979290962219238 s
        # n_pre: 10000, n_post: 1000, conn_prob: 0.01, spk_prob: 0.01, Matmul: 1.2113032341003418 s
        # Acceleration ratio: 7.086519162725995
        #
        # n_pre: 10000, n_post: 20000, conn_prob: 0.01, spk_prob: 0.01, Linear: 1.2156291007995605 s
        # n_pre: 10000, n_post: 20000, conn_prob: 0.01, spk_prob: 0.01, Matmul: 2.826601982116699 s
        # Acceleration ratio: 1.3252174370106369
        #
        # n_pre: 20000, n_post: 10000, conn_prob: 0.01, spk_prob: 0.01, Linear: 0.6445927619934082 s
        # n_pre: 20000, n_post: 10000, conn_prob: 0.01, spk_prob: 0.01, Matmul: 3.1173269748687744 s
        # Acceleration ratio: 3.8361184901121383
        #
        # n_pre: 20000, n_post: 20000, conn_prob: 0.01, spk_prob: 0.01, Linear: 0.31626296043395996 s
        # n_pre: 20000, n_post: 20000, conn_prob: 0.01, spk_prob: 0.01, Matmul: 5.701655149459839 s
        # Acceleration ratio: 17.028210264130575
        #
        # n_pre: 20000, n_post: 30000, conn_prob: 0.01, spk_prob: 0.01, Linear: 0.12007498741149902 s
        # n_pre: 20000, n_post: 30000, conn_prob: 0.01, spk_prob: 0.01, Matmul: 6.82172417640686 s
        # Acceleration ratio: 55.81219980501597
        #
        # n_pre: 30000, n_post: 20000, conn_prob: 0.01, spk_prob: 0.01, Linear: 0.16502714157104492 s
        # n_pre: 30000, n_post: 20000, conn_prob: 0.01, spk_prob: 0.01, Matmul: 6.689691066741943 s
        # Acceleration ratio: 39.53691412852837
        #

        # 对于具有形状 [n_event] 的 spikes 向量，以及形状 [n_event, n_conn] 的 indices 和 weights 矩阵，
        # 这个算子的计算逻辑为：
        #
        # - 每个block处理 [block_size] 个事件，每个事件对应一个 pre-synaptic neuron
        # - 每个block处理 [block_size, block_size] 个 indices 和 weights

        if weight_info.size == 1:
            def _ell_mv_kernel_homo(
                sp_ref,  # [block_size]
                ind_ref,  # [block_size, block_size]
                _,
                y_ref,  # [n_post]
            ):
                r_pid = pl.program_id(0)
                c_start = pl.program_id(1) * block_size
                row_length = jnp.minimum(n_pre - r_pid * block_size, block_size)
                mask = jnp.arange(block_size) + c_start < n_conn

                def body_fn(j, _):
                    if sp_ref.dtype == jnp.bool_:
                        def true_fn():
                            ind = pl.load(ind_ref, (j, pl.dslice(None)), mask=mask)
                            pl.atomic_add(y_ref, ind, jnp.ones(block_size, dtype=weight_info.dtype), mask=mask)

                        jax.lax.cond(sp_ref[j], true_fn, lambda: None)


                    else:
                        def true_fn(sp):
                            ind = pl.load(ind_ref, (j, pl.dslice(None)), mask=mask)
                            if float_as_event:
                                pl.atomic_add(y_ref, ind, jnp.ones(block_size, dtype=weight_info.dtype), mask=mask)
                            else:
                                pl.atomic_add(y_ref, ind, jnp.ones(block_size, dtype=weight_info.dtype) * sp, mask=mask)

                        sp_ = sp_ref[j]
                        jax.lax.cond(sp_ != 0., true_fn, lambda _: None, sp_)

                jax.lax.fori_loop(0, row_length, body_fn, None)

            # homogenous weights
            kernel = pl.pallas_call(
                _ell_mv_kernel_homo,
                out_shape=[
                    jax.ShapeDtypeStruct((n_post,), weight_info.dtype),
                ],
                in_specs=[
                    pl.BlockSpec((block_size,), lambda i, j: i),
                    pl.BlockSpec((block_size, block_size), lambda i, j: (i, j)),
                    pl.BlockSpec((n_post,), lambda i, j: 0)
                ],
                grid=(
                    pl.cdiv(n_pre, block_size),
                    pl.cdiv(n_conn, block_size),
                ),
                input_output_aliases={2: 0},
                interpret=False
            )
            return (
                lambda spikes, weight, indices:
                [kernel(spikes, indices, jnp.zeros(n_post, dtype=weight.dtype))[0] * weight]
            )

        else:
            def _ell_mv_kernel_heter(
                sp_ref,  # [block_size]
                ind_ref,  # [block_size, block_size]
                w_ref,  # [block_size, block_size]
                _,
                y_ref,  # [n_post]
            ):
                r_pid = pl.program_id(0)
                c_start = pl.program_id(1) * block_size
                row_length = jnp.minimum(n_pre - r_pid * block_size, block_size)
                mask = jnp.arange(block_size) + c_start < n_conn

                def body_fn(j, _):
                    if sp_ref.dtype == jnp.bool_:
                        def true_fn():
                            ind = pl.load(ind_ref, (j, pl.dslice(None)), mask=mask)
                            w = pl.load(w_ref, (j, pl.dslice(None)), mask=mask)
                            pl.atomic_add(y_ref, ind, w, mask=mask)

                        jax.lax.cond(sp_ref[j], true_fn, lambda: None)
                    else:
                        def true_fn(spk):
                            ind = pl.load(ind_ref, (j, pl.dslice(None)), mask=mask)
                            w = pl.load(w_ref, (j, pl.dslice(None)), mask=mask)
                            if not float_as_event:
                                w = w * spk
                            pl.atomic_add(y_ref, ind, w, mask=mask)

                        sp_ = sp_ref[j]
                        jax.lax.cond(sp_ != 0., true_fn, lambda _: None, sp_)

                jax.lax.fori_loop(0, row_length, body_fn, None)

            # heterogeneous weights
            kernel = pl.pallas_call(
                _ell_mv_kernel_heter,
                out_shape=[
                    jax.ShapeDtypeStruct((n_post,), weight_info.dtype),
                ],
                in_specs=[
                    pl.BlockSpec((block_size,), lambda i, j: i),  # sp_ref
                    pl.BlockSpec((block_size, block_size), lambda i, j: (i, j)),  # ind_ref
                    pl.BlockSpec((block_size, block_size), lambda i, j: (i, j)),  # w_ref,
                    pl.BlockSpec((n_post,), lambda i, j: 0)
                ],
                grid=(
                    pl.cdiv(n_pre, block_size),
                    pl.cdiv(n_conn, block_size),
                ),
                input_output_aliases={3: 0},
                interpret=False
            )
            return (
                lambda spikes, weight, indices:
                kernel(spikes, indices, weight, jnp.zeros(n_post, dtype=weight_info.dtype))
            )

    def version2():
        # [NVIDIA GeForce RTX 3080 Ti Laptop GPU]
        #
        # n_pre: 1000, n_post: 1000, conn_prob: 0.01, spk_prob: 0.01, Linear: 0.09133028984069824 s
        # n_pre: 1000, n_post: 1000, conn_prob: 0.01, spk_prob: 0.01, Matmul: 0.09012126922607422 s
        # Acceleration ratio: -0.01323789311008261
        #
        # n_pre: 1000, n_post: 10000, conn_prob: 0.01, spk_prob: 0.01, Linear: 0.12473106384277344 s
        # n_pre: 1000, n_post: 10000, conn_prob: 0.01, spk_prob: 0.01, Matmul: 1.2428414821624756 s
        # Acceleration ratio: 8.96416966128909
        #
        # n_pre: 10000, n_post: 10000, conn_prob: 0.01, spk_prob: 0.01, Linear: 0.13015508651733398 s
        # n_pre: 10000, n_post: 10000, conn_prob: 0.01, spk_prob: 0.01, Matmul: 1.5407586097717285 s
        # Acceleration ratio: 10.837867047681852
        #
        # n_pre: 10000, n_post: 1000, conn_prob: 0.01, spk_prob: 0.01, Linear: 0.14979290962219238 s
        # n_pre: 10000, n_post: 1000, conn_prob: 0.01, spk_prob: 0.01, Matmul: 1.2113032341003418 s
        # Acceleration ratio: 7.086519162725995
        #
        # n_pre: 10000, n_post: 20000, conn_prob: 0.01, spk_prob: 0.01, Linear: 1.2156291007995605 s
        # n_pre: 10000, n_post: 20000, conn_prob: 0.01, spk_prob: 0.01, Matmul: 2.826601982116699 s
        # Acceleration ratio: 1.3252174370106369
        #
        # n_pre: 20000, n_post: 10000, conn_prob: 0.01, spk_prob: 0.01, Linear: 0.6445927619934082 s
        # n_pre: 20000, n_post: 10000, conn_prob: 0.01, spk_prob: 0.01, Matmul: 3.1173269748687744 s
        # Acceleration ratio: 3.8361184901121383
        #
        # n_pre: 20000, n_post: 20000, conn_prob: 0.01, spk_prob: 0.01, Linear: 0.31626296043395996 s
        # n_pre: 20000, n_post: 20000, conn_prob: 0.01, spk_prob: 0.01, Matmul: 5.701655149459839 s
        # Acceleration ratio: 17.028210264130575
        #
        # n_pre: 20000, n_post: 30000, conn_prob: 0.01, spk_prob: 0.01, Linear: 0.12007498741149902 s
        # n_pre: 20000, n_post: 30000, conn_prob: 0.01, spk_prob: 0.01, Matmul: 6.82172417640686 s
        # Acceleration ratio: 55.81219980501597
        #
        # n_pre: 30000, n_post: 20000, conn_prob: 0.01, spk_prob: 0.01, Linear: 0.16502714157104492 s
        # n_pre: 30000, n_post: 20000, conn_prob: 0.01, spk_prob: 0.01, Matmul: 6.689691066741943 s
        # Acceleration ratio: 39.53691412852837
        #

        # 对于具有形状 [n_event] 的 spikes 向量，以及形状 [n_event, n_conn] 的 indices 和 weights 矩阵，
        # 这个算子的计算逻辑为：
        #
        # - 每个block处理 [block_size] 个事件，每个事件对应一个 pre-synaptic neuron
        # - 每个block处理 [block_size, block_size] 个 indices 和 weights

        if weight_info.size == 1:
            def _ell_mv_kernel_homo(
                sp_ref,  # [n_pre]
                ind_ref,  # [block_size, block_size]
                _,
                y_ref,  # [n_post]
            ):
                r_start = pl.program_id(0) * block_size
                row_mask = (jnp.arange(block_size) + r_start) < n_pre

                c_start = pl.program_id(1) * block_size
                col_mask = (jnp.arange(block_size) + c_start) < n_conn

                def body_fn(j, event):
                    if event.dtype == jnp.bool_:
                        def true_fn():
                            ind = pl.load(ind_ref, (j, pl.dslice(None)), mask=col_mask)
                            pl.atomic_add(y_ref, ind, jnp.ones(block_size, dtype=weight_info.dtype), mask=col_mask)

                        jax.lax.cond(event, true_fn, lambda: None)

                    else:
                        def true_fn():
                            ind = pl.load(ind_ref, (j, pl.dslice(None)), mask=col_mask)
                            if float_as_event:
                                pl.atomic_add(y_ref, ind, jnp.ones(block_size, dtype=weight_info.dtype), mask=col_mask)
                            else:
                                pl.atomic_add(y_ref, ind, jnp.ones(block_size, dtype=weight_info.dtype) * event,
                                              mask=col_mask)

                        jax.lax.cond(event != 0., true_fn, lambda _: None, )

                events = pl.load(sp_ref, pl.dslice(c_start, block_size), mask=row_mask)
                if float_as_event and sp_ref.dtype != jnp.bool_:
                    events = events != 0.

                jax.lax.fori_loop(0, block_size, body_fn, events)

            # homogenous weights
            kernel = pl.pallas_call(
                _ell_mv_kernel_homo,
                out_shape=[
                    jax.ShapeDtypeStruct((n_post,), weight_info.dtype),
                ],
                in_specs=[
                    pl.BlockSpec((n_pre,), lambda i, j: 0),
                    pl.BlockSpec((block_size, block_size), lambda i, j: (i, j)),
                    pl.BlockSpec((n_post,), lambda i, j: 0)
                ],
                grid=(
                    pl.cdiv(n_pre, block_size),
                    pl.cdiv(n_conn, block_size),
                ),
                input_output_aliases={2: 0},
                interpret=False
            )
            return (
                lambda spikes, weight, indices:
                [kernel(spikes, indices, jnp.zeros(n_post, dtype=weight.dtype))[0] * weight]
            )

        else:
            def _ell_mv_kernel_heter(
                sp_ref,  # [n_pre]
                ind_ref,  # [block_size, block_size]
                w_ref,  # [block_size, block_size]
                _,
                y_ref,  # [n_post]
            ):
                r_start = pl.program_id(0) * block_size
                row_mask = (jnp.arange(block_size) + r_start) < n_pre
                row_length = jnp.minimum(n_pre - r_start, block_size)

                c_start = pl.program_id(1) * block_size
                col_mask = (jnp.arange(block_size) + c_start) < n_conn

                def body_fn(j, event):
                    if sp_ref.dtype == jnp.bool_:
                        def true_fn():
                            ind = pl.load(ind_ref, (j, pl.dslice(None)), mask=col_mask)
                            w = pl.load(w_ref, (j, pl.dslice(None)), mask=col_mask)
                            pl.atomic_add(y_ref, ind, w, mask=col_mask)

                        jax.lax.cond(event, true_fn, lambda: None)
                    else:
                        def true_fn():
                            ind = pl.load(ind_ref, (j, pl.dslice(None)), mask=col_mask)
                            w = pl.load(w_ref, (j, pl.dslice(None)), mask=col_mask)
                            if not float_as_event:
                                w = w * event
                            pl.atomic_add(y_ref, ind, w, mask=col_mask)

                        jax.lax.cond(event != 0., true_fn, lambda _: None)

                events = pl.load(sp_ref, pl.dslice(c_start, block_size), mask=row_mask)
                if float_as_event and sp_ref.dtype != jnp.bool_:
                    events = events != 0.
                jax.lax.fori_loop(0, row_length, body_fn, events)

            # heterogeneous weights
            kernel = pl.pallas_call(
                _ell_mv_kernel_heter,
                out_shape=[
                    jax.ShapeDtypeStruct((n_post,), weight_info.dtype),
                ],
                in_specs=[
                    pl.BlockSpec((n_pre,), lambda i, j: 0),  # sp_ref
                    pl.BlockSpec((block_size, block_size), lambda i, j: (i, j)),  # ind_ref
                    pl.BlockSpec((block_size, block_size), lambda i, j: (i, j)),  # w_ref,
                    pl.BlockSpec((n_post,), lambda i, j: 0)
                ],
                grid=(
                    pl.cdiv(n_pre, block_size),
                    pl.cdiv(n_conn, block_size),
                ),
                input_output_aliases={3: 0},
                interpret=False
            )
            return (
                lambda spikes, weight, indices:
                kernel(spikes, indices, weight, jnp.zeros(n_post, dtype=weight_info.dtype))
            )

    return version1()


def jvp_spikes(
    spk_dot, spikes, weights, indices,
    *,
    n_post, block_size, **kwargs
):
    return ellmv_p_call(
        spk_dot,
        weights,
        indices,
        n_post=n_post,
        block_size=block_size,
    )


def jvp_weights(
    w_dot, spikes, weights, indices,
    *,
    float_as_event, block_size, n_post, **kwargs
):
    return event_ellmv_p_call(
        spikes,
        w_dot,
        indices,
        n_post=n_post,
        block_size=block_size,
        float_as_event=float_as_event
    )


def transpose_rule(
    ct, spikes, weights, indices,
    *,
    float_as_event, n_post, n_conn, block_size, weight_info, **kwargs
):
    if ad.is_undefined_primal(indices):
        raise ValueError("Cannot transpose with respect to sparse indices.")

    ct = ct[0]

    # ∂L/∂spk = ∂L/∂y * ∂y/∂spk
    homo = weight_info.size == 1
    if ad.is_undefined_primal(spikes):
        if homo:
            # homogeneous weight
            ct_spk = jax.vmap(lambda idx: jnp.sum(ct[idx] * weights))(indices)
        else:
            # heterogeneous weight
            ct_spk = jax.vmap(lambda idx, w: jnp.inner(ct[idx], w))(indices, weights)
        return (ad.Zero(spikes) if type(ct) is ad.Zero else ct_spk), weights, indices

    else:
        # ∂L/∂w = ∂L/∂y * ∂y/∂w
        if homo:
            # scalar
            ct_gmax = event_ellmv_p_call(
                spikes,
                jnp.asarray(1., dtype=weight_info.dtype),
                indices,
                n_post=n_post,
                block_size=block_size,
                float_as_event=float_as_event
            )
            ct_gmax = jnp.inner(ct, ct_gmax[0])
        else:
            def map_fn(one_spk, one_ind):
                if spikes.dtype == jnp.bool_:
                    return jax.lax.cond(
                        one_spk,
                        lambda: ct[one_ind],
                        lambda: jnp.zeros([n_conn], weight_info.dtype)
                    )
                else:
                    if float_as_event:
                        return jax.lax.cond(
                            one_spk == 0.,
                            lambda: jnp.zeros([n_conn], weight_info.dtype),
                            lambda: ct[one_ind]
                        )
                    else:
                        return jax.lax.cond(
                            one_spk == 0.,
                            lambda: jnp.zeros([n_conn], weight_info.dtype),
                            lambda: ct[one_ind] * one_spk
                        )

            ct_gmax = jax.vmap(map_fn)(spikes, indices)
        return spikes, (ad.Zero(weights) if type(ct) is ad.Zero else ct_gmax), indices


event_ellmv_p = XLACustomOp(
    'event_ell_mv',
    cpu_kernel=NumbaOpGenerator(cpu_kernel_generator),
    gpu_kernel=PallasOpGenerator(gpu_kernel_generator),
)
event_ellmv_p.defjvp(jvp_spikes, jvp_weights, None)
event_ellmv_p.def_transpose_rule(transpose_rule)


def event_ellmv_p_call(
    spikes, weights, indices,
    *,
    n_post, block_size, float_as_event
):
    n_conn = indices.shape[1]
    if block_size is None:
        if n_conn <= 16:
            block_size = 16
        elif n_conn <= 32:
            block_size = 32
        elif n_conn <= 64:
            block_size = 64
        elif n_conn <= 128:
            block_size = 128
        elif n_conn <= 256:
            block_size = 256
        else:
            block_size = 128
    return event_ellmv_p(
        spikes,
        weights,
        indices,
        outs=[jax.ShapeDtypeStruct([n_post], weights.dtype)],
        block_size=block_size,
        float_as_event=float_as_event,
        n_pre=spikes.shape[0],
        n_conn=indices.shape[1],
        n_post=n_post,
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        spike_info=jax.ShapeDtypeStruct(spikes.shape, spikes.dtype),
    )


def ell_cpu_kernel_generator(
    weight_info: jax.ShapeDtypeStruct,
    **kwargs
):
    import numba  # pylint: disable=import-outside-toplevel

    if jnp.size(weight_info) == 1:
        @numba.njit(**op_environ.numba_setting)
        def ell_mv(vector, weights, indices, posts):
            posts[:] = 0.
            w = weights[()]
            for i in range(vector.shape[0]):
                wv = w * vector[i]
                for j in range(indices.shape[1]):
                    posts[indices[i, j]] += wv

    else:
        @numba.njit(**op_environ.numba_setting)
        def ell_mv(vector, weights, indices, posts):
            posts[:] = 0.
            for i in range(vector.shape[0]):
                for j in range(indices.shape[1]):
                    posts[indices[i, j]] += weights[i, j] * vector[i]

    return ell_mv


def ell_gpu_kernel_generator(
    block_size: int,
    n_pre: int,
    n_conn: int,
    n_post: int,
    weight_info: jax.ShapeDtypeStruct,
    **kwargs
):
    homo = jnp.size(weight_info) == 1

    if homo:
        def _kernel(
            vec_ref, ind_ref, _, out_ref,
        ):
            # 每个block 处理 [block_size] 大小的vector
            # 每个block 处理 [block_size, block_size] 大小的indices 和 weights

            # -------------------------------
            # vec_ref: [block_size]
            # ind_ref: [block_size, block_size]
            # out_ref: [n_post]

            r_pid = pl.program_id(0)
            c_start = pl.program_id(1) * block_size
            mask = jnp.arange(block_size) + c_start
            row_length = jnp.minimum(n_pre - r_pid * block_size, block_size)

            def body_fn(j, _):
                y = vec_ref[j] * jnp.ones(block_size, dtype=weight_info.dtype)
                ind = pl.load(ind_ref, (j, pl.dslice(None)), mask=mask)
                pl.atomic_add(out_ref, ind, y, mask=mask)

            jax.lax.fori_loop(0, row_length, body_fn, None)

        # heterogeneous weights
        kernel = pl.pallas_call(
            _kernel,
            out_shape=[
                jax.ShapeDtypeStruct((n_post,), weight_info.dtype),
            ],
            in_specs=[
                pl.BlockSpec((block_size,), lambda i, j: i),  # vec_ref
                pl.BlockSpec((block_size, block_size), lambda i, j: (i, j)),  # ind_ref
                pl.BlockSpec((n_post,), lambda i, j: 0)  # out_ref
            ],
            grid=(
                pl.cdiv(n_pre, block_size),
                pl.cdiv(n_conn, block_size),
            ),
            input_output_aliases={2: 0},
            interpret=False
        )
        return lambda vector, weight, indices: kernel(vector, indices, jnp.zeros(n_post, dtype=weight.dtype)) * weight

    else:
        def _kernel(
            vec_ref, ind_ref, w_ref, _, out_ref,
        ):
            # 每个block 处理 [block_size] 大小的vector
            # 每个block 处理 [block_size, n_conn] 大小的indices 和 weights

            # -------------------------------
            # vec_ref: [block_size]
            # ind_ref: [block_size, block_size]
            # w_ref: [block_size, block_size]
            # out_ref: [n_post]

            r_pid = pl.program_id(0)
            c_start = pl.program_id(1) * block_size
            mask = jnp.arange(block_size) + c_start
            row_length = jnp.minimum(n_pre - r_pid * block_size, block_size)

            def body_fn(j, _):
                w = pl.load(w_ref, (j, pl.dslice(None)), mask=mask)
                y = w * vec_ref[j]
                ind = pl.load(ind_ref, (j, pl.dslice(None)), mask=mask)
                pl.atomic_add(out_ref, ind, y, mask=mask)

            jax.lax.fori_loop(0, row_length, body_fn, None)

        # heterogeneous weights
        kernel = pl.pallas_call(
            _kernel,
            out_shape=[
                jax.ShapeDtypeStruct((n_post,), weight_info.dtype),
            ],
            in_specs=[
                pl.BlockSpec((block_size,), lambda i, j: i),  # vec_ref
                pl.BlockSpec((block_size, block_size), lambda i, j: (i, j)),  # ind_ref
                pl.BlockSpec((block_size, block_size), lambda i, j: (i, j)),  # w_ref
                pl.BlockSpec((n_post,), lambda i: 0)  # out_ref
            ],
            grid=(
                pl.cdiv(n_pre, block_size),
                pl.cdiv(n_conn, block_size),
            ),
            input_output_aliases={3: 0},
            interpret=False
        )
        return lambda vector, weight, indices: kernel(vector, indices, weight, jnp.zeros(n_post, dtype=weight.dtype))


def jvp_weights_no_spk(w_dot, vector, weights, indices, *, block_size, n_post, **kwargs):
    return ellmv_p_call(
        vector,
        w_dot,
        indices,
        block_size=block_size,
        n_post=n_post,
    )


def transpose_rule_no_spk(
    ct, vector, weights, indices,
    *,
    n_post, block_size, weight_info, **kwargs
):
    if ad.is_undefined_primal(indices):
        raise ValueError("Cannot transpose with respect to sparse indices.")

    ct = ct[0]

    # ∂L/∂spk = ∂L/∂y * ∂y/∂spk
    homo = weight_info.size == 1
    if ad.is_undefined_primal(vector):
        if homo:
            # homogeneous weight
            ct_spk = jax.vmap(lambda idx: jnp.sum(ct[idx] * weights))(indices)
        else:
            # heterogeneous weight
            ct_spk = jax.vmap(lambda idx, w: jnp.inner(ct[idx], w))(indices, weights)
        return (ad.Zero(vector) if type(ct) is ad.Zero else ct_spk), weights, indices

    else:
        # ∂L/∂w = ∂L/∂y * ∂y/∂w
        if homo:
            # scalar
            ct_gmax = ellmv_p_call(
                vector,
                jnp.asarray(1., dtype=weight_info.dtype),
                indices,
                block_size=block_size,
                n_post=n_post,
            )
            ct_gmax = jnp.inner(ct, ct_gmax[0])
        else:
            ct_gmax = jax.vmap(lambda vec, one_ind: ct[one_ind] * vec)(vector, indices)
        return vector, (ad.Zero(weights) if type(ct) is ad.Zero else ct_gmax), indices


ellmv_p = XLACustomOp(
    'ell_mv',
    cpu_kernel=NumbaOpGenerator(ell_cpu_kernel_generator),
    gpu_kernel=PallasOpGenerator(ell_gpu_kernel_generator),
)
ellmv_p.defjvp(jvp_spikes, jvp_weights_no_spk, None)
ellmv_p.def_transpose_rule(transpose_rule_no_spk)


def ellmv_p_call(vector, weights, indices, *, n_post, block_size):
    n_conn = indices.shape[1]
    if block_size is None:
        if n_conn <= 16:
            block_size = 16
        elif n_conn <= 32:
            block_size = 32
        elif n_conn <= 64:
            block_size = 64
        elif n_conn <= 128:
            block_size = 128
        elif n_conn <= 256:
            block_size = 256
        else:
            block_size = 128
    return ellmv_p(
        vector,
        weights,
        indices,
        n_post=n_post,
        n_pre=indices.shape[0],
        n_conn=indices.shape[1],
        block_size=block_size,
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        outs=[jax.ShapeDtypeStruct([n_post], weights.dtype)]
    )
