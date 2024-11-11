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

from functools import partial
from typing import Union, Callable, Optional, Tuple

import brainunit as u
import jax
import jax.experimental.pallas as pl
import jax.numpy as jnp
import numpy as np

from brainstate._state import ParamState
from brainstate.compile import for_loop
from brainstate.init import param
from brainstate.nn._module import Module
from brainstate.random import RandomState
from brainstate.typing import ArrayLike, Size
from ._misc import FloatScalar

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
    grad_mode : str, optional
        Gradient mode. Default is 'vjp'. Can be 'vjp' or 'jvp'.

        - 'vjp': Compatible with the vector-Jacobian product (VJP) autodiff method.
        - 'jvp': Compatible with the Jacobian-vector product (JVP) autodiff method.
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
        grad_mode: str = 'vjp',
        float_as_event: bool = True,
        block_size: int = 64,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)

        # network parameters
        self.in_size = in_size
        self.out_size = out_size
        self.n_conn = int(self.out_size[-1] * prob)
        if self.n_conn < 1:
            raise ValueError(f"The number of connections must be at least 1. "
                             f"Got: int({self.out_size[-1]} * {prob}) = {self.n_conn}")
        self.float_as_event = float_as_event
        self.block_size = block_size

        # gradient mode
        assert grad_mode in ['vjp', 'jvp'], f"Unsupported grad_mode: {grad_mode}"
        self.grad_mode = grad_mode

        # indices of post connected neurons
        with jax.ensure_compile_time_eval():
            if allow_multi_conn:
                rng = np.random.RandomState(seed)
                self.indices = rng.randint(0, self.out_size[-1], size=(self.in_size[-1], self.n_conn))
            else:
                rng = RandomState(seed)
                self.indices = for_loop(lambda i: rng.choice(self.out_size[-1], size=(self.n_conn,), replace=False),
                                        np.arange(self.in_size[-1]))
            self.indices = u.math.asarray(self.indices)

        # maximum synaptic conductance
        weight = param(weight, (self.in_size[-1], self.n_conn), allow_none=False)
        self.weight = ParamState(weight)

    def update(self, spk: jax.Array) -> Union[jax.Array, u.Quantity]:
        device_kind = jax.devices()[0].platform  # spk.device.device_kind

        if device_kind == 'cpu':
            return CPUImpl(
                self.indices,
                n_post=self.out_size[-1],
                block_size=self.block_size,
                grad_mode=self.grad_mode,
                float_as_event=self.float_as_event
            )(spk, self.weight.value)
        elif device_kind in ['gpu', 'tpu']:
            return GPUImpl(
                self.indices,
                n_post=self.out_size[-1],
                block_size=self.block_size,
                grad_mode=self.grad_mode,
                float_as_event=self.float_as_event
            )(spk, self.weight.value)

        else:
            raise ValueError(f"Unsupported device: {device_kind}")


class Implementation:
    """
    The FixedProb module implements a fixed probability connection with CSR sparse data structure.

    Parameters
    ----------
    n_post : int
        Number of post-synaptic neurons.
    indices : jax.Array
        Indices of post connected neurons.
    grad_mode : str, optional
        Gradient mode. Default is 'vjp'. Can be 'vjp' or 'jvp'.
    float_as_event : bool, optional
        Whether to treat float as event. Default is True.
    block_size : int, optional
        Block size for parallel computation. Default is 64. This is only used for GPU.
    """

    def __init__(
        self,
        indices: jax.Array,
        n_post: int,
        block_size: int,
        grad_mode: str = 'vjp',
        float_as_event: bool = True,
    ):
        assert u.get_unit(indices).is_unitless, f"indices must be unitless. Got: {u.get_unit(indices)}"
        assert indices.ndim == 2, f"indices must be 2D. Got: {indices.ndim}"

        self.block_size = block_size
        self.indices = indices
        self.n_pre = indices.shape[0]
        self.n_post = n_post
        self.n_conn = indices.shape[1]
        self.float_as_event = float_as_event
        self.grad_mode = grad_mode
        self.interpret = False

        # vjp rule
        self.mv_vjp = jax.custom_vjp(self.mv)
        self.mv_vjp.defvjp(self.mv_fwd, self.mv_bwd)

        # jvp rule
        self.mv_jvp = jax.custom_jvp(self.mv)
        self.mv_jvp.defjvp(self.mv_jvp_rule)

    def __call__(self, spk, weight):
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
            indices = jnp.asarray(self.indices)
            spk = jnp.asarray(spk)

        def mv(spk_vector):
            assert spk_vector.ndim == 1, f"spk must be 1D. Got: {spk.ndim}"
            if self.grad_mode == 'vjp':
                post_data = self.mv_vjp(weight, spk_vector)
            elif self.grad_mode == 'jvp':
                post_data = self.mv_jvp(weight, spk_vector)
            else:
                raise ValueError(f"Unsupported grad_mode: {self.grad_mode}")
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

    def mv(self, *args):
        raise NotImplementedError

    def mv_fwd(self, weight, spk):
        fwd = self.mv(weight, spk)
        return fwd, (weight, spk)

    def mv_bwd(self, res, ct):
        weight, spk = res

        # ∂L/∂spk = ∂L/∂y * ∂y/∂spk
        homo = jnp.size(weight) == 1
        if homo:
            # homogeneous weight
            ct_spk = jax.vmap(lambda idx: jnp.sum(ct[idx] * weight))(self.indices)
        else:
            # heterogeneous weight
            ct_spk = jax.vmap(lambda idx, w: jnp.inner(ct[idx], w))(self.indices, weight)

        # ∂L/∂w = ∂L/∂y * ∂y/∂w
        if homo:
            # scalar
            ct_gmax = self.mv(jnp.asarray(1., dtype=weight.dtype), spk)
            ct_gmax = jnp.inner(ct, ct_gmax)
        else:
            n_conn = weight.shape[1]
            w_dtype = weight.dtype

            def map_fn(one_spk, one_ind):
                if spk.dtype == jnp.bool_:
                    return jax.lax.cond(
                        one_spk,
                        lambda: ct[one_ind],
                        lambda: jnp.zeros([n_conn], w_dtype)
                    )
                else:
                    if self.float_as_event:
                        return jax.lax.cond(
                            one_spk == 0.,
                            lambda: jnp.zeros([n_conn], w_dtype),
                            lambda: ct[one_ind]
                        )
                    else:
                        return jax.lax.cond(
                            one_spk == 0.,
                            lambda: jnp.zeros([n_conn], w_dtype),
                            lambda: ct[one_ind] * one_spk
                        )

            ct_gmax = jax.vmap(map_fn)(spk, self.indices)
        return ct_gmax, ct_spk

    def mv_jvp_rule(self, *args):
        raise NotImplementedError


# -------------------
# CPU Implementation
# -------------------

class CPUImpl(Implementation):
    def mv(self, weight, spk) -> jax.Array:
        def scan_fn(post, i):
            w = weight if jnp.size(weight) == 1 else weight[i]
            ids = self.indices[i]
            sp = spk[i]
            if spk.dtype == jnp.bool_:
                post = jax.lax.cond(sp, lambda: post.at[ids].add(w), lambda: post)
            else:
                if self.float_as_event:
                    post = jax.lax.cond(sp == 0., lambda: post, lambda: post.at[ids].add(w))
                else:
                    post = jax.lax.cond(sp == 0., lambda: post, lambda: post.at[ids].add(w * sp))
            return post, None

        return jax.lax.scan(scan_fn, jnp.zeros((self.n_post,), dtype=weight.dtype), np.arange(len(spk)))[0]

    def mv_jvp_rule(self, primals, tangents):
        # forward pass
        weight, spk = primals
        y = self.mv(weight, spk)

        # forward gradients
        gmax_dot, spk_dot = tangents

        # ∂y/∂gmax
        dgmax = self.mv(gmax_dot, spk)

        def scan_fn(post, i):
            ids = self.indices[i]
            w = weight if jnp.size(weight) == 1 else weight[i]
            post = post.at[ids].add(w * spk_dot[i])
            return post, None

        # ∂y/∂gspk
        dspk = jax.lax.scan(
            scan_fn,
            jnp.zeros((self.n_post,), dtype=weight.dtype),
            np.arange(len(spk))
        )[0]
        return y, dgmax + dspk


# -------------------
# CPU Implementation
# -------------------

class GPUImpl(Implementation):
    def _ell_mv_kernel_homo(
        self,
        sp_ref,
        ind_ref,
        _,
        y_ref,
    ):
        pid = pl.program_id(0)
        row_length = jnp.minimum(self.n_pre - pid * self.block_size, self.block_size)

        def loop_fn(i_start, i_end):
            def body_fn(j, _):
                def true_fn():
                    y_ref[ind_ref[j, i_start: i_end]] += 1.0

                if sp_ref.dtype == jnp.bool_:
                    jax.lax.cond(sp_ref[j], true_fn, lambda: None)
                else:
                    jax.lax.cond(sp_ref[j] != 0., true_fn, lambda: None)

            jax.lax.fori_loop(0, row_length, body_fn, None)

        for i in range(0, ind_ref.shape[1], self.block_size):
            loop_fn(i, i + self.block_size if (i + self.block_size < ind_ref.shape[1]) else ind_ref.shape[1])

    def _ell_mv_kernel_heter(
        self,
        sp_ref,
        ind_ref,
        w_ref,
        _,
        y_ref,
    ):
        pid = pl.program_id(0)
        row_length = jnp.minimum(self.n_pre - pid * self.block_size, self.block_size)

        def loop_fn(i_start, i_end):
            def body_fn(j, _):
                if sp_ref.dtype == jnp.bool_:
                    def true_fn():
                        y_ref[ind_ref[j, i_start: i_end]] += w_ref[j, i_start: i_end]

                    jax.lax.cond(sp_ref[j], true_fn, lambda: None)

                else:
                    def true_fn(spk):
                        if self.float_as_event:
                            y_ref[ind_ref[j, i_start: i_end]] += w_ref[j, i_start: i_end]
                        else:
                            y_ref[ind_ref[j, i_start: i_end]] += w_ref[j, i_start: i_end] * spk

                    sp_ = sp_ref[j]
                    jax.lax.cond(sp_ != 0., true_fn, lambda _: None, sp_)

            jax.lax.fori_loop(0, row_length, body_fn, None)

        for i in range(0, ind_ref.shape[1], self.block_size):
            loop_fn(i, i + self.block_size if (i + self.block_size < ind_ref.shape[1]) else ind_ref.shape[1])

    @partial(jax.jit, static_argnums=0)
    def _event_ell_mv(self, weight, spikes):
        # 对于具有形状 [n_event] 的 spikes 向量，以及形状 [n_event, n_conn] 的 indices 和 weights 矩阵，
        # 这个算子的计算逻辑为：
        #
        # - 每个block处理 block_size 个事件，每个事件对应一个 pre-synaptic neuron
        # - 每个block处理 [block_size, n_conn] 个 indices 和 weights

        n_event = spikes.shape[0]
        dtype = weight.dtype

        if jnp.size(weight) == 1:
            # homogenous weights
            kernel = pl.pallas_call(
                self._ell_mv_kernel_homo,
                out_shape=jax.ShapeDtypeStruct((self.n_post,), dtype),
                in_specs=[
                    pl.BlockSpec((self.block_size,), lambda i: i),
                    pl.BlockSpec((self.block_size, self.indices.shape[1]), lambda i: (i, 0)),
                    pl.BlockSpec((self.n_post,), lambda i: 0)
                ],
                grid=(
                    pl.cdiv(self.n_pre, self.block_size),
                ),
                input_output_aliases={2: 0},
                interpret=self.interpret
            )
            return kernel(spikes, self.indices, jnp.zeros(self.n_post, dtype=dtype)) * weight

        else:

            # heterogeneous weights
            kernel = pl.pallas_call(
                self._ell_mv_kernel_heter,
                out_shape=jax.ShapeDtypeStruct((self.n_post,), dtype),
                in_specs=[
                    pl.BlockSpec((self.block_size,), lambda i: i),
                    pl.BlockSpec((self.block_size, self.indices.shape[1]), lambda i: (i, 0)),
                    pl.BlockSpec((self.block_size, weight.shape[1]), lambda i: (i, 0)),
                    pl.BlockSpec((self.n_post,), lambda i: 0)
                ],
                grid=(
                    pl.cdiv(n_event, self.block_size),
                ),
                input_output_aliases={3: 0},
                interpret=self.interpret
            )
            return kernel(spikes, self.indices, weight, jnp.zeros(self.n_post, dtype=dtype))

    def mv(self, weight, spikes):
        return self._event_ell_mv(weight, spikes)

    def mv_jvp_rule(self, primals, tangents):
        # forward pass
        weight, spk = primals
        y = self.mv(weight, spk)

        # forward gradients
        gmax_dot, spk_dot = tangents

        # ∂y/∂gmax
        dgmax = self.mv(gmax_dot, spk)

        # ∂y/∂gspk
        dspk = self._ell_mv(spk_dot, weight)
        return y, dgmax + dspk

    @partial(jax.jit, static_argnums=0)
    def _ell_mv(
        self,
        vector,
        weight,
    ):
        n_conn = self.indices.shape[1]
        dtype = weight.dtype
        homo = jnp.size(weight) == 1

        if homo:
            def _kernel(
                vec_ref, ind_ref, _, out_ref,
            ):
                # 每个block 处理 [block_size] 大小的vector
                # 每个block 处理 [block_size, n_conn] 大小的indices 和 weights

                # -------------------------------
                # vec_ref: [block_size]
                # ind_ref: [block_size, n_conn]
                # w_ref: [block_size, n_conn]
                # out_ref: [n_post]

                pid = pl.program_id(0)
                row_length = jnp.minimum(self.n_pre - pid * self.block_size, self.block_size)

                def body_fn(col_start, col_end):
                    def body_fn(j, _):
                        y = vec_ref[j] * jnp.ones(col_end - col_start)
                        out_ref[ind_ref[j, col_start: col_end]] += y

                    jax.lax.fori_loop(0, row_length, body_fn, None)

                for i in range(0, n_conn, self.block_size):
                    body_fn(i, i + self.block_size if (i + self.block_size < n_conn) else n_conn)

            # heterogeneous weights
            kernel = pl.pallas_call(
                _kernel,
                out_shape=jax.ShapeDtypeStruct((self.n_post,), dtype),
                in_specs=[
                    pl.BlockSpec((self.block_size,), lambda i: i),
                    pl.BlockSpec((self.block_size, self.indices.shape[1]), lambda i: (i, 0)),
                    pl.BlockSpec((self.n_post,), lambda i: 0)
                ],
                grid=(
                    pl.cdiv(self.n_pre, self.block_size),
                ),
                input_output_aliases={2: 0},
                interpret=self.interpret
            )
            return kernel(vector, self.indices, jnp.zeros(self.n_post, dtype=dtype)) * weight

        else:
            def _kernel(
                vec_ref, ind_ref, w_ref, _, out_ref,
            ):
                # 每个block 处理 [block_size] 大小的vector
                # 每个block 处理 [block_size, n_conn] 大小的indices 和 weights

                # -------------------------------
                # vec_ref: [block_size]
                # ind_ref: [block_size, n_conn]
                # w_ref: [block_size, n_conn]
                # out_ref: [n_post]

                pid = pl.program_id(0)
                row_length = jnp.minimum(self.n_pre - pid * self.block_size, self.block_size)

                def body_fn(col_start, col_end):
                    def body_fn(j, _):
                        y = w_ref[j, col_start: col_end] * vec_ref[j]
                        out_ref[ind_ref[j, col_start: col_end]] += y

                    jax.lax.fori_loop(0, row_length, body_fn, None)

                for i in range(0, n_conn, self.block_size):
                    body_fn(i, i + self.block_size if (i + self.block_size < n_conn) else n_conn)

            # heterogeneous weights
            kernel = pl.pallas_call(
                _kernel,
                out_shape=jax.ShapeDtypeStruct((self.n_post,), dtype),
                in_specs=[
                    pl.BlockSpec((self.block_size,), lambda i: i),
                    pl.BlockSpec((self.block_size, self.indices.shape[1]), lambda i: (i, 0)),
                    pl.BlockSpec((self.block_size, weight.shape[1]), lambda i: (i, 0)),
                    pl.BlockSpec((self.n_post,), lambda i: 0)
                ],
                grid=(
                    pl.cdiv(self.n_pre, self.block_size),
                ),
                input_output_aliases={3: 0},
                interpret=self.interpret
            )
            return kernel(vector, self.indices, weight, jnp.zeros(self.n_post, dtype=dtype))

    def _event_ell_weight(
        self,
        events,
        post_ct,
        *,
        w_shape: Tuple[int, int],
        w_dtype: jax.typing.DTypeLike,
    ):
        n_pre = events.shape[0]
        n_post = post_ct.shape[0]

        def _kernel(
            spk_ref, ind_ref, post_ref, _, w_ref,
        ):
            # 每个block 处理 [block_size] 大小的events
            # 每个block 处理 [block_size, n_conn] 大小的indices 和 weights

            # -------------------------------
            # spk_ref: [block_size]
            # ind_ref: [block_size, n_conn]
            # post_ref: [n_post]
            # w_ref: [block_size, n_conn]

            pid = pl.program_id(0)
            row_length = jnp.minimum(n_pre - pid * self.block_size, self.block_size)

            def body_fn(j, _):
                if spk_ref.dtype == jnp.bool_:
                    def true_fn():
                        w_ref[j, ...] += post_ref[ind_ref[j, ...]]

                    jax.lax.cond(spk_ref[j], true_fn, lambda: None)

                else:
                    def true_fn(spk):
                        if self.float_as_event:
                            w_ref[j, ...] += post_ref[ind_ref[j, ...]]
                        else:
                            w_ref[j, ...] += post_ref[ind_ref[j, ...]] * spk

                    spk = spk_ref[j]
                    jax.lax.cond(spk != 0., true_fn, lambda _: None, spk)

            jax.lax.fori_loop(0, row_length, body_fn, None)

        # heterogeneous weights
        kernel = pl.pallas_call(
            _kernel,
            out_shape=jax.ShapeDtypeStruct(w_shape, w_dtype),
            in_specs=[
                pl.BlockSpec((self.block_size,), lambda i: i),
                pl.BlockSpec((self.block_size, self.indices.shape[1]), lambda i: (i, 0)),
                pl.BlockSpec((n_post,), lambda i: 0),
                pl.BlockSpec(w_shape, lambda i: (i, 0))
            ],
            grid=(
                pl.cdiv(n_pre, self.block_size),
            ),
            input_output_aliases={3: 0},
            interpret=self.interpret
        )
        return kernel(events, self.indices, post_ct, jnp.zeros(w_shape, dtype=w_dtype))
