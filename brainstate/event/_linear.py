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
from typing import Union, Callable, Optional

import brainunit as u
import jax
import jax.experimental.pallas as pl
import jax.numpy as jnp

from brainstate._state import ParamState, State
from brainstate.init import param
from brainstate.nn._module import Module
from brainstate.typing import ArrayLike, Size

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
    name : str, optional
        Name of the module.
    """

    __module__ = 'brainstate.event'

    def __init__(
        self,
        in_size: Size,
        out_size: Size,
        weight: Union[Callable, ArrayLike],
        grad_mode: str = 'vjp',
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

        # gradient mode
        assert grad_mode in ['vjp', 'jvp'], f"Unsupported grad_mode: {grad_mode}"
        self.grad_mode = grad_mode

        # maximum synaptic conductance
        weight = param(weight, (self.in_size[-1], self.out_size[-1]), allow_none=False)
        self.weight = ParamState(weight)

    def update(self, spk: jax.Array) -> Union[jax.Array, u.Quantity]:
        weight = self.weight.value if isinstance(self.weight, State) else self.weight
        if u.math.size(weight) == 1:
            return u.math.ones(self.out_size) * (u.math.sum(spk) * weight)

        device_kind = jax.devices()[0].platform  # spk.device.device_kind
        if device_kind == 'cpu':
            impl = CPUImpl(
                block_size=self.block_size,
                grad_mode=self.grad_mode,
                float_as_event=self.float_as_event,
            )

        elif device_kind in ['gpu', 'tpu']:
            impl = GPUImpl(
                block_size=self.block_size,
                grad_mode=self.grad_mode,
                float_as_event=self.float_as_event,
            )

        else:
            raise ValueError(f"Unsupported device: {device_kind}")

        return impl(spk, weight)


class Implementation:
    """
    The FixedProb module implements a fixed probability connection with CSR sparse data structure.

    Parameters
    ----------
    grad_mode : str, optional
        Gradient mode. Default is 'vjp'. Can be 'vjp' or 'jvp'.
    float_as_event : bool, optional
        Whether to treat float as event. Default is True.
    block_size : int, optional
        Block size for parallel computation. Default is 64. This is only used for GPU.
    """

    def __init__(
        self,
        block_size: int,
        grad_mode: str = 'vjp',
        float_as_event: bool = True,
    ):
        self.block_size = block_size
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
            spk = jnp.asarray(spk)

        n_pre = weight.shape[0]
        n_post = weight.shape[1]

        def mv(spk_vector):
            assert spk_vector.ndim == 1, f"spk must be 1D. Got: {spk.ndim}"
            if jnp.size(weight) == 1:
                assert isinstance(n_post, int), f"n_post must be an integer when weight is homogenous. Got: {n_post}"
                return jnp.ones((n_post,), dtype=weight.dtype) * (jnp.sum(spk_vector) * weight)

            if self.grad_mode == 'vjp':
                post = self.mv_vjp(weight, spk_vector)
            elif self.grad_mode == 'jvp':
                post = self.mv_jvp(weight, spk_vector)
            else:
                raise ValueError(f"Unsupported grad_mode: {self.grad_mode}")
            return post

        assert spk.ndim >= 1, f"spk must be at least 1D. Got: {spk.ndim}"
        assert weight.ndim in [2, 0], f"weight must be 2D or 0D. Got: {weight.ndim}"

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
        ct_spk = jnp.matmul(weight, ct)

        # ∂L/∂w = ∂L/∂y * ∂y/∂w
        def map_fn(sp):
            if spk.dtype == jnp.bool_:
                d_gmax = jax.lax.cond(sp, lambda: ct, lambda: jnp.zeros_like(ct))
            else:
                if self.float_as_event:
                    d_gmax = jax.lax.cond(sp == 0., lambda: jnp.zeros_like(ct), lambda: ct)
                else:
                    d_gmax = jax.lax.cond(sp == 0., lambda: jnp.zeros_like(ct), lambda: ct * sp)
            return d_gmax

        ct_gmax = jax.vmap(map_fn)(spk)
        return ct_gmax, ct_spk

    def mv_jvp_rule(self, primals, tangents):
        # forward pass
        weight, spk = primals
        y = self.mv(weight, spk)

        # forward gradients
        gmax_dot, spk_dot = tangents

        # ∂y/∂gmax
        dgmax = self.mv(gmax_dot, spk)

        # ∂y/∂gspk
        dspk = spk_dot @ weight
        return y, dgmax + dspk


class CPUImpl(Implementation):
    def mv(self, weight, spk) -> jax.Array:
        def scan_fn(post, xs):
            sp, w = xs
            if spk.dtype == jnp.bool_:
                post = jax.lax.cond(sp, lambda p: p + w, lambda p: p, post)
            else:
                if self.float_as_event:
                    post = jax.lax.cond(sp == 0., lambda p: p, lambda p: p + w, post)
                else:
                    post = jax.lax.cond(sp == 0., lambda p: p, lambda p: p + w * sp, post)
            return post, None

        return jax.lax.scan(
            scan_fn,
            jnp.zeros(weight.shape[1], dtype=weight.dtype),
            [spk, weight]
        )[0]


class GPUImpl(Implementation):

    def _mv_kernel(
        self,
        w_ref,
        sp_ref,
        post_ref,
        *,
        n_pre: int
    ):
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
                if self.float_as_event:
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
        mask = jnp.arange(self.block_size) + pid * self.block_size < n_pre
        pl.store(post_ref, pl.dslice(None, None), post, mask=mask)

    @partial(jax.jit, static_argnums=(0,))
    def mv(self, weight, spk) -> jax.Array:
        n_pre = weight.shape[0]
        n_post = weight.shape[1]

        kernel = pl.pallas_call(
            partial(self._mv_kernel, n_pre=n_pre),
            out_shape=jax.ShapeDtypeStruct([weight.shape[1]], weight.dtype),
            out_specs=pl.BlockSpec((self.block_size,), lambda i: i),
            in_specs=[
                pl.BlockSpec((n_pre, self.block_size), lambda i: (0, i)),
                pl.BlockSpec((n_pre,), lambda i: 0),
            ],
            grid=(
                pl.cdiv(n_post, self.block_size),
            ),
            interpret=self.interpret
        )
        return kernel(weight, spk)
