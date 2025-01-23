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
import unittest

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

import pytest
import brainstate as bst
from typing import Any


def test1():
    import numba
    def add_vectors_kernel(x_ref, y_ref, o_ref):
        x, y = x_ref[...], y_ref[...]
        o_ref[...] = x + y

    def cpu_kernel(**kwargs):
        @numba.njit
        def add_kernel_numba(x, y, out):
            out[...] = x + y

        return add_kernel_numba

    def gpu_kernel(x_info, **kwargs):
        return pl.pallas_call(
            add_vectors_kernel,
            out_shape=[jax.ShapeDtypeStruct(x_info.shape, x_info.dtype)],
            interpret=jax.default_backend() == 'cpu',
        )

    prim = bst.event.XLACustomKernel(
        'add',
        cpu_kernel=bst.event.NumbaKernelGenerator(cpu_kernel),
        gpu_kernel=bst.event.PallasKernelGenerator(gpu_kernel),
    )

    a = bst.random.rand(64)
    b = bst.random.rand(64)
    x_info = jax.ShapeDtypeStruct(a.shape, a.dtype)
    r1 = prim(a, b, outs=[jax.ShapeDtypeStruct((64,), jax.numpy.float32)], x_info=x_info)
    r2 = gpu_kernel(x_info)(a, b)

    assert jnp.allclose(r1[0], r2[0])


@pytest.mark.skipif(jax.default_backend() != 'gpu', reason="No GPU available")
class TestWarpGPU(unittest.TestCase):
    def test_warp1(self):
        import warp as wp

        # generic kernel definition using Any as a placeholder for concrete types
        @wp.kernel
        def scale(x: wp.array(dtype=float), y: wp.array(dtype=float), ):
            i = wp.tid()
            y[i] = x[i] * x[i]

        data = jnp.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=jnp.float32)

        op = bst.event.XLACustomKernel(
            name="scale",
            gpu_kernel=bst.event.WarpKernelGenerator(
                lambda **kwargs: scale,
                dim=data.shape,
            ),
        )
        r = op.call(data, outs=jax.ShapeDtypeStruct(data.shape, data.dtype))
        print(r)

        self.assertTrue(jnp.allclose(r, data * data))

    def test_warp_scalar(self):
        import warp as wp

        # generic kernel definition using Any as a placeholder for concrete types
        @wp.kernel
        def scale2(
            x: wp.array(dtype=float),
            s: wp.array(dtype=float),
            y: wp.array(dtype=float)
        ):
            i = wp.tid()
            y[i] = s[0] * x[i]

        data = jnp.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=jnp.float32)

        op = bst.event.XLACustomKernel(
            name="scale2",
            gpu_kernel=bst.event.WarpKernelGenerator(
                lambda **kwargs: scale2,
                dim=data.shape,
            ),
        )
        r = op.call(data, jnp.asarray([1.5]), outs=jax.ShapeDtypeStruct(data.shape, data.dtype))
        print(r)
        self.assertTrue(jnp.allclose(r, 1.5 * data))

    def test_warp_two_vectors(self):
        import warp as wp

        # generic kernel definition using Any as a placeholder for concrete types
        @wp.kernel
        def scale2(
            x: wp.array(dtype=float),
            y: wp.array(dtype=float),
            z: wp.array(dtype=float)
        ):
            i = wp.tid()
            z[i] = x[i] * y[i]

        xs = jnp.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=jnp.float32)
        ys = bst.random.rand_like(xs)

        op = bst.event.XLACustomKernel(
            name="scale2",
            gpu_kernel=bst.event.WarpKernelGenerator(
                lambda **kwargs: scale2,
                dim=xs.shape,
            ),
        )
        r = op.call(xs, ys, outs=jax.ShapeDtypeStruct(xs.shape, xs.dtype))
        print(r)

        self.assertTrue(jnp.allclose(r, xs * ys))
