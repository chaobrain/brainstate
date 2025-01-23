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

import importlib.util
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl

import brainstate as bst

warp_installed = importlib.util.find_spec('warp') is not None
numba_installed = importlib.util.find_spec('numba') is not None

if warp_installed:
    import warp as wp
    import numba


@pytest.mark.skipif(not numba_installed, reason="Numba not installed")
class TestNumbaCPU(unittest.TestCase):
    def test1(self):
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

    def test_warp_change_with_dtype(self):
        def generate(**kwargs):
            outs = kwargs["outs"][0]
            dtype = bst.event.dtype_to_warp_type(outs.dtype)

            # generic kernel definition using Any as a placeholder for concrete types
            @wp.kernel
            def scale(x: wp.array(dtype=dtype), y: wp.array(dtype=dtype)):
                i = wp.tid()
                y[i] = x[i] * x[i]

            return scale

        op = bst.event.XLACustomKernel(
            name="scale",
            gpu_kernel=bst.event.WarpKernelGenerator(
                generate,
                dim=lambda **kwargs: kwargs["outs"][0].shape,
            ),
        )

        @jax.jit
        def f(x):
            return op.call(x, outs=jax.ShapeDtypeStruct(x.shape, x.dtype))

        with bst.environ.context(precision=64):
            print(f(jnp.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=jnp.float32)))
            print(f(bst.random.rand(20, dtype=jnp.float32)))
            print(f(bst.random.rand(20, dtype=jnp.float16)))
            print(f(bst.random.rand(20, dtype=jnp.float64)))

    def test_warp_scalar(self):
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

    def test_tile1(self):
        TILE_SIZE = wp.constant(256)
        TILE_THREADS = 64

        @wp.kernel
        def compute(
            a: wp.array2d(dtype=float),
            b: wp.array2d(dtype=float),
        ):
            # obtain our block index
            i = wp.tid()

            # load a row from global memory
            t = wp.tile_load(a[i], 0, TILE_SIZE)

            # cooperatively compute the sum of the tile elements; s is a 1x1 tile
            s = wp.tile_sum(t)

            # store s in global memory
            wp.tile_store(b[0], i, s)

        N = 10
        a_np = np.arange(N).reshape(-1, 1) * np.ones((1, 256), dtype=float)

        op = bst.event.XLACustomKernel(
            name="mm",
            gpu_kernel=bst.event.WarpKernelGenerator(
                lambda **kwargs: compute,
                dim=(a_np.shape[0], TILE_THREADS),
                block_dim=TILE_THREADS,
            ),
        )
        r = op.call(
            jax.numpy.asarray(a_np, dtype=jax.numpy.float32),
            outs=jax.core.ShapedArray([1, N], dtype=jax.numpy.float32)
        )
        r_true = a_np.sum(axis=1)
        print(r)
        print(r_true)
        self.assertTrue(jnp.allclose(r[0], r_true))

    def test_tile_matrix_multiplication(self):
        TILE_M = wp.constant(8)
        TILE_N = wp.constant(4)
        TILE_K = wp.constant(8)
        TILE_THREADS = 64

        @wp.kernel
        def tile_gemm(
            A: wp.array2d(dtype=float),
            B: wp.array2d(dtype=float),
            C: wp.array2d(dtype=float),
        ):
            # output tile index
            i, j = wp.tid()

            sum = wp.tile_zeros(m=TILE_M, n=TILE_N, dtype=wp.float32)

            M = A.shape[0]
            N = B.shape[1]
            K = A.shape[1]

            count = int(K / TILE_K)

            for k in range(0, count):
                a = wp.tile_load(A, i, k, m=TILE_M, n=TILE_K)
                b = wp.tile_load(B, k, j, m=TILE_K, n=TILE_N)

                # sum += a*b
                wp.tile_matmul(a, b, sum)

            wp.tile_store(C, i, j, sum)

        # generate some tile aligned matrix dimensions
        M = TILE_M * 7
        K = TILE_K * 6
        N = TILE_N * 5

        bst.random.seed(42)
        A = bst.random.random((M, K), dtype=np.float32)
        B = bst.random.random((K, N), dtype=np.float32)
        C_true = A @ B

        op = bst.event.XLACustomKernel(
            name="mm",
            gpu_kernel=bst.event.WarpKernelGenerator(
                lambda **kwargs: tile_gemm,
                dim=(int(M / TILE_M), int(N / TILE_N), TILE_THREADS),
                block_dim=TILE_THREADS,
            ),
        )
        r = op.call(A, B, outs=jax.core.ShapedArray([M, N], dtype=jax.numpy.float32))

        self.assertTrue(jnp.allclose(r, C_true, atol=1e-3))
