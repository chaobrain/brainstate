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

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

import brainstate as bst


def test1():
    import numba
    def add_vectors_kernel(x_ref, y_ref, o_ref):
        x, y = x_ref[...], y_ref[...]
        o_ref[...] = x + y

    def cpu_kernel(**kwargs):
        @numba.njit
        def add_kernel_numba(x, y, out):
            out[...] = x + y

    def gpu_kernel(x_info):
        return pl.pallas_call(
            add_vectors_kernel,
            out_shape=[jax.ShapeDtypeStruct(x_info.shape, x_info.dtype)],
            interpret=jax.default_backend() == 'cpu',
        )

    prim = bst.event.XLACustomOp(
        'add',
        cpu_kernel,
        gpu_kernel,
    )

    a = bst.random.rand(64)
    b = bst.random.rand(64)
    x_info = jax.ShapeDtypeStruct(a.shape, a.dtype)
    r1 = prim(a, b, outs=[jax.ShapeDtypeStruct((64,), jax.numpy.float32)], x_info=x_info)
    r2 = gpu_kernel(x_info)(a, b)

    assert jnp.allclose(r1[0], r2[0])
