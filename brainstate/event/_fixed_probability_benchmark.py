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

import os

os.environ['XLA_FLAGS'] = '--xla_cpu_use_thunk_runtime=false'
os.environ['JAX_TRACEBACK_FILTERING'] = 'off'

import jax

jax.config.update('jax_cpu_enable_async_dispatch', False)

import time
import brainstate as bst
import taichi as ti
import braintaichi as bti


@ti.kernel
def kernel(
    indices: ti.types.ndarray(ndim=2),
    weights: ti.types.ndarray(ndim=2),
    events: ti.types.ndarray(ndim=1),
    out: ti.types.ndarray(ndim=1),
):
    # ti.loop_config(serialize=True)
    # for i in range(indices.shape[0]):
    #     if events[i]:
    #         for j in range(indices.shape[1]):
    #             out[indices[i, j]] += weights[i, j]  # * events[i]
    for i, j in ti.ndrange(*indices.shape):
        if events[i]:
            out[indices[i, j]] += weights[i, j]


def taichi_fixedprob(indices, weights, events, *, n_post):
    return bti.XLACustomOp(cpu_kernel=kernel)(
        indices, weights, events,
        outs=[jax.ShapeDtypeStruct(shape=[n_post], dtype=weights.dtype)],
    )


def forward(n_pre, n_post, conn_prob, spk_prob, as_float: bool):
    linear = bst.event.FixedProb(n_pre, n_post, prob=conn_prob, weight=bst.init.Normal())
    ti_linear = jax.jit(taichi_fixedprob, static_argnames=['n_post'])
    spike = (bst.random.rand(n_pre) < spk_prob)

    if as_float:
        spike = spike.astype(float)

    @jax.jit
    def f1(spike):
        return linear(spike)

    weight = bst.init.Normal()([n_pre, n_post])

    @jax.jit
    def f2(spike):
        return spike @ weight

    y1 = jax.block_until_ready(f1(spike))
    y2 = jax.block_until_ready(f2(spike))
    y3 = jax.block_until_ready(ti_linear(linear.indices, linear.weight.value, spike, n_post=n_post))
    # print(jax.numpy.max(jax.numpy.abs(y2 - y1)))

    n = 1000
    t0 = time.time()
    for _ in range(n):
        jax.block_until_ready(f1(spike))
    r1 = time.time() - t0
    print(f"n_pre: {n_pre}, n_post: {n_post}, conn_prob: {conn_prob}, spk_prob: {spk_prob}, Linear: {r1} s")

    t0 = time.time()
    for _ in range(n):
        jax.block_until_ready(f2(spike))
    r1 = time.time() - t0
    print(f"n_pre: {n_pre}, n_post: {n_post}, conn_prob: {conn_prob}, spk_prob: {spk_prob}, Matmul: {r1} s")

    t0 = time.time()
    for _ in range(n):
        jax.block_until_ready(ti_linear(linear.indices, linear.weight.value, spike, n_post=n_post))
    r1 = time.time() - t0
    print(f"n_pre: {n_pre}, n_post: {n_post}, conn_prob: {conn_prob}, spk_prob: {spk_prob}, Taichi: {r1} s")

    print()
    bst.util.clear_buffer_memory()


def benchmark_forward():
    for n_pre, n_post in [
        # (1000, 1000),
        # (1000, 10000),
        (10000, 10000),
        (10000, 1000),
        (10000, 20000),
        (40000, 10000),
        (40000, 20000),
        (20000, 50000),
        (50000, 20000),
    ]:
        forward(n_pre, n_post, 0.01, 0.01, False)


if __name__ == '__main__':
    pass
    # forward(1000, 6400, 0.01, 0.01, False)
    # forward(10000, 12800, 0.01, 0.01, False)

    benchmark_forward()
