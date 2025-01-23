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

os.environ['JAX_TRACEBACK_FILTERING'] = 'off'

import jax

import time
import brainstate as bst


# bst.environ.set(platform='cpu')


def forward(n_pre, n_post, spk_prob, as_float: bool):
    linear = bst.event.Linear(n_pre, n_post, weight=bst.init.KaimingUniform(), block_size=256)
    spike = (bst.random.rand(n_pre) < spk_prob)

    if as_float:
        spike = spike.astype(float)

    @jax.jit
    def f1(spike):
        return linear(spike)

    @jax.jit
    def f2(spike):
        return spike @ linear.weight.value

    y1 = jax.block_until_ready(f1(spike))
    y2 = jax.block_until_ready(f2(spike))
    print('max difference:', jax.numpy.abs(y1 - y2).max())

    n = 100
    t0 = time.time()
    for _ in range(n):
        jax.block_until_ready(f1(spike))
    r1 = time.time() - t0
    print(f"n_pre: {n_pre}, n_post: {n_post}, spike probability: {spk_prob}, Linear: {r1} s")

    t0 = time.time()
    for _ in range(n):
        jax.block_until_ready(f2(spike))
    r2 = time.time() - t0
    print(f"n_pre: {n_pre}, n_post: {n_post}, spike probability: {spk_prob}, Matmul: {r2} s")
    print('Acceleration ratio:', r2 / r1 - 1.)

    print()


def benchmark_forward():
    for n_pre, n_post in [
        (1000, 1000),
        (1000, 10000),
        (10000, 10000),
        (10000, 1000),
        (20000, 10000),
        (20000, 20000),
        # (10000, 100000),
    ]:
        forward(n_pre, n_post, 0.01, True)
        forward(n_pre, n_post, 0.1, True)
        print()
        print()


if __name__ == '__main__':
    # forward(1000, 2000, 0.01, True)
    # forward(2000, 4000, 0.01, True)
    # forward(10000, 20000, 0.01, True)
    benchmark_forward()
