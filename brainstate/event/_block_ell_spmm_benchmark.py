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

import functools
import timeit

import jax
import jax.numpy as jnp

import brainstate as bst
from brainstate.event._block_ell_spmm import sample_sparse_matrix, sdd_matmul, native_sdd_matmul


def main(dtype=jnp.float16, sparse_prob=0.2):
    bst.random.seed(1234)
    # data
    m, k, n = 4096, 4096, 4096
    bm, bn, bk = 32, 32, 256
    print(f"Matrix Shape: {m} x {k} x {n}, dtype: {dtype}, sparse_prob: {sparse_prob}")

    x = sample_sparse_matrix(m, k, bm, bn, sparse_prob=sparse_prob, dtype=dtype)
    x_dense = x.todense()
    y = bst.random.randn(k, n, dtype=dtype)

    # operations
    interpret = jax.devices()[0].platform == "cpu"
    # sdd_matmul(x, y, debug=False, block_size=bk, interpret=interpret).block_until_ready()
    native_matmul = jax.jit(native_sdd_matmul)
    pl_matmul = jax.jit(functools.partial(sdd_matmul, block_size=bk, interpret=interpret))
    dense_matmul = jax.jit(jnp.matmul)
    native_grad = jax.jit(jax.grad(native_sdd_matmul, argnums=(0, 1)))
    pl_grad = jax.jit(jax.grad(functools.partial(sdd_matmul, block_size=bk, interpret=interpret), argnums=(0, 1)))
    dense_grad = jax.jit(jax.grad(jnp.matmul, argnums=(0, 1)))

    # compilation
    out_pl = pl_matmul(x, y)
    out_hlo = native_matmul(x, y)
    out_ref = dense_matmul(x_dense, y)
    # out_pl = pl_grad(x, y)
    # out_hlo = native_grad(x, y)
    # out_ref = dense_grad(x_dense, y)

    # print(jnp.max(jnp.abs(out_pl - out_ref)))
    # print(jnp.max(jnp.abs(out_pl - out_ref) / jnp.abs(out_pl)))
    # np.testing.assert_allclose(out_pl, out_ref, atol=0.04, rtol=0.04)
    # np.testing.assert_allclose(out_hlo, out_ref, atol=0.04, rtol=0.04)

    n_trial1, n_trial2 = (10, 2) if interpret else (1000, 20)
    duration = timeit.timeit(lambda: dense_matmul(x_dense, y).block_until_ready(), number=n_trial1)
    s1_forward = duration / n_trial1 * 1000
    print(f"Dense Matmul, forward: {s1_forward:.2f}ms")
    # duration = timeit.timeit(lambda: jax.block_until_ready(dense_grad(x, y)), number=n_trial1)
    # s1_backward = duration / n_trial1 * 1000
    # print(f"Dense Matmul, backward: {s1_backward:.2f}ms")

    duration = timeit.timeit(lambda: pl_matmul(x, y).block_until_ready(), number=n_trial1)
    s2_forward = duration / n_trial1 * 1000
    print(f"Pallas Blocksparse Matmul, forward: {s2_forward:.2f}ms")
    # duration = timeit.timeit(lambda: jax.block_until_ready(pl_grad(x, y)), number=n_trial1)
    # s2_backward = duration / n_trial1 * 1000
    # print(f"Pallas Blocksparse Matmul, backward: {s2_backward:.2f}ms")

    duration = timeit.timeit(lambda: native_matmul(x, y).block_until_ready(), number=n_trial2)
    s3_forward = duration / n_trial2 * 1000
    print(f"HLO Blocksparse Matmul, forward: {s3_forward:.2f}ms")
    # duration = timeit.timeit(lambda: jax.block_until_ready(native_grad(x, y)), number=n_trial2)
    # s3_backward = duration / n_trial2 * 1000
    # print(f"HLO Blocksparse Matmul, backward: {s3_backward:.2f}ms")

    print(f"Forward speedup: {s1_forward / s2_forward:.2f}x (Dense vs. Pallas), "
          f"{s3_forward / s2_forward:.2f}x (HLO vs. Pallas)")
    # print(f"Backward speedup: {s1_backward / s2_backward:.2f}x (Dense vs. Pallas), "
    #       f"{s3_backward / s2_backward:.2f}x (HLO vs. Pallas)")
    print()


if __name__ == "__main__":
    main(jnp.float32, 0.3)
    main(jnp.float32, 0.2)
    main(jnp.float32, 0.1)
    main(jnp.float32, 0.05)
    main(jnp.float16, 0.3)
    main(jnp.float16, 0.2)
    main(jnp.float16, 0.1)
    main(jnp.float16, 0.05)
