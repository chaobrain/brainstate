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

import jax.numpy
import jax.numpy as jnp
import numpy as np
from absl.testing import parameterized

import brainstate as bst


def _get_csr(n_pre, n_post, prob):
    n_conn = int(n_post * prob)
    indptr = np.arange(n_pre + 1) * n_conn
    indices = np.random.randint(0, n_post, (n_pre * n_conn,))
    return indptr, indices


def true_fn(x, w, indices, indptr, n_out):
    homo_w = jnp.size(w) == 1

    post = jnp.zeros((n_out,))
    for i_pre in range(x.shape[0]):
        ids = indices[indptr[i_pre]: indptr[i_pre + 1]]
        post = post.at[ids].add(w * x[i_pre] if homo_w else w[indptr[i_pre]: indptr[i_pre + 1]] * x[i_pre])
    return post


# class TestFixedProbCSR(parameterized.TestCase):
#     @parameterized.product(
#         homo_w=[True, False],
#     )
#     def test1(self, homo_w):
#         x = bst.random.rand(20) < 0.1
#         indptr, indices = _get_csr(20, 40, 0.1)
#         m = bst.event.CSRLinear(20, 40, indptr, indices, 1.5 if homo_w else bst.init.Normal())
#         y = m(x)
#         y2 = true_fn(x, m.weight.value, indices, indptr, 40)
#         self.assertTrue(jnp.allclose(y, y2))
#
#     @parameterized.product(
#         bool_x=[True, False],
#         homo_w=[True, False]
#     )
#     def test_vjp(self, bool_x, homo_w):
#         n_in = 20
#         n_out = 30
#         if bool_x:
#             x = jax.numpy.asarray(bst.random.rand(n_in) < 0.3, dtype=float)
#         else:
#             x = bst.random.rand(n_in)
#
#         indptr, indices = _get_csr(n_in, n_out, 0.1)
#         fn = bst.event.CSRLinear(n_in, n_out, indptr, indices, 1.5 if homo_w else bst.init.Normal())
#         w = fn.weight.value
#
#         def f(x, w):
#             fn.weight.value = w
#             return fn(x).sum()
#
#         r = jax.grad(f, argnums=(0, 1))(x, w)
#
#         # -------------------
#         # TRUE gradients
#
#         def f2(x, w):
#             return true_fn(x, w, indices, indptr, n_out).sum()
#
#         r2 = jax.grad(f2, argnums=(0, 1))(x, w)
#         self.assertTrue(jnp.allclose(r[0], r2[0]))
#         self.assertTrue(jnp.allclose(r[1], r2[1]))
#
#     @parameterized.product(
#         bool_x=[True, False],
#         homo_w=[True, False]
#     )
#     def test_jvp(self, bool_x, homo_w):
#         n_in = 20
#         n_out = 30
#         if bool_x:
#             x = jax.numpy.asarray(bst.random.rand(n_in) < 0.3, dtype=float)
#         else:
#             x = bst.random.rand(n_in)
#
#         indptr, indices = _get_csr(n_in, n_out, 0.1)
#         fn = bst.event.CSRLinear(n_in, n_out, indptr, indices,
#                                  1.5 if homo_w else bst.init.Normal(), grad_mode='jvp')
#         w = fn.weight.value
#
#         def f(x, w):
#             fn.weight.value = w
#             return fn(x)
#
#         o1, r1 = jax.jvp(f, (x, w), (jnp.ones_like(x), jnp.ones_like(w)))
#
#         # -------------------
#         # TRUE gradients
#
#         def f2(x, w):
#             return true_fn(x, w, indices, indptr, n_out)
#
#         o2, r2 = jax.jvp(f2, (x, w), (jnp.ones_like(x), jnp.ones_like(w)))
#         self.assertTrue(jnp.allclose(r1, r2))
#         self.assertTrue(jnp.allclose(o1, o2))
