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
from absl.testing import parameterized

import brainstate as bst
from brainstate.event._fixed_probability import FixedProb


class TestFixedProbCSR(parameterized.TestCase):
  @parameterized.product(
    allow_multi_conn=[True, False]
  )
  def test1(self, allow_multi_conn):
    x = bst.random.rand(20) < 0.1
    # x = bst.random.rand(20)
    m = FixedProb(20, 40, 0.1, 1.0, seed=123, allow_multi_conn=allow_multi_conn)
    y = m(x)
    print(y)

    m2 = FixedProb(20, 40, 0.1, bst.init.KaimingUniform(), seed=123)
    print(m2(x))

  def test_grad_bool(self):
    n_in = 20
    n_out = 30
    x = bst.random.rand(n_in) < 0.3
    fn = FixedProb(n_in, n_out, 0.1, bst.init.KaimingUniform(), seed=123)

    def f(x):
      return fn(x).sum()

    with self.assertRaises(TypeError):
      print(jax.grad(f)(x))

  @parameterized.product(
    bool_x=[True, False],
    homo_w=[True, False]
  )
  def test_vjp(self, bool_x, homo_w):
    n_in = 20
    n_out = 30
    if bool_x:
      x = jax.numpy.asarray(bst.random.rand(n_in) < 0.3, dtype=float)
    else:
      x = bst.random.rand(n_in)

    if homo_w:
      fn = FixedProb(n_in, n_out, 0.1, 1.5, seed=123)
    else:
      fn = FixedProb(n_in, n_out, 0.1, bst.init.KaimingUniform(), seed=123)
    w = fn.weight.value

    def f(x, w):
      fn.weight.value = w
      return fn(x).sum()

    r = bst.transform.grad(f, argnums=(0, 1))(x, w)

    # -------------------
    # TRUE gradients

    def true_fn(x, w, indices, n_post):
      post = jnp.zeros((n_post,))
      for i in range(n_in):
        post = post.at[indices[i]].add(w * x[i] if homo_w else w[i] * x[i])
      return post

    def f2(x, w):
      return true_fn(x, w, fn.indices, n_out).sum()

    r2 = jax.grad(f2, argnums=(0, 1))(x, w)
    self.assertTrue(jnp.allclose(r[0], r2[0]))
    self.assertTrue(jnp.allclose(r[1], r2[1]))
    print(r[1])

  @parameterized.product(
    bool_x=[True, False],
    homo_w=[True, False]
  )
  def test_jvp(self, bool_x, homo_w):
    n_in = 20
    n_out = 30
    if bool_x:
      x = jax.numpy.asarray(bst.random.rand(n_in) < 0.3, dtype=float)
    else:
      x = bst.random.rand(n_in)

    fn = FixedProb(n_in, n_out, 0.1, 1.5 if homo_w else bst.init.KaimingUniform(), seed=123, grad_mode='jvp')
    w = fn.weight.value

    def f(x, w):
      fn.weight.value = w
      return fn(x)

    o1, r1 = jax.jvp(f, (x, w), (jnp.ones_like(x), jnp.ones_like(w)))

    # -------------------
    # TRUE gradients

    def true_fn(x, w, indices, n_post):
      post = jnp.zeros((n_post,))
      for i in range(n_in):
        post = post.at[indices[i]].add(w * x[i] if homo_w else w[i] * x[i])
      return post

    def f2(x, w):
      return true_fn(x, w, fn.indices, n_out)

    o2, r2 = jax.jvp(f2, (x, w), (jnp.ones_like(x), jnp.ones_like(w)))
    self.assertTrue(jnp.allclose(r1, r2))
    self.assertTrue(jnp.allclose(o1, o2))
