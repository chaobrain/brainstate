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
import jax
import jax.numpy as jnp
from absl.testing import parameterized

import brainstate as bst
from brainstate.event._linear import Linear


class TestEventLinear(parameterized.TestCase):
  @parameterized.product(
    homo_w=[True, False],
    bool_x=[True, False],
  )
  def test1(self, homo_w, bool_x):
    x = bst.random.rand(20) < 0.1
    if not bool_x:
      x = jnp.asarray(x, dtype=float)
    m = Linear(20, 40, 1.5 if homo_w else bst.init.KaimingUniform())
    y = m(x)
    print(y)

    self.assertTrue(jnp.allclose(y, (x.sum() * m.weight.value) if homo_w else (x @ m.weight.value)))

  def test_grad_bool(self):
    n_in = 20
    n_out = 30
    x = bst.random.rand(n_in) < 0.3
    fn = Linear(n_in, n_out, bst.init.KaimingUniform())

    with self.assertRaises(TypeError):
      print(jax.grad(lambda x: fn(x).sum())(x))

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

    fn = Linear(n_in, n_out, 1.5 if homo_w else bst.init.KaimingUniform())
    w = fn.weight.value

    def f(x, w):
      fn.weight.value = w
      return fn(x).sum()

    r1 = jax.grad(f, argnums=(0, 1))(x, w)

    # -------------------
    # TRUE gradients

    def f2(x, w):
      y = (x @ (jnp.ones([n_in, n_out]) * w)) if homo_w else (x @ w)
      return y.sum()

    r2 = jax.grad(f2, argnums=(0, 1))(x, w)
    self.assertTrue(jnp.allclose(r1[0], r2[0]))
    self.assertTrue(jnp.allclose(r1[1], r2[1]))

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

    fn = Linear(n_in, n_out, 1.5 if homo_w else bst.init.KaimingUniform(), grad_mode='jvp')
    w = fn.weight.value

    def f(x, w):
      fn.weight.value = w
      return fn(x)

    o1, r1 = jax.jvp(f, (x, w), (jnp.ones_like(x), jnp.ones_like(w)))

    # -------------------
    # TRUE gradients

    def f2(x, w):
      y = (x @ (jnp.ones([n_in, n_out]) * w)) if homo_w else (x @ w)
      return y

    o2, r2 = jax.jvp(f, (x, w), (jnp.ones_like(x), jnp.ones_like(w)))
    self.assertTrue(jnp.allclose(o1, o2))
    self.assertTrue(jnp.allclose(r1, r2))
