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
import unittest

import jax
import optax

import brainstate as bst


class TestOptaxOptimizer(unittest.TestCase):
  def test1(self):
    class Model(bst.nn.Module):
      def __init__(self):
        super().__init__()
        self.linear1 = bst.nn.Linear(2, 3)
        self.linear2 = bst.nn.Linear(3, 4)

      def __call__(self, x):
        return self.linear2(self.linear1(x))

    x = bst.random.randn(1, 2)
    y = jax.numpy.ones((1, 4))

    model = Model()
    tx = optax.adam(1e-3)
    optimizer = bst.optim.OptaxOptimizer(model.states(bst.ParamState), tx)

    loss_fn = lambda: ((model(x) - y) ** 2).mean()
    prev_loss = loss_fn()

    grads = bst.augment.grad(loss_fn, model.states(bst.ParamState))()
    optimizer.update(grads)

    new_loss = loss_fn()

    print(new_loss, prev_loss)
    self.assertLess(new_loss, prev_loss)
