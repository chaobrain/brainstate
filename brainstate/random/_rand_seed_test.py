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




import platform
import unittest

import jax.numpy as jnp
import jax.random
import jax.random as jr
import numpy as np
import pytest

import brainstate as bst


class TestRandom(unittest.TestCase):

  def test_seed2(self):
    test_seed = 299
    key = jax.random.PRNGKey(test_seed)
    bst.random.seed(key)

    @jax.jit
    def jit_seed(key):
      bst.random.seed(key)
      with bst.random.seed_context(key):
        print(bst.random.DEFAULT.value)

    jit_seed(key)
    jit_seed(1)
    jit_seed(None)
    bst.random.seed(1)

  def test_seed(self):
    test_seed = 299
    bst.random.seed(test_seed)
    a = bst.random.rand(3)
    bst.random.seed(test_seed)
    b = bst.random.rand(3)
    self.assertTrue(jnp.array_equal(a, b))

