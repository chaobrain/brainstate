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

import jax.numpy as jnp
import pytest
from absl.testing import absltest
from absl.testing import parameterized

import brainstate as bst





class TestDense(parameterized.TestCase):
    @parameterized.product(
        size=[(10,),
              (20, 10),
              (5, 8, 10)],
        num_out=[20, ]
    )
    def test_Dense1(self, size, num_out):
        f = bst.nn.Linear(10, num_out)
        x = bst.random.random(size)
        y = f(x)
        self.assertTrue(y.shape == size[:-1] + (num_out,))

