# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
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
from absl.testing import absltest

import brainstate


class TestRemat(absltest.TestCase):
    def test_basic_remat(self):
        module = brainstate.transform.remat(brainstate.nn.Linear(2, 3))
        y = module(jnp.ones((1, 2)))
        assert y.shape == (1, 3)

    def test_remat_with_scan(self):
        class ScanLinear(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = brainstate.nn.Linear(3, 3)

            def __call__(self, x: jax.Array):
                @brainstate.transform.remat
                def fun(x: jax.Array, _):
                    x = self.linear(x)
                    return x, None

                return brainstate.transform.scan(fun, x, None, length=10)[0]

        m = ScanLinear()

        assert m.linear.weight.value['weight'].shape == (3, 3)
        assert m.linear.weight.value['bias'].shape == (3,)

        y = m(jnp.ones((10, 3)))
        assert y.shape == (10, 3)
