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

import jax.numpy as jnp

import brainstate


class TestMultiStepLR(unittest.TestCase):
    def test1(self):
        lr = brainstate.optim.MultiStepLR(0.1, [10, 20, 30], gamma=0.1)
        for i in range(40):
            r = lr(i)
            if i < 10:
                self.assertEqual(r, 0.1)
            elif i < 20:
                self.assertTrue(jnp.allclose(r, 0.01))
            elif i < 30:
                self.assertTrue(jnp.allclose(r, 0.001))
            else:
                self.assertTrue(jnp.allclose(r, 0.0001))

    def test2(self):
        lr = brainstate.compile.jit(brainstate.optim.MultiStepLR(0.1, [10, 20, 30], gamma=0.1))
        for i in range(40):
            r = lr(i)
            if i < 10:
                self.assertEqual(r, 0.1)
            elif i < 20:
                self.assertTrue(jnp.allclose(r, 0.01))
            elif i < 30:
                self.assertTrue(jnp.allclose(r, 0.001))
            else:
                self.assertTrue(jnp.allclose(r, 0.0001))
