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


import unittest

import jax.numpy as jnp

import brainstate as bst


class TestEnviron(unittest.TestCase):
    def test_precision(self):
        with bst.environ.context(precision=64):
            a = bst.random.randn(1)
            self.assertEqual(a.dtype, jnp.float64)

        with bst.environ.context(precision=32):
            a = bst.random.randn(1)
            self.assertEqual(a.dtype, jnp.float32)

        with bst.environ.context(precision=16):
            a = bst.random.randn(1)
            self.assertEqual(a.dtype, jnp.float16)

        with bst.environ.context(precision='bf16'):
            a = bst.random.randn(1)
            self.assertEqual(a.dtype, jnp.bfloat16)

    def test_platform(self):
        with self.assertRaises(ValueError):
            with bst.environ.context(platform='cpu'):
                a = bst.random.randn(1)
                self.assertEqual(a.device(), 'cpu')

    def test_register_default_behavior(self):
        bst.environ.set(dt=0.1)

        dt_ = 0.1

        def dt_behavior(dt):
            nonlocal dt_
            dt_ = dt
            print(f'dt: {dt}')

        bst.environ.register_default_behavior('dt', dt_behavior)

        with bst.environ.context(dt=0.2):
            self.assertEqual(dt_, 0.2)
        self.assertEqual(dt_, 0.1)
