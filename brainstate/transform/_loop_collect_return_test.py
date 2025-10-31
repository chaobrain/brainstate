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

import unittest

import jax.numpy as jnp
import numpy as np

import brainstate


class TestForLoop(unittest.TestCase):
    def test_for_loop(self):
        a = brainstate.ShortTermState(0.)
        b = brainstate.ShortTermState(0.)

        def f(i):
            a.value += (1 + b.value)
            return a.value

        n_iter = 10
        ops = np.arange(n_iter)
        r = brainstate.transform.for_loop(f, ops)

        print(a)
        print(b)
        self.assertTrue(a.value == n_iter)
        self.assertTrue(jnp.allclose(r, ops + 1))

    def test_checkpointed_for_loop(self):
        a = brainstate.ShortTermState(0.)
        b = brainstate.ShortTermState(0.)

        def f(i):
            a.value += (1 + b.value)
            return a.value

        n_iter = 18
        ops = jnp.arange(n_iter)
        r = brainstate.transform.checkpointed_for_loop(f, ops, base=2, pbar=brainstate.transform.ProgressBar())

        print(a)
        print(b)
        print(r)
        self.assertTrue(a.value == n_iter)
        self.assertTrue(jnp.allclose(r, ops + 1))
