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

import jax
import jax.numpy as jnp

import brainstate


class TestJitError(unittest.TestCase):
    def test1(self):
        with self.assertRaises(Exception):
            brainstate.transform.jit_error_if(True, 'error')

        def err_f(x):
            raise ValueError(f'error: {x}')

        brainstate.transform.jit_error_if(False, err_f, 1.)
        with self.assertRaises(Exception):
            brainstate.transform.jit_error_if(True, err_f, 1.)

    def test_vmap(self):
        def f(x):
            brainstate.transform.jit_error_if(x, 'error: {x}', x=x)

        jax.vmap(f)(jnp.array([False, False, False]))
        with self.assertRaises(Exception):
            jax.vmap(f)(jnp.array([True, False, False]))

    def test_vmap_vmap(self):
        def f(x):
            brainstate.transform.jit_error_if(x, 'error: {x}', x=x)

        jax.vmap(jax.vmap(f))(jnp.array([[False, False, False],
                                         [False, False, False]]))
        with self.assertRaises(Exception):
            jax.vmap(jax.vmap(f))(jnp.array([[False, False, False],
                                             [True, False, False]]))
