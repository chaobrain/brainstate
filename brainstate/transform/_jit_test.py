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

from __future__ import annotations

import unittest

import jax.numpy as jnp

import brainstate as bst


class TestJIT(unittest.TestCase):
    def test_inner_state_are_not_catched(self):
        a = bst.State(bst.random.randn(10))

        @bst.compile.jit
        def fun1(inp):
            a.value += inp

            b = bst.State(bst.random.randn(1))

            def inner_fun(x):
                b.value += x

            bst.compile.for_loop(inner_fun, bst.random.randn(100))

            return a.value + b.value

        print(fun1(1.))
        key = fun1.stateful_fun.get_arg_cache_key(1.)
        self.assertTrue(len(fun1.stateful_fun.get_states_by_cache(key)) == 2)

        x = bst.random.randn(10)
        print(fun1(x))
        key = fun1.stateful_fun.get_arg_cache_key(x)
        self.assertTrue(len(fun1.stateful_fun.get_states_by_cache(key)) == 2)

    def test_kwargs(self):
        a = bst.State(bst.random.randn(10))

        @bst.compile.jit
        def fun1(inp):
            a.value += inp

            b = bst.State(bst.random.randn(1))

            def inner_fun(x):
                b.value += x

            bst.compile.for_loop(inner_fun, bst.random.randn(100))

            return a.value + b.value

        # test kwargs
        print(fun1(inp=bst.random.randn(10)))

    def test_jit_compile_sensitive_to_input_shape(self):
        global_data = [0]

        @bst.compile.jit
        def fun1(inp):
            global_data[0] += 1
            return inp

        print(fun1(1.))
        self.assertTrue(global_data[0] == 1)

        print(fun1(2.))
        self.assertTrue(global_data[0] == 1)

        print(fun1(bst.random.randn(10)))
        self.assertTrue(global_data[0] == 2)

        print(fun1(bst.random.randn(10, 10)))
        self.assertTrue(global_data[0] == 3)

    def test_jit_clear_cache(self):
        a = bst.State(bst.random.randn(1))
        compiling = []

        @bst.compile.jit
        def log2(x):
            print('compiling')
            compiling.append(1)
            ln_x = jnp.log(x)
            ln_2 = jnp.log(2.0) + a.value
            return ln_x / ln_2

        x = bst.random.randn(1)
        print(log2(x))  # compiling
        self.assertTrue(len(compiling) == 1)
        print(log2(x))  # no compiling
        self.assertTrue(len(compiling) == 1)

        log2.clear_cache()
        print(log2(x))  # compiling
        self.assertTrue(len(compiling) == 2)

    def test_jit_attribute_origin_fun(self):
        def fun1(x):
            return x

        jitted_fun = bst.compile.jit(fun1)
        self.assertTrue(jitted_fun.origin_fun is fun1)
        self.assertTrue(isinstance(jitted_fun.stateful_fun, bst.compile.StatefulFunction))
        self.assertTrue(callable(jitted_fun.jitted_fun))
        self.assertTrue(callable(jitted_fun.clear_cache))

    def test_clear_cache(self):
        a = bst.State(bst.random.randn(1))

        @bst.compile.jit
        def f_jit(x, y):
            print('Compiling')
            a.value = jnp.sin(x) + jnp.cos(y)

        f_jit(0.5, 1.0)
        f_jit.clear_cache()
        f_jit(0.5, 1.0)

    def test_cache(self):
        @bst.compile.jit
        @bst.compile.jit
        @bst.compile.jit
        def f(a):
            print('Compiling')
            print(a)
            return a + 1

        print(f(1.))
