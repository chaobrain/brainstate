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

import jax
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


class TestJitNameAndStaging(unittest.TestCase):
    """Named scopes, the disable-jit fast path, and AOT staging methods."""

    def test_named_jit_runs(self):
        """A ``name`` is attached and the function still computes correctly."""
        st = bst.State(jnp.zeros((2,)))

        @bst.transform.jit(name='stepper')
        def step(x):
            st.value = st.value + x
            return st.value

        out = step(jnp.ones((2,)))
        self.assertTrue(bool(jnp.allclose(out, 1.0)))
        self.assertTrue(bool(jnp.allclose(st.value, 1.0)))

    def test_static_argnums_recompile(self):
        """Different static-arg values recompile; the result tracks the arg."""

        def f(x, n):
            return x * n

        jf = bst.transform.jit(f, static_argnums=(1,))
        self.assertTrue(bool(jnp.allclose(jf(jnp.ones((3,)), 2), 2.0)))
        self.assertTrue(bool(jnp.allclose(jf(jnp.ones((3,)), 3), 3.0)))

    def test_keyword_arguments(self):
        """``jit`` forwards keyword arguments."""
        jf = bst.transform.jit(lambda x, *, scale: x * scale)
        self.assertTrue(bool(jnp.allclose(jf(jnp.ones((2,)), scale=4.0), 4.0)))

    def test_disable_jit_calls_original(self):
        """Under ``disable_jit`` the original function runs eagerly."""
        st = bst.State(jnp.zeros((2,)))

        @bst.transform.jit
        def step(x):
            st.value = st.value + x
            return st.value

        with jax.disable_jit():
            out = step(jnp.ones((2,)))
        self.assertTrue(bool(jnp.allclose(out, 1.0)))

    def test_lower_trace_compile_eval_shape(self):
        """The ahead-of-time staging helpers run without error."""
        jf = bst.transform.jit(lambda x: x * 2)
        x = jnp.ones((3,))
        self.assertIsNotNone(jf.lower(x))
        self.assertIsNotNone(jf.trace(x))
        self.assertIsNotNone(jf.compile(x))
        shape = jf.eval_shape(x)
        self.assertEqual(jax.tree.leaves(shape)[0].shape, (3,))

    def test_clear_cache_method(self):
        """``clear_cache`` resets caches and the function recompiles cleanly."""
        jf = bst.transform.jit(lambda x: x + 1)
        jf(jnp.ones((2,)))
        jf.clear_cache()
        self.assertTrue(bool(jnp.allclose(jf(jnp.ones((2,))), 2.0)))


class TestJitShardings(unittest.TestCase):
    """``in_shardings``/``out_shardings`` must account for the hidden
    state-values argument that ``jit`` prepends (audit M4), and negative
    static/donate argnums must be rejected instead of silently shifting
    onto the state tuple."""

    def _sharding(self, spec):
        import numpy as np
        from jax.sharding import Mesh, NamedSharding
        mesh = Mesh(np.array(jax.devices()[:1]), ('x',))
        return NamedSharding(mesh, spec)

    def test_tuple_in_shardings_aligned(self):
        from jax.sharding import PartitionSpec as P
        st = bst.State(jnp.zeros(4))

        @bst.transform.jit(in_shardings=(self._sharding(P('x')),))
        def f(x):
            st.value = st.value + x
            return x * 2.0

        out = f(jnp.arange(4.0))
        self.assertTrue(bool(jnp.allclose(out, jnp.arange(4.0) * 2.0)))
        self.assertTrue(bool(jnp.allclose(st.value, jnp.arange(4.0))))

    def test_out_shardings_aligned(self):
        from jax.sharding import PartitionSpec as P
        st = bst.State(jnp.asarray(0.0))

        @bst.transform.jit(out_shardings=self._sharding(P('x')))
        def g(x):
            st.value = st.value + 1.0  # scalar state write: P('x') cannot apply
            return x * 2.0

        out = g(jnp.arange(4.0))
        self.assertTrue(bool(jnp.allclose(out, jnp.arange(4.0) * 2.0)))
        self.assertEqual(float(st.value), 1.0)

    def test_single_in_sharding_rejected(self):
        from jax.sharding import PartitionSpec as P
        with self.assertRaises(NotImplementedError):
            bst.transform.jit(lambda x: x, in_shardings=self._sharding(P()))

    def test_negative_donate_argnums_rejected(self):
        with self.assertRaises(ValueError):
            bst.transform.jit(lambda x: x, donate_argnums=-1)

    def test_negative_static_argnums_rejected(self):
        with self.assertRaises(ValueError):
            bst.transform.jit(lambda x, n: x * n, static_argnums=-1)
