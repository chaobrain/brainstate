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


class TestStateAvalStaleness(unittest.TestCase):
    """Out-of-band state shape/dtype changes must trigger recompilation,
    never silent reuse of a stale jaxpr (audit BUG 1)."""

    def test_state_shape_change_recompiles(self):
        st = bst.State(jnp.ones(3))

        @bst.transform.jit
        def mean_fn(x):
            v = st.value
            # v.shape[0] is baked into the jaxpr at trace time: a stale
            # jaxpr reused after reshaping the state gives a wrong mean
            return v.sum() / v.shape[0] + x * 0.0

        self.assertEqual(float(mean_fn(0.0)), 1.0)
        st.value = jnp.ones(6)
        self.assertEqual(float(mean_fn(0.0)), 1.0)

    def test_state_dtype_change_recompiles(self):
        st = bst.State(jnp.ones(4, dtype=jnp.float32))

        @bst.transform.jit
        def f():
            v = st.value
            if jnp.issubdtype(v.dtype, jnp.floating):
                return v / 2
            return v * 2

        self.assertTrue(bool(jnp.allclose(f(), jnp.full(4, 0.5))))
        st.value = jnp.ones(4, dtype=jnp.int32)
        out = f()
        self.assertEqual(out.dtype, jnp.int32)
        self.assertTrue(bool(jnp.all(out == 2)))

    def test_no_spurious_recompile_when_unchanged(self):
        st = bst.State(jnp.ones(3))

        @bst.transform.jit
        def f(x):
            st.value = st.value + x
            return st.value

        f(jnp.ones(3))
        key = f.stateful_fun.get_arg_cache_key(jnp.ones(3))
        j1 = f.stateful_fun.get_jaxpr_by_cache(key)
        f(jnp.ones(3))
        j2 = f.stateful_fun.get_jaxpr_by_cache(key)
        self.assertIs(j1, j2)

    def test_random_state_no_spurious_recompile(self):
        rng = bst.random.RandomState(42)

        @bst.transform.jit
        def f():
            return rng.rand(3)

        a = f()
        key = f.stateful_fun.get_arg_cache_key()
        j1 = f.stateful_fun.get_jaxpr_by_cache(key)
        b = f()
        j2 = f.stateful_fun.get_jaxpr_by_cache(key)
        self.assertIs(j1, j2)
        self.assertFalse(bool(jnp.allclose(a, b)))


class TestAotPathsDoNotWriteStates(unittest.TestCase):
    """AOT inspection (eval_shape/lower) must not mark states as written
    in an enclosing trace (audit M3)."""

    def _written_in_outer(self, aot_call):
        st = bst.State(jnp.zeros(3))

        @bst.transform.jit
        def inner(x):
            st.value = st.value + x
            return st.value

        # pre-compile at top level: when the AOT path later runs inside the
        # outer trace it hits the compilation cache, so the only thing that
        # could mark the state written there is a (spurious) writeback
        aot_call(inner, jnp.zeros(3))

        def outer(x):
            aot_call(inner, x)
            return x * 1.0

        sf = bst.transform.StatefulFunction(outer)
        sf.make_jaxpr(jnp.zeros(3))
        trace = sf.get_state_trace(jnp.zeros(3))
        for s, written in zip(trace.states, trace.been_writen):
            if s is st:
                return written
        return None

    def test_eval_shape_does_not_mark_states_written(self):
        self.assertFalse(self._written_in_outer(lambda jf, x: jf.eval_shape(x)))

    def test_lower_does_not_mark_states_written(self):
        self.assertFalse(self._written_in_outer(lambda jf, x: jf.lower(x)))
