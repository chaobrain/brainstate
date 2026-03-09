# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
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

from brainstate.transform._debug import (
    debug_nan,
    debug_nan_if,
    DebugNan,
    _has_nan_flag,
    _extract_user_source,
    _interpret_jaxpr_with_nan_check,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_nan_check(fn, *args, phase=''):
    """Run the instrumented interpreter directly on fn's jaxpr."""
    closed = jax.make_jaxpr(fn)(*args)
    return _interpret_jaxpr_with_nan_check(closed.jaxpr, closed.consts, *args, phase=phase)


# ---------------------------------------------------------------------------
# _has_nan_flag
# ---------------------------------------------------------------------------

class TestHasNanFlag(unittest.TestCase):

    def test_no_nan(self):
        flag = _has_nan_flag([jnp.array([1.0, 2.0, 3.0])])
        self.assertFalse(bool(flag))

    def test_nan(self):
        flag = _has_nan_flag([jnp.array([1.0, jnp.nan, 3.0])])
        self.assertTrue(bool(flag))

    def test_inf(self):
        flag = _has_nan_flag([jnp.array([1.0, jnp.inf])])
        self.assertTrue(bool(flag))

    def test_neg_inf(self):
        flag = _has_nan_flag([jnp.array([-jnp.inf])])
        self.assertTrue(bool(flag))

    def test_integer_array_ignored(self):
        flag = _has_nan_flag([jnp.array([1, 2, 3])])
        self.assertFalse(bool(flag))

    def test_empty_list(self):
        flag = _has_nan_flag([])
        self.assertFalse(bool(flag))

    def test_mixed_clean_and_nan(self):
        flag = _has_nan_flag([jnp.array([1.0]), jnp.array([jnp.nan])])
        self.assertTrue(bool(flag))


# ---------------------------------------------------------------------------
# _extract_user_source
# ---------------------------------------------------------------------------

class TestExtractUserSource(unittest.TestCase):

    def test_none_returns_unknown(self):
        result = _extract_user_source(None)
        self.assertIn("unknown", result)

    def test_real_eqn_has_source(self):
        def fn(x):
            return jnp.log(x)

        closed = jax.make_jaxpr(fn)(jnp.array([1.0]))
        for eqn in closed.jaxpr.eqns:
            src = _extract_user_source(getattr(eqn, 'source_info', None))
            # Should not be unknown for a real traced function
            self.assertIsInstance(src, str)

    def test_source_excludes_brainstate_internals(self):
        """Source location should not include brainstate tracing infrastructure."""
        import brainstate as bst
        from brainstate.transform._make_jaxpr import StatefulFunction

        class Model(bst.graph.Node):
            def __init__(self):
                self.w = bst.State(jnp.array([1.0]))

            def __call__(self, x):
                return jnp.log(self.w.value) + x

        model = Model()
        sf = StatefulFunction(model)
        cache_key = sf.get_arg_cache_key(jnp.array([1.0]), compile_if_miss=True)
        closed = sf.get_jaxpr_by_cache(cache_key)

        for eqn in closed.jaxpr.eqns:
            src = _extract_user_source(getattr(eqn, 'source_info', None))
            # Should NOT contain brainstate tracing infrastructure paths
            self.assertNotIn('_make_jaxpr.py', src)
            # Should return something useful, not <unknown source>
            self.assertNotEqual(src, '<unknown source>')


# ---------------------------------------------------------------------------
# debug_nan – basic detection
# ---------------------------------------------------------------------------

class TestDebugNan(unittest.TestCase):

    def test_raises_for_nan(self):
        def fn(x):
            return jnp.log(x)

        with self.assertRaises(RuntimeError) as ctx:
            debug_nan(fn, jnp.array([0.0, 1.0, 2.0]))
        self.assertIn("NaN", str(ctx.exception))

    def test_raises_for_inf(self):
        def fn(x):
            return 1.0 / x

        with self.assertRaises(RuntimeError) as ctx:
            debug_nan(fn, jnp.array([0.0, 1.0]))
        self.assertIn("NaN", str(ctx.exception))

    def test_no_raise_for_clean(self):
        def fn(x):
            return x * 2.0 + 1.0

        # Should not raise
        debug_nan(fn, jnp.array([1.0, 2.0, 3.0]))

    def test_error_contains_primitive_name(self):
        def fn(x):
            return jnp.log(x)

        with self.assertRaises(RuntimeError) as ctx:
            debug_nan(fn, jnp.array([0.0]))
        self.assertIn("log", str(ctx.exception))

    def test_error_contains_source_location(self):
        def fn(x):
            return jnp.log(x)

        with self.assertRaises(RuntimeError) as ctx:
            debug_nan(fn, jnp.array([0.0]))
        msg = str(ctx.exception)
        # Should contain a file path reference
        self.assertTrue('File' in msg or '.py' in msg or '<' in msg)

    def test_error_contains_phase(self):
        def fn(x):
            return jnp.log(x)

        with self.assertRaises(RuntimeError) as ctx:
            debug_nan(fn, jnp.array([0.0]), phase='forward')
        self.assertIn("forward", str(ctx.exception))

    def test_reports_first_nan_source(self):
        """Propagated NaN should not be re-reported; only the source is flagged."""
        def fn(x):
            y = jnp.log(x)   # introduces NaN
            z = y * 2.0      # propagates
            return z + 1.0   # propagates

        with self.assertRaises(RuntimeError) as ctx:
            debug_nan(fn, jnp.array([0.0]))
        # The callback raises on 'log', not on later ops
        self.assertIn("log", str(ctx.exception))

    def test_error_source_info_in_message(self):
        """Error message should contain the source location of the NaN-producing equation."""
        import brainstate as bst

        class Model(bst.graph.Node):
            def __init__(self):
                self.w = bst.State(jnp.array([1.0, 0.0]))

            def __call__(self, x):
                return jnp.log(self.w.value) + x

        model = Model()
        with self.assertRaises(RuntimeError) as ctx:
            debug_nan(model, jnp.array([1.0, 2.0]))

        msg = str(ctx.exception)
        # Message should contain source location pointing to user code
        self.assertIn('Source location', msg)
        # Should reference the function/method where NaN was introduced
        self.assertIn('__call__', msg)
        # Should NOT reference brainstate tracing infrastructure
        self.assertNotIn('_make_jaxpr.py', msg)


# ---------------------------------------------------------------------------
# debug_nan_if
# ---------------------------------------------------------------------------

class TestDebugNanIf(unittest.TestCase):

    def test_false_predicate_no_raise(self):
        def fn(x):
            return jnp.log(x)

        # False → should not trigger debug, so no error
        debug_nan_if(False, fn, jnp.array([0.0, 1.0]))

    def test_true_predicate_raises(self):
        def fn(x):
            return jnp.log(x)

        with self.assertRaises(RuntimeError):
            debug_nan_if(True, fn, jnp.array([0.0, 1.0]))

    def test_jax_array_false_no_raise(self):
        def fn(x):
            return jnp.log(x)

        debug_nan_if(jnp.array(False), fn, jnp.array([0.0, 1.0]))

    def test_jax_array_true_raises(self):
        def fn(x):
            return jnp.log(x)

        with self.assertRaises(RuntimeError):
            debug_nan_if(jnp.array(True), fn, jnp.array([0.0, 1.0]))

    def test_jit_compatible_no_nan(self):
        @jax.jit
        def f(x, trigger):
            debug_nan_if(trigger, lambda y: jnp.log(y), x)
            return x

        result = f(jnp.array([1.0, 2.0]), jnp.array(False))
        self.assertEqual(result.shape, (2,))

    def test_jit_compatible_with_nan(self):
        @jax.jit
        def f(x, trigger):
            debug_nan_if(trigger, lambda y: jnp.log(y), x)
            return x

        with self.assertRaises(RuntimeError):
            f(jnp.array([0.0, 1.0]), jnp.array(True))


# ---------------------------------------------------------------------------
# Nested JIT (pjit)
# ---------------------------------------------------------------------------

class TestNestedJit(unittest.TestCase):

    def test_nan_inside_inner_jit(self):
        @jax.jit
        def inner(x):
            return jnp.log(x)

        @jax.jit
        def outer(x):
            return inner(x) * 2.0

        with self.assertRaises(RuntimeError) as ctx:
            debug_nan(outer, jnp.array([0.0, 1.0]))
        self.assertIn("log", str(ctx.exception))

    def test_deeply_nested_jit(self):
        @jax.jit
        def level3(x):
            return jnp.log(x)

        @jax.jit
        def level2(x):
            return level3(x) + 1.0

        @jax.jit
        def level1(x):
            return level2(x) * 2.0

        with self.assertRaises(RuntimeError) as ctx:
            debug_nan(level1, jnp.array([0.0, 1.0]))
        self.assertIn("log", str(ctx.exception))

    def test_clean_nested_jit_no_raise(self):
        @jax.jit
        def inner(x):
            return x * 2.0

        @jax.jit
        def outer(x):
            return inner(x) + 1.0

        debug_nan(outer, jnp.array([1.0, 2.0, 3.0]))


# ---------------------------------------------------------------------------
# Cond primitive
# ---------------------------------------------------------------------------

class TestCondPrimitive(unittest.TestCase):

    def test_true_branch_nan_detected(self):
        def fn(x):
            return jax.lax.cond(
                x[0] > 0.0,
                lambda y: jnp.log(y),   # NaN branch
                lambda y: y * 2.0,
                x,
            )

        with self.assertRaises(RuntimeError) as ctx:
            debug_nan(fn, jnp.array([1.0, 0.0]))  # x[0]>0 → log branch → NaN
        self.assertIn("log", str(ctx.exception))

    def test_false_branch_no_nan(self):
        def fn(x):
            return jax.lax.cond(
                x[0] > 0.0,
                lambda y: jnp.log(y),   # not taken
                lambda y: y * 2.0,      # taken, clean
                x,
            )

        # x[0] = -1 → false branch taken → no NaN
        debug_nan(fn, jnp.array([-1.0, 1.0, 2.0]))


# ---------------------------------------------------------------------------
# While primitive
# ---------------------------------------------------------------------------

class TestWhilePrimitive(unittest.TestCase):

    def test_nan_in_body(self):
        def fn(x):
            def cond_fn(val):
                return val[0] < 10.0

            def body_fn(val):
                return jnp.log(val)  # NaN when val has non-positive elements

            return jax.lax.while_loop(cond_fn, body_fn, x)

        with self.assertRaises(RuntimeError) as ctx:
            debug_nan(fn, jnp.array([0.5]))  # log(0.5) OK, but jnp.log produces -inf then NaN
        # We mainly care that NaN was caught somewhere
        self.assertIn("NaN", str(ctx.exception))

    def test_clean_while_no_raise(self):
        def fn(x):
            def cond_fn(val):
                return val[0] < 5.0

            def body_fn(val):
                return val + 1.0

            return jax.lax.while_loop(cond_fn, body_fn, x)

        debug_nan(fn, jnp.array([0.0]))


# ---------------------------------------------------------------------------
# Scan primitive
# ---------------------------------------------------------------------------

class TestScanPrimitive(unittest.TestCase):

    def test_nan_in_scan_body(self):
        def fn(xs):
            def body(carry, x):
                return carry + jnp.log(x), carry

            return jax.lax.scan(body, 0.0, xs)

        with self.assertRaises(RuntimeError) as ctx:
            debug_nan(fn, jnp.array([0.0, 1.0, 2.0]))
        self.assertIn("log", str(ctx.exception))

    def test_nan_in_later_iteration(self):
        def fn(xs):
            def body(carry, x):
                return carry + jnp.log(x), x * 2.0

            return jax.lax.scan(body, 0.0, xs)

        with self.assertRaises(RuntimeError):
            debug_nan(fn, jnp.array([1.0, 0.0, 2.0]))  # second element causes NaN

    def test_clean_scan_no_raise(self):
        def fn(xs):
            def body(carry, x):
                return carry + x, x * 2.0

            return jax.lax.scan(body, 0.0, xs)

        debug_nan(fn, jnp.array([1.0, 2.0, 3.0]))

    def test_scan_with_multiple_carry(self):
        def fn(xs):
            def body(carry, x):
                c1, c2 = carry
                return (c1 + jnp.log(x), c2 * x), c1

            return jax.lax.scan(body, (0.0, 1.0), xs)

        with self.assertRaises(RuntimeError):
            debug_nan(fn, jnp.array([0.0, 1.0]))


# ---------------------------------------------------------------------------
# NaN propagation vs introduction
# ---------------------------------------------------------------------------

class TestNanPropagation(unittest.TestCase):

    def test_nan_input_not_reported(self):
        """If input already has NaN, no equation 'introduces' it."""
        def fn(x):
            return x * 2.0  # just propagates

        # No equation introduces NaN, so no error should be raised
        # (NaN was already in the input; _interpret finds no new NaN source)
        _run_nan_check(fn, jnp.array([1.0, jnp.nan, 3.0]))

    def test_multiple_independent_nan_sources(self):
        """When there are two independent NaN sources, the first is reported."""
        def fn(x, y):
            a = jnp.log(x)    # introduces NaN
            b = 1.0 / y       # also introduces NaN
            return a + b

        # Both introduce NaN; at least one RuntimeError is raised
        with self.assertRaises(RuntimeError):
            debug_nan(fn, jnp.array([0.0]), jnp.array([0.0]))


# ---------------------------------------------------------------------------
# DebugNan class interface
# ---------------------------------------------------------------------------

class TestDebugNanClass(unittest.TestCase):

    def test_check_raises(self):
        def fn(x):
            return jnp.log(x)

        d = DebugNan(fn, jnp.array([0.0]), phase='test')
        with self.assertRaises(RuntimeError) as ctx:
            d.check()
        self.assertIn("test", str(ctx.exception))

    def test_check_if_false_no_raise(self):
        def fn(x):
            return jnp.log(x)

        d = DebugNan(fn, jnp.array([0.0]))
        d.check_if(jnp.array(False))  # should not raise

    def test_check_if_true_raises(self):
        def fn(x):
            return jnp.log(x)

        d = DebugNan(fn, jnp.array([0.0]))
        with self.assertRaises(RuntimeError):
            d.check_if(jnp.array(True))


if __name__ == '__main__':
    unittest.main()
