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
    _check_for_nan,
    _check_pytree_for_nan,
    _eval_jaxpr_with_nan_check,
    _format_nan_report,
    debug_nan,
    debug_nan_if,
)


class TestCheckForNan(unittest.TestCase):
    """Tests for _check_for_nan function."""

    def test_no_nan(self):
        """Clean array should return (False, 0, None)."""
        x = jnp.array([1.0, 2.0, 3.0])
        has_nan, count, indices = _check_for_nan(x)
        self.assertFalse(has_nan)
        self.assertEqual(count, 0)
        self.assertIsNone(indices)

    def test_with_nan(self):
        """Array with NaN should be detected."""
        x = jnp.array([1.0, jnp.nan, 3.0])
        has_nan, count, indices = _check_for_nan(x)
        self.assertTrue(has_nan)
        self.assertEqual(count, 1)
        self.assertIsNotNone(indices)

    def test_with_inf(self):
        """Array with Inf should be detected."""
        x = jnp.array([1.0, jnp.inf, 3.0])
        has_nan, count, indices = _check_for_nan(x)
        self.assertTrue(has_nan)
        self.assertEqual(count, 1)

    def test_with_neg_inf(self):
        """Array with -Inf should be detected."""
        x = jnp.array([1.0, -jnp.inf, 3.0])
        has_nan, count, indices = _check_for_nan(x)
        self.assertTrue(has_nan)
        self.assertEqual(count, 1)

    def test_mixed_nan_inf(self):
        """Array with both NaN and Inf should count all."""
        x = jnp.array([jnp.nan, jnp.inf, -jnp.inf, 4.0])
        has_nan, count, indices = _check_for_nan(x)
        self.assertTrue(has_nan)
        self.assertEqual(count, 3)

    def test_scalar(self):
        """Scalar value should be handled correctly."""
        x = jnp.array(jnp.nan)
        has_nan, count, indices = _check_for_nan(x)
        self.assertTrue(has_nan)
        self.assertEqual(count, 1)
        self.assertEqual(indices, ())  # Empty tuple for scalar

    def test_non_float(self):
        """Integer arrays should return (False, 0, None)."""
        x = jnp.array([1, 2, 3])
        has_nan, count, indices = _check_for_nan(x)
        self.assertFalse(has_nan)
        self.assertEqual(count, 0)
        self.assertIsNone(indices)

    def test_non_array(self):
        """Non-array inputs should return (False, 0, None)."""
        has_nan, count, indices = _check_for_nan(42)
        self.assertFalse(has_nan)
        self.assertEqual(count, 0)
        self.assertIsNone(indices)

        has_nan, count, indices = _check_for_nan("string")
        self.assertFalse(has_nan)


class TestCheckPytreeForNan(unittest.TestCase):
    """Tests for _check_pytree_for_nan function."""

    def test_clean_pytree(self):
        """Pytree with no NaN should return (False, [])."""
        pytree = {
            'a': jnp.array([1.0, 2.0]),
            'b': jnp.array([3.0, 4.0]),
        }
        has_nan, results = _check_pytree_for_nan(pytree)
        self.assertFalse(has_nan)
        self.assertEqual(len(results), 0)

    def test_nan_in_one_leaf(self):
        """Pytree with NaN in one leaf should detect it."""
        pytree = {
            'a': jnp.array([1.0, jnp.nan]),
            'b': jnp.array([3.0, 4.0]),
        }
        has_nan, results = _check_pytree_for_nan(pytree)
        self.assertTrue(has_nan)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['nan_count'], 1)

    def test_nan_in_multiple_leaves(self):
        """Pytree with NaN in multiple leaves should detect all."""
        pytree = {
            'a': jnp.array([1.0, jnp.nan]),
            'b': jnp.array([jnp.inf, 4.0]),
        }
        has_nan, results = _check_pytree_for_nan(pytree)
        self.assertTrue(has_nan)
        self.assertEqual(len(results), 2)

    def test_nested_pytree(self):
        """Nested pytree should be checked correctly."""
        pytree = {
            'outer': {
                'inner': jnp.array([jnp.nan, 2.0]),
            },
            'other': jnp.array([1.0, 2.0]),
        }
        has_nan, results = _check_pytree_for_nan(pytree)
        self.assertTrue(has_nan)
        self.assertEqual(len(results), 1)

    def test_empty_pytree(self):
        """Empty pytree should return (False, [])."""
        has_nan, results = _check_pytree_for_nan({})
        self.assertFalse(has_nan)
        self.assertEqual(len(results), 0)


class TestDebugNanIf(unittest.TestCase):
    """Tests for debug_nan_if function."""

    def test_no_nan_no_error(self):
        """When has_nan is False, no error should be raised."""
        def fn(x):
            return x * 2

        # Should not raise
        debug_nan_if(False, fn, jnp.array([1.0, 2.0]))

    def test_nan_raises_error(self):
        """When has_nan is True and function produces NaN, error should be raised."""
        def fn(x):
            return jnp.log(x)  # log(0) produces -inf

        with self.assertRaises(Exception):
            debug_nan_if(True, fn, jnp.array([0.0, 1.0]))

    def test_inf_raises_error(self):
        """When function produces Inf, error should be raised."""
        def fn(x):
            return 1.0 / x  # 1/0 produces inf

        with self.assertRaises(Exception):
            debug_nan_if(True, fn, jnp.array([0.0, 1.0]))

    def test_boolean_predicate(self):
        """Boolean has_nan parameter should work."""
        def fn(x):
            return x * 2

        # False predicate - no error
        debug_nan_if(False, fn, jnp.array([1.0, 2.0]))

        # True predicate with clean function - should still raise
        # because debug_nan always calls the callback
        with self.assertRaises(Exception):
            debug_nan_if(True, fn, jnp.array([1.0, 2.0]))

    def test_array_predicate(self):
        """JAX array has_nan parameter should work."""
        def fn(x):
            return jnp.log(x)

        # Array predicate that is False
        debug_nan_if(jnp.array(False), fn, jnp.array([1.0, 2.0]))

        # Array predicate that is True
        with self.assertRaises(Exception):
            debug_nan_if(jnp.array(True), fn, jnp.array([0.0, 1.0]))

    def test_jit_compatible(self):
        """debug_nan_if should work under jax.jit."""
        @jax.jit
        def f(x, trigger):
            debug_nan_if(trigger, lambda y: jnp.log(y), x)
            return x

        # Should not raise when trigger is False
        result = f(jnp.array([0.0, 1.0]), jnp.array(False))
        self.assertEqual(result.shape, (2,))

        # Should raise when trigger is True
        with self.assertRaises(Exception):
            f(jnp.array([0.0, 1.0]), jnp.array(True))


class TestDebugNan(unittest.TestCase):
    """Tests for debug_nan function."""

    def test_basic_nan_detection(self):
        """debug_nan should detect NaN in function output."""
        def fn(x):
            return jnp.log(x)

        # Function that produces NaN/Inf should raise
        with self.assertRaises(Exception):
            debug_nan(fn, jnp.array([0.0, 1.0, 2.0]))

    def test_error_message_contains_primitive(self):
        """Error message should contain primitive name."""
        def fn(x):
            return jnp.log(x)

        try:
            debug_nan(fn, jnp.array([0.0, 1.0]))
            self.fail("Expected RuntimeError")
        except Exception as e:
            error_msg = str(e)
            self.assertIn("log", error_msg.lower())

    def test_error_message_contains_input_values(self):
        """Error message should contain input values for small arrays."""
        def fn(x):
            return jnp.log(x)

        try:
            debug_nan(fn, jnp.array([0.0, 1.0]))
            self.fail("Expected RuntimeError")
        except Exception as e:
            error_msg = str(e)
            # Should show the input value that caused NaN
            self.assertIn("0.", error_msg)


class TestEvalJaxprWithNanCheck(unittest.TestCase):
    """Tests for _eval_jaxpr_with_nan_check function."""

    def test_log_of_zero(self):
        """Should detect NaN from log(0)."""
        def fn(x):
            return jnp.log(x)

        jaxpr = jax.make_jaxpr(fn)(jnp.array([0.0, 1.0, 2.0]))
        outputs, nan_report, eqn_strs = _eval_jaxpr_with_nan_check(
            jaxpr.jaxpr, jaxpr.consts, jnp.array([0.0, 1.0, 2.0])
        )

        self.assertTrue(len(nan_report) > 0)
        self.assertEqual(nan_report[0]['primitive'], 'log')

    def test_division_by_zero(self):
        """Should detect NaN from division by zero."""
        def fn(x):
            return 1.0 / x

        jaxpr = jax.make_jaxpr(fn)(jnp.array([0.0, 1.0, 2.0]))
        outputs, nan_report, eqn_strs = _eval_jaxpr_with_nan_check(
            jaxpr.jaxpr, jaxpr.consts, jnp.array([0.0, 1.0, 2.0])
        )

        self.assertTrue(len(nan_report) > 0)
        self.assertEqual(nan_report[0]['primitive'], 'div')

    def test_sqrt_negative(self):
        """Should detect NaN from sqrt of negative number."""
        def fn(x):
            return jnp.sqrt(x)

        jaxpr = jax.make_jaxpr(fn)(jnp.array([-1.0, 1.0, 4.0]))
        outputs, nan_report, eqn_strs = _eval_jaxpr_with_nan_check(
            jaxpr.jaxpr, jaxpr.consts, jnp.array([-1.0, 1.0, 4.0])
        )

        self.assertTrue(len(nan_report) > 0)
        self.assertEqual(nan_report[0]['primitive'], 'sqrt')

    def test_clean_function(self):
        """Clean function should produce no NaN report."""
        def fn(x):
            return x * 2 + 1

        jaxpr = jax.make_jaxpr(fn)(jnp.array([1.0, 2.0, 3.0]))
        outputs, nan_report, eqn_strs = _eval_jaxpr_with_nan_check(
            jaxpr.jaxpr, jaxpr.consts, jnp.array([1.0, 2.0, 3.0])
        )

        self.assertEqual(len(nan_report), 0)

    def test_nan_report_format(self):
        """NaN report should contain expected fields."""
        def fn(x):
            return jnp.log(x)

        jaxpr = jax.make_jaxpr(fn)(jnp.array([0.0, 1.0]))
        outputs, nan_report, eqn_strs = _eval_jaxpr_with_nan_check(
            jaxpr.jaxpr, jaxpr.consts, jnp.array([0.0, 1.0])
        )

        self.assertTrue(len(nan_report) > 0)
        report = nan_report[0]
        self.assertIn('eqn_index', report)
        self.assertIn('primitive', report)
        self.assertIn('input_shapes', report)
        self.assertIn('output_shapes', report)
        self.assertIn('input_values', report)

    def test_nested_jit_expansion(self):
        """Should expand and check inside JIT primitives."""
        @jax.jit
        def inner_fn(x):
            return jnp.log(x)

        def outer_fn(x):
            return inner_fn(x) * 2

        jaxpr = jax.make_jaxpr(outer_fn)(jnp.array([0.0, 1.0, 2.0]))
        outputs, nan_report, eqn_strs = _eval_jaxpr_with_nan_check(
            jaxpr.jaxpr, jaxpr.consts, jnp.array([0.0, 1.0, 2.0])
        )

        self.assertTrue(len(nan_report) > 0)
        # Should detect NaN inside the JIT
        found_log = any(r['primitive'] == 'log' for r in nan_report)
        self.assertTrue(found_log)


class TestFormatNanReport(unittest.TestCase):
    """Tests for _format_nan_report function."""

    def test_empty_report(self):
        """Empty report should produce 'No NaN/Inf detected' message."""
        result = _format_nan_report([], 10, [])
        self.assertIn("No NaN/Inf detected", result)

    def test_single_equation_report(self):
        """Single NaN source should be formatted correctly."""
        nan_report = [{
            'eqn_index': 2,
            'primitive': 'log',
            'input_shapes': [(3,)],
            'output_shapes': [(3,)],
            'input_values': [jnp.array([0.0, 1.0, 2.0])],
            'nan_details': [],
            'equation_str': 'a = log b',
            'source_info': None,
        }]
        eqn_strs = ['a = add b c', 'b = mul c d', 'c = log d', 'd = sub e f']

        result = _format_nan_report(nan_report, 4, eqn_strs)

        self.assertIn("NaN/Inf detected", result)
        self.assertIn("log", result)
        self.assertIn("Equation 2", result)

    def test_multiple_equation_report(self):
        """Multiple NaN sources should all be reported."""
        nan_report = [
            {
                'eqn_index': 0,
                'primitive': 'log',
                'input_shapes': [(3,)],
                'output_shapes': [(3,)],
                'input_values': [jnp.array([0.0, 1.0, 2.0])],
                'nan_details': [],
                'equation_str': 'a = log b',
                'source_info': None,
            },
            {
                'eqn_index': 2,
                'primitive': 'div',
                'input_shapes': [(), (3,)],
                'output_shapes': [(3,)],
                'input_values': [1.0, jnp.array([0.0, 1.0, 2.0])],
                'nan_details': [],
                'equation_str': 'a = div b c',
                'source_info': None,
            },
        ]
        eqn_strs = ['a = log b', 'b = mul c d', 'c = div e f']

        result = _format_nan_report(nan_report, 3, eqn_strs)

        self.assertIn("Found in 2 equation(s)", result)
        self.assertIn("log", result)
        self.assertIn("div", result)

    def test_context_window(self):
        """Context window should show surrounding equations."""
        nan_report = [{
            'eqn_index': 5,
            'primitive': 'log',
            'input_shapes': [(3,)],
            'output_shapes': [(3,)],
            'input_values': [jnp.array([0.0, 1.0, 2.0])],
            'nan_details': [],
            'equation_str': 'a = log b',
            'source_info': None,
        }]
        eqn_strs = [f'eqn_{i}' for i in range(10)]

        result = _format_nan_report(nan_report, 10, eqn_strs, context=2)

        # Should show equations 3, 4, 5, 6, 7 (within context=2 of index 5)
        self.assertIn("eqn_3", result)
        self.assertIn("eqn_4", result)
        self.assertIn("eqn_5", result)
        self.assertIn("eqn_6", result)
        self.assertIn("eqn_7", result)


class TestNestedHighLevelPrimitives(unittest.TestCase):
    """Tests for nested high-level primitives (JIT, cond, etc.)."""

    # =========================================================================
    # Nested JIT Tests
    # =========================================================================

    def test_nested_jit_nan_detected(self):
        """JIT inside JIT should detect NaN in innermost function."""
        @jax.jit
        def inner_fn(x):
            return jnp.log(x)  # NaN source

        @jax.jit
        def outer_fn(x):
            return inner_fn(x) * 2

        jaxpr = jax.make_jaxpr(outer_fn)(jnp.array([0.0, 1.0, 2.0]))
        outputs, nan_report, eqn_strs = _eval_jaxpr_with_nan_check(
            jaxpr.jaxpr, jaxpr.consts, jnp.array([0.0, 1.0, 2.0])
        )

        self.assertTrue(len(nan_report) > 0)
        # Should detect log as the NaN source
        found_log = any(r['primitive'] == 'log' for r in nan_report)
        self.assertTrue(found_log)

    def test_deeply_nested_jit(self):
        """Multiple levels of JIT nesting should all be expanded."""
        @jax.jit
        def level3(x):
            return jnp.log(x)  # NaN source

        @jax.jit
        def level2(x):
            return level3(x) + 1

        @jax.jit
        def level1(x):
            return level2(x) * 2

        with self.assertRaises(RuntimeError) as cm:
            debug_nan(level1, jnp.array([0.0, 1.0, 2.0]))
        self.assertIn("NaN/Inf detected", str(cm.exception))
        self.assertIn("log", str(cm.exception))

        # jaxpr = jax.make_jaxpr(level1)(jnp.array([0.0, 1.0, 2.0]))
        # outputs, nan_report, eqn_strs = _eval_jaxpr_with_nan_check(
        #     jaxpr.jaxpr, jaxpr.consts, jnp.array([0.0, 1.0, 2.0])
        # )
        #
        # self.assertTrue(len(nan_report) > 0)
        # found_log = any(r['primitive'] == 'log' for r in nan_report)
        # self.assertTrue(found_log)

    def test_jit_with_clean_inner(self):
        """JIT with clean inner computation should have no NaN report."""
        @jax.jit
        def inner_fn(x):
            return x * 2

        @jax.jit
        def outer_fn(x):
            return inner_fn(x) + 1

        jaxpr = jax.make_jaxpr(outer_fn)(jnp.array([1.0, 2.0, 3.0]))
        outputs, nan_report, eqn_strs = _eval_jaxpr_with_nan_check(
            jaxpr.jaxpr, jaxpr.consts, jnp.array([1.0, 2.0, 3.0])
        )

        self.assertEqual(len(nan_report), 0)

    # =========================================================================
    # Nested Cond Tests
    # =========================================================================

    def test_cond_inside_jit(self):
        """Cond primitive inside JIT should be expanded."""
        @jax.jit
        def fn(x):
            return jax.lax.cond(
                x[0] > 0,
                lambda y: jnp.log(y),  # NaN if y contains 0
                lambda y: y,
                x
            )

        jaxpr = jax.make_jaxpr(fn)(jnp.array([1.0, 0.0, 2.0]))
        # Use input that triggers the true branch (x[0] > 0)
        outputs, nan_report, eqn_strs = _eval_jaxpr_with_nan_check(
            jaxpr.jaxpr, jaxpr.consts, jnp.array([1.0, 0.0, 2.0])
        )

        self.assertTrue(len(nan_report) > 0)
        found_log = any(r['primitive'] == 'log' for r in nan_report)
        self.assertTrue(found_log)

    def test_jit_inside_cond_branch(self):
        """JIT inside cond branch should be expanded."""
        @jax.jit
        def nan_producer(x):
            return jnp.log(x)

        def fn(x):
            return jax.lax.cond(
                x[0] > 0,
                nan_producer,  # JIT-compiled function
                lambda y: y,
                x
            )

        jaxpr = jax.make_jaxpr(fn)(jnp.array([1.0, 0.0, 2.0]))
        outputs, nan_report, eqn_strs = _eval_jaxpr_with_nan_check(
            jaxpr.jaxpr, jaxpr.consts, jnp.array([1.0, 0.0, 2.0])
        )

        self.assertTrue(len(nan_report) > 0)

    def test_cond_false_branch_no_nan(self):
        """Cond taking false branch should not report NaN from true branch."""
        def fn(x):
            return jax.lax.cond(
                x[0] > 0,
                lambda y: jnp.log(y),  # NaN producer - not taken
                lambda y: y * 2,  # Clean - taken
                x
            )

        jaxpr = jax.make_jaxpr(fn)(jnp.array([-1.0, 1.0, 2.0]))
        # Use input where x[0] <= 0, so false branch is taken
        outputs, nan_report, eqn_strs = _eval_jaxpr_with_nan_check(
            jaxpr.jaxpr, jaxpr.consts, jnp.array([-1.0, 1.0, 2.0])
        )

        # False branch is clean, should have no NaN report
        self.assertEqual(len(nan_report), 0)

    # =========================================================================
    # Empty NaN Report Edge Cases
    # =========================================================================

    def test_nan_in_inputs_propagated(self):
        """NaN in inputs that gets propagated should still be detected."""
        def fn(x):
            # Just multiply - doesn't introduce NaN, but propagates it
            return x * 2

        jaxpr = jax.make_jaxpr(fn)(jnp.array([1.0, jnp.nan, 3.0]))
        outputs, nan_report, eqn_strs = _eval_jaxpr_with_nan_check(
            jaxpr.jaxpr, jaxpr.consts, jnp.array([1.0, jnp.nan, 3.0])
        )

        # NaN exists in inputs, so no equation "introduces" it
        # The report should be empty because input already has NaN
        self.assertEqual(len(nan_report), 0)
        # But outputs should still contain NaN
        self.assertTrue(jnp.any(jnp.isnan(outputs[0])))

    def test_nan_introduced_then_propagated(self):
        """NaN introduced by one op and propagated should only report source."""
        def fn(x):
            y = jnp.log(x)  # Introduces NaN for x=0
            z = y * 2  # Propagates NaN
            return z + 1  # Still propagates NaN

        jaxpr = jax.make_jaxpr(fn)(jnp.array([0.0, 1.0, 2.0]))
        outputs, nan_report, eqn_strs = _eval_jaxpr_with_nan_check(
            jaxpr.jaxpr, jaxpr.consts, jnp.array([0.0, 1.0, 2.0])
        )

        # Should only report log as the source, not mul or add
        self.assertEqual(len(nan_report), 1)
        self.assertEqual(nan_report[0]['primitive'], 'log')

    def test_multiple_nan_sources(self):
        """Multiple independent NaN sources should all be reported."""
        def fn(x, y):
            a = jnp.log(x)  # NaN from log(0)
            b = 1.0 / y  # NaN from 1/0
            return a + b

        jaxpr = jax.make_jaxpr(fn)(jnp.array([0.0]), jnp.array([0.0]))
        outputs, nan_report, eqn_strs = _eval_jaxpr_with_nan_check(
            jaxpr.jaxpr, jaxpr.consts, jnp.array([0.0]), jnp.array([0.0])
        )

        # Should report both log and div
        self.assertEqual(len(nan_report), 2)
        primitives = {r['primitive'] for r in nan_report}
        self.assertIn('log', primitives)
        self.assertIn('div', primitives)

    def test_nested_jit_report_has_metadata(self):
        """NaN report from nested JIT should include nesting metadata."""
        @jax.jit
        def inner_fn(x):
            return jnp.log(x)

        def outer_fn(x):
            return inner_fn(x) * 2

        jaxpr = jax.make_jaxpr(outer_fn)(jnp.array([0.0, 1.0]))
        outputs, nan_report, eqn_strs = _eval_jaxpr_with_nan_check(
            jaxpr.jaxpr, jaxpr.consts, jnp.array([0.0, 1.0])
        )

        self.assertTrue(len(nan_report) > 0)
        # Reports from inside JIT should have 'inside_jit' flag
        has_jit_flag = any(r.get('inside_jit', False) for r in nan_report)
        self.assertTrue(has_jit_flag)

    def test_cond_report_has_branch_metadata(self):
        """NaN report from cond should include branch metadata."""
        def fn(x):
            return jax.lax.cond(
                x[0] > 0,
                lambda y: jnp.log(y),
                lambda y: y,
                x
            )

        jaxpr = jax.make_jaxpr(fn)(jnp.array([1.0, 0.0]))
        outputs, nan_report, eqn_strs = _eval_jaxpr_with_nan_check(
            jaxpr.jaxpr, jaxpr.consts, jnp.array([1.0, 0.0])
        )

        self.assertTrue(len(nan_report) > 0)
        # Reports from inside cond should have branch info
        has_cond_flag = any(r.get('inside_cond', False) for r in nan_report)
        self.assertTrue(has_cond_flag)

    # =========================================================================
    # Integration with debug_nan_if
    # =========================================================================

    def test_debug_nan_if_nested_jit(self):
        """debug_nan_if should work with nested JIT functions."""
        @jax.jit
        def inner(x):
            return jnp.log(x)

        @jax.jit
        def outer(x):
            return inner(x) * 2

        with self.assertRaises(Exception):
            debug_nan_if(True, outer, jnp.array([0.0, 1.0]))

    def test_debug_nan_if_cond(self):
        """debug_nan_if should work with cond primitives."""
        def fn(x):
            return jax.lax.cond(
                x[0] > 0,
                lambda y: jnp.log(y),
                lambda y: y,
                x
            )

        with self.assertRaises(Exception):
            debug_nan_if(True, fn, jnp.array([1.0, 0.0]))


class TestWhilePrimitive(unittest.TestCase):
    """Tests for while_loop NaN detection."""

    def test_while_nan_in_first_iteration(self):
        """Should detect NaN in the first iteration of while loop."""
        def fn(x):
            def cond(val):
                return val[0] < 10

            def body(val):
                return jnp.log(val)  # NaN if val contains 0 or negative

            return jax.lax.while_loop(cond, body, x)

        jaxpr = jax.make_jaxpr(fn)(jnp.array([0.5]))
        outputs, nan_report, eqn_strs = _eval_jaxpr_with_nan_check(
            jaxpr.jaxpr, jaxpr.consts, jnp.array([0.0])  # 0.0 causes NaN
        )

        self.assertTrue(len(nan_report) > 0)
        # Check that it's tagged as inside_while
        has_while_flag = any(r.get('inside_while', False) for r in nan_report)
        self.assertTrue(has_while_flag)

    def test_while_clean_no_nan(self):
        """Clean while loop should produce no NaN report."""
        def fn(x):
            def cond(val):
                return val[0] < 10

            def body(val):
                return val + 1.0

            return jax.lax.while_loop(cond, body, x)

        jaxpr = jax.make_jaxpr(fn)(jnp.array([0.0]))
        outputs, nan_report, eqn_strs = _eval_jaxpr_with_nan_check(
            jaxpr.jaxpr, jaxpr.consts, jnp.array([0.0])
        )

        self.assertEqual(len(nan_report), 0)

    def test_while_report_has_iteration_metadata(self):
        """NaN report from while should include iteration metadata."""
        def fn(x):
            def cond(val):
                return val[0] < 10

            def body(val):
                return jnp.log(val)

            return jax.lax.while_loop(cond, body, x)

        jaxpr = jax.make_jaxpr(fn)(jnp.array([0.5]))
        outputs, nan_report, eqn_strs = _eval_jaxpr_with_nan_check(
            jaxpr.jaxpr, jaxpr.consts, jnp.array([0.0])
        )

        self.assertTrue(len(nan_report) > 0)
        report = nan_report[0]
        self.assertIn('inside_while', report)
        self.assertIn('iteration_index', report)
        self.assertIn('while_part', report)


class TestScanPrimitive(unittest.TestCase):
    """Tests for scan NaN detection."""

    def test_scan_nan_in_first_iteration(self):
        """Should detect NaN in the first iteration of scan."""
        def fn(xs):
            def body(carry, x):
                return carry + jnp.log(x), carry

            return jax.lax.scan(body, 0.0, xs)

        jaxpr = jax.make_jaxpr(fn)(jnp.array([0.0, 1.0, 2.0]))
        outputs, nan_report, eqn_strs = _eval_jaxpr_with_nan_check(
            jaxpr.jaxpr, jaxpr.consts, jnp.array([0.0, 1.0, 2.0])  # First element causes NaN
        )

        self.assertTrue(len(nan_report) > 0)
        # Should detect log as the NaN source
        found_log = any(r['primitive'] == 'log' for r in nan_report)
        self.assertTrue(found_log)

    def test_scan_nan_in_later_iteration(self):
        """Should detect NaN that appears in a later iteration."""
        def fn(xs):
            def body(carry, x):
                return carry + jnp.log(x), x * 2

            return jax.lax.scan(body, 0.0, xs)

        jaxpr = jax.make_jaxpr(fn)(jnp.array([1.0, 0.0, 2.0]))
        outputs, nan_report, eqn_strs = _eval_jaxpr_with_nan_check(
            jaxpr.jaxpr, jaxpr.consts, jnp.array([1.0, 0.0, 2.0])  # Second element causes NaN
        )

        self.assertTrue(len(nan_report) > 0)
        # Check iteration index
        has_scan_flag = any(r.get('inside_scan', False) for r in nan_report)
        self.assertTrue(has_scan_flag)

    def test_scan_clean_no_nan(self):
        """Clean scan should produce no NaN report."""
        def fn(xs):
            def body(carry, x):
                return carry + x, x * 2

            return jax.lax.scan(body, 0.0, xs)

        jaxpr = jax.make_jaxpr(fn)(jnp.array([1.0, 2.0, 3.0]))
        outputs, nan_report, eqn_strs = _eval_jaxpr_with_nan_check(
            jaxpr.jaxpr, jaxpr.consts, jnp.array([1.0, 2.0, 3.0])
        )

        self.assertEqual(len(nan_report), 0)

    def test_scan_report_has_iteration_metadata(self):
        """NaN report from scan should include iteration metadata."""
        def fn(xs):
            def body(carry, x):
                return carry + jnp.log(x), carry

            return jax.lax.scan(body, 0.0, xs)

        jaxpr = jax.make_jaxpr(fn)(jnp.array([0.0, 1.0]))
        outputs, nan_report, eqn_strs = _eval_jaxpr_with_nan_check(
            jaxpr.jaxpr, jaxpr.consts, jnp.array([0.0, 1.0])
        )

        self.assertTrue(len(nan_report) > 0)
        report = nan_report[0]
        self.assertIn('inside_scan', report)
        self.assertIn('iteration_index', report)

    def test_scan_with_multiple_carry(self):
        """Should handle scan with multiple carry values."""
        def fn(xs):
            def body(carry, x):
                c1, c2 = carry
                new_c1 = c1 + jnp.log(x)  # NaN source
                new_c2 = c2 * x
                return (new_c1, new_c2), c1

            return jax.lax.scan(body, (0.0, 1.0), xs)

        jaxpr = jax.make_jaxpr(fn)(jnp.array([0.0, 1.0]))
        outputs, nan_report, eqn_strs = _eval_jaxpr_with_nan_check(
            jaxpr.jaxpr, jaxpr.consts, jnp.array([0.0, 1.0])
        )

        self.assertTrue(len(nan_report) > 0)


class TestNestedWhileScanPrimitives(unittest.TestCase):
    """Tests for nested combinations of while, scan, cond, and jit."""

    def test_while_inside_jit(self):
        """while_loop inside JIT should be expanded and checked."""
        @jax.jit
        def fn(x):
            def cond(val):
                return val[0] < 10

            def body(val):
                return jnp.log(val)

            return jax.lax.while_loop(cond, body, x)

        jaxpr = jax.make_jaxpr(fn)(jnp.array([0.5]))
        outputs, nan_report, eqn_strs = _eval_jaxpr_with_nan_check(
            jaxpr.jaxpr, jaxpr.consts, jnp.array([0.0])
        )

        self.assertTrue(len(nan_report) > 0)

    def test_scan_inside_jit(self):
        """scan inside JIT should be expanded and checked."""
        @jax.jit
        def fn(xs):
            def body(carry, x):
                return carry + jnp.log(x), carry

            return jax.lax.scan(body, 0.0, xs)

        jaxpr = jax.make_jaxpr(fn)(jnp.array([0.0, 1.0]))
        outputs, nan_report, eqn_strs = _eval_jaxpr_with_nan_check(
            jaxpr.jaxpr, jaxpr.consts, jnp.array([0.0, 1.0])
        )

        self.assertTrue(len(nan_report) > 0)

    def test_scan_inside_cond(self):
        """scan inside cond should be expanded and checked."""
        def fn(x, xs):
            def true_branch(xs_):
                def body(carry, x):
                    return carry + jnp.log(x), carry

                return jax.lax.scan(body, 0.0, xs_)[0]

            def false_branch(xs_):
                return jnp.sum(xs_)

            return jax.lax.cond(x > 0, true_branch, false_branch, xs)

        jaxpr = jax.make_jaxpr(fn)(1.0, jnp.array([0.0, 1.0]))
        # Taking true branch with NaN-causing input
        outputs, nan_report, eqn_strs = _eval_jaxpr_with_nan_check(
            jaxpr.jaxpr, jaxpr.consts, 1.0, jnp.array([0.0, 1.0])
        )

        self.assertTrue(len(nan_report) > 0)

    def test_debug_nan_if_with_while(self):
        """debug_nan_if should work with while_loop."""
        def fn(x):
            def cond(val):
                return val[0] < 10

            def body(val):
                return jnp.log(val)

            return jax.lax.while_loop(cond, body, x)

        with self.assertRaises(Exception):
            debug_nan_if(True, fn, jnp.array([0.0]))

    def test_debug_nan_if_with_scan(self):
        """debug_nan_if should work with scan."""
        def fn(xs):
            def body(carry, x):
                return carry + jnp.log(x), carry

            return jax.lax.scan(body, 0.0, xs)[0]

        with self.assertRaises(Exception):
            debug_nan_if(True, fn, jnp.array([0.0, 1.0]))


if __name__ == '__main__':
    unittest.main()
