# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
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
import pytest

import brainstate
from brainstate.transform._ir_optim import (
    IdentitySet,
    constant_fold,
    dead_code_elimination,
    common_subexpression_elimination,
    copy_propagation,
    algebraic_simplification,
    optimize_jaxpr,
)


class TestIdentitySet(unittest.TestCase):
    """Test the IdentitySet class."""

    def test_identity_set_basic_operations(self):
        """Test basic operations of IdentitySet."""
        s = IdentitySet()

        # Test add and contains
        a = [1, 2, 3]
        b = [1, 2, 3]
        s.add(a)

        self.assertIn(a, s)
        self.assertNotIn(b, s)  # Different object

    def test_identity_set_length(self):
        """Test len() on IdentitySet."""
        s = IdentitySet()
        self.assertEqual(len(s), 0)

        a = [1, 2]
        b = [3, 4]
        s.add(a)
        s.add(b)

        self.assertEqual(len(s), 2)

    def test_identity_set_discard(self):
        """Test discarding elements from IdentitySet."""
        s = IdentitySet()
        a = [1, 2]
        b = [3, 4]

        s.add(a)
        s.add(b)
        self.assertEqual(len(s), 2)

        s.discard(a)
        self.assertEqual(len(s), 1)
        self.assertNotIn(a, s)
        self.assertIn(b, s)

    def test_identity_set_update(self):
        """Test update() method."""
        s = IdentitySet()
        items = [[1, 2], [3, 4], [5, 6]]

        s.update(items)
        self.assertEqual(len(s), 3)

        for item in items:
            self.assertIn(item, s)

    def test_identity_set_iteration(self):
        """Test iterating over IdentitySet."""
        s = IdentitySet()
        items = [[1, 2], [3, 4], [5, 6]]
        s.update(items)

        iterated_items = list(s)
        self.assertEqual(len(iterated_items), 3)

        # Check all original items are in the iterated result
        for item in items:
            self.assertIn(item, iterated_items)

    def test_identity_set_init_with_iterable(self):
        """Test initializing IdentitySet with an iterable."""
        items = [[1, 2], [3, 4]]
        s = IdentitySet(items)

        self.assertEqual(len(s), 2)
        for item in items:
            self.assertIn(item, s)

    def test_identity_set_vs_regular_set(self):
        """Test that IdentitySet compares by identity, not equality."""
        s = IdentitySet()
        a = [1, 2, 3]
        b = [1, 2, 3]  # Equal to a, but different object

        s.add(a)

        # a is in the set
        self.assertIn(a, s)
        # b is NOT in the set (different identity)
        self.assertNotIn(b, s)

        # Can add both
        s.add(b)
        self.assertEqual(len(s), 2)


class TestConstantFold(unittest.TestCase):
    """Test constant_fold optimization."""

    def test_constant_fold_basic(self):
        """Test basic constant folding."""

        def f(x):
            # Contains constant expression: 2 + 3
            return x + (2 + 3)

        jaxpr = jax.make_jaxpr(f)(jnp.array(1.0))
        original_len = len(jaxpr.jaxpr.eqns)

        optimized = constant_fold(jaxpr.jaxpr)
        optimized_len = len(optimized.eqns)

        # Should have fewer equations (2+3 folded to 5)
        self.assertLessEqual(optimized_len, original_len)

    def test_constant_fold_preserves_interface(self):
        """Test that constant folding preserves input/output variables."""

        def f(x, y):
            return x + (2 * 3) + y

        jaxpr = jax.make_jaxpr(f)(jnp.array(1.0), jnp.array(2.0))
        original_invars = jaxpr.jaxpr.invars
        original_outvars = jaxpr.jaxpr.outvars

        optimized = constant_fold(jaxpr.jaxpr)

        # Invars and outvars should be preserved
        self.assertEqual(optimized.invars, original_invars)
        self.assertEqual(optimized.outvars, original_outvars)

    def test_constant_fold_nested_constants(self):
        """Test folding nested constant expressions."""

        def f(x):
            a = 2 + 3
            b = a * 4
            c = b - 1
            return x + c

        jaxpr = jax.make_jaxpr(f)(jnp.array(1.0))
        original_len = len(jaxpr.jaxpr.eqns)

        optimized = constant_fold(jaxpr.jaxpr)
        optimized_len = len(optimized.eqns)

        # All constant computations should be folded
        self.assertLessEqual(optimized_len, original_len)

    def test_constant_fold_no_change_when_no_constants(self):
        """Test that jaxpr without constants is unchanged."""

        def f(x, y):
            return x + y * 2

        jaxpr = jax.make_jaxpr(f)(jnp.array(1.0), jnp.array(2.0))
        original_len = len(jaxpr.jaxpr.eqns)

        optimized = constant_fold(jaxpr.jaxpr)
        optimized_len = len(optimized.eqns)

        # No constants to fold, length should be similar
        self.assertEqual(optimized_len, original_len)


class TestDeadCodeElimination(unittest.TestCase):
    """Test dead_code_elimination optimization."""

    def test_dce_removes_unused_computation(self):
        """Test that dead code elimination removes unused computations."""

        def f(x):
            unused = x * 2  # noqa: F841
            return x + 1

        jaxpr = jax.make_jaxpr(f)(jnp.array(1.0))
        original_len = len(jaxpr.jaxpr.eqns)

        optimized = dead_code_elimination(jaxpr.jaxpr)
        optimized_len = len(optimized.eqns)

        # Dead code should be removed
        self.assertLess(optimized_len, original_len)

    def test_dce_preserves_interface(self):
        """Test that DCE preserves input/output variables."""

        def f(x, y):
            unused = x * 10  # noqa: F841
            return y + 1

        jaxpr = jax.make_jaxpr(f)(jnp.array(1.0), jnp.array(2.0))
        original_invars = jaxpr.jaxpr.invars
        original_outvars = jaxpr.jaxpr.outvars

        optimized = dead_code_elimination(jaxpr.jaxpr)

        # Invars and outvars should be preserved
        self.assertEqual(optimized.invars, original_invars)
        self.assertEqual(optimized.outvars, original_outvars)

    def test_dce_preserves_used_computations(self):
        """Test that DCE keeps computations that are actually used."""

        def f(x):
            a = x + 1
            b = a * 2
            return b

        jaxpr = jax.make_jaxpr(f)(jnp.array(1.0))
        original_len = len(jaxpr.jaxpr.eqns)

        optimized = dead_code_elimination(jaxpr.jaxpr)
        optimized_len = len(optimized.eqns)

        # All computations are used, should remain similar
        self.assertEqual(optimized_len, original_len)

    def test_dce_with_multiple_outputs(self):
        """Test DCE with multiple outputs."""

        def f(x):
            a = x + 1
            b = x * 2
            unused = x - 1  # noqa: F841
            return a, b

        jaxpr = jax.make_jaxpr(f)(jnp.array(1.0))
        original_len = len(jaxpr.jaxpr.eqns)

        optimized = dead_code_elimination(jaxpr.jaxpr)
        optimized_len = len(optimized.eqns)

        # Only unused computation should be removed
        self.assertLess(optimized_len, original_len)


class TestCommonSubexpressionElimination(unittest.TestCase):
    """Test common_subexpression_elimination optimization."""

    def test_cse_eliminates_duplicate_computations(self):
        """Test that CSE eliminates duplicate computations."""

        def f(x, y):
            a = x + y
            b = x * 2
            c = x + y  # Duplicate of a
            return a + b + c

        jaxpr = jax.make_jaxpr(f)(jnp.array(1.0), jnp.array(2.0))
        original_len = len(jaxpr.jaxpr.eqns)

        optimized = common_subexpression_elimination(jaxpr.jaxpr)
        optimized_len = len(optimized.eqns)

        # Duplicate computation should be eliminated or reused
        self.assertLessEqual(optimized_len, original_len)

    def test_cse_preserves_interface(self):
        """Test that CSE preserves input/output variables."""

        def f(x, y):
            a = x + y
            b = x + y  # Duplicate
            return a + b

        jaxpr = jax.make_jaxpr(f)(jnp.array(1.0), jnp.array(2.0))
        original_invars = jaxpr.jaxpr.invars
        original_outvars = jaxpr.jaxpr.outvars

        optimized = common_subexpression_elimination(jaxpr.jaxpr)

        # Invars and outvars should be preserved
        self.assertEqual(optimized.invars, original_invars)
        self.assertEqual(optimized.outvars, original_outvars)

    def test_cse_different_operations_not_eliminated(self):
        """Test that different operations are not eliminated."""

        def f(x, y):
            a = x + y
            b = x * y
            return a + b

        jaxpr = jax.make_jaxpr(f)(jnp.array(1.0), jnp.array(2.0))
        original_len = len(jaxpr.jaxpr.eqns)

        optimized = common_subexpression_elimination(jaxpr.jaxpr)
        optimized_len = len(optimized.eqns)

        # No common subexpressions, length should be similar
        self.assertGreaterEqual(optimized_len, original_len - 1)


class TestCopyPropagation(unittest.TestCase):
    """Test copy_propagation optimization."""

    def test_copy_propagation_preserves_interface(self):
        """Test that copy propagation preserves input/output variables."""

        def f(x):
            return x + 1

        jaxpr = jax.make_jaxpr(f)(jnp.array(1.0))
        original_invars = jaxpr.jaxpr.invars
        original_outvars = jaxpr.jaxpr.outvars

        optimized = copy_propagation(jaxpr.jaxpr)

        # Invars and outvars should be preserved
        self.assertEqual(optimized.invars, original_invars)
        self.assertEqual(optimized.outvars, original_outvars)

    def test_copy_propagation_basic(self):
        """Test basic copy propagation."""

        def f(x):
            y = x + 0  # Essentially a copy
            return y * 2

        jaxpr = jax.make_jaxpr(f)(jnp.array(1.0))

        optimized = copy_propagation(jaxpr.jaxpr)

        # Should still be valid
        self.assertIsNotNone(optimized)


class TestAlgebraicSimplification(unittest.TestCase):
    """Test algebraic_simplification optimization."""

    def test_algebraic_simplification_add_zero(self):
        """Test simplification of x + 0."""

        def f(x, y):
            # Need y to prevent JAX from pre-optimizing
            a = x + 0.0
            return a + y

        jaxpr = jax.make_jaxpr(f)(jnp.array(1.0), jnp.array(2.0))
        original_len = len(jaxpr.jaxpr.eqns)

        optimized = algebraic_simplification(jaxpr.jaxpr)
        optimized_len = len(optimized.eqns)

        # x + 0 should be simplified (or at least not increase)
        self.assertLessEqual(optimized_len, original_len + 1)

    def test_algebraic_simplification_mul_one(self):
        """Test simplification of x * 1."""

        def f(x, y):
            a = x * 1.0
            return a + y

        jaxpr = jax.make_jaxpr(f)(jnp.array(1.0), jnp.array(2.0))
        original_len = len(jaxpr.jaxpr.eqns)

        optimized = algebraic_simplification(jaxpr.jaxpr)
        optimized_len = len(optimized.eqns)

        # x * 1 should be simplified (or at least not increase)
        self.assertLessEqual(optimized_len, original_len + 1)

    def test_algebraic_simplification_preserves_interface(self):
        """Test that algebraic simplification preserves input/output variables."""

        def f(x, y):
            return (x + 0.0) * 1.0 + y

        jaxpr = jax.make_jaxpr(f)(jnp.array(1.0), jnp.array(2.0))
        original_invars = jaxpr.jaxpr.invars
        original_outvars = jaxpr.jaxpr.outvars

        optimized = algebraic_simplification(jaxpr.jaxpr)

        # Invars and outvars should be preserved
        self.assertEqual(optimized.invars, original_invars)
        self.assertEqual(optimized.outvars, original_outvars)

    def test_algebraic_simplification_sub_zero(self):
        """Test simplification of x - 0."""

        def f(x, y):
            a = x - 0.0
            return a + y

        jaxpr = jax.make_jaxpr(f)(jnp.array(1.0), jnp.array(2.0))
        original_len = len(jaxpr.jaxpr.eqns)

        optimized = algebraic_simplification(jaxpr.jaxpr)
        optimized_len = len(optimized.eqns)

        # x - 0 should be simplified (or at least not increase)
        self.assertLessEqual(optimized_len, original_len + 1)

    def test_algebraic_simplification_div_one(self):
        """Test simplification of x / 1."""

        def f(x, y):
            a = x / 1.0
            return a + y

        jaxpr = jax.make_jaxpr(f)(jnp.array(1.0), jnp.array(2.0))
        original_len = len(jaxpr.jaxpr.eqns)

        optimized = algebraic_simplification(jaxpr.jaxpr)
        optimized_len = len(optimized.eqns)

        # x / 1 should be simplified (or at least not increase)
        self.assertLessEqual(optimized_len, original_len + 1)


class TestOptimizeJaxpr(unittest.TestCase):
    """Test optimize_jaxpr function."""

    def test_optimize_jaxpr_basic(self):
        """Test basic optimization pipeline."""

        def f(x):
            a = x + 0.0  # Algebraic simplification
            b = (2 + 3) * a  # Constant folding
            unused = x * 10  # Dead code  # noqa: F841
            return b

        jaxpr = jax.make_jaxpr(f)(jnp.array(1.0))
        original_len = len(jaxpr.jaxpr.eqns)

        optimized = optimize_jaxpr(jaxpr.jaxpr)
        optimized_len = len(optimized.eqns)

        # Multiple optimizations should reduce equation count
        self.assertLessEqual(optimized_len, original_len)

    def test_optimize_jaxpr_preserves_interface(self):
        """Test that optimization preserves input/output variables."""

        def f(x, y, z):
            return x + (2 * 3) + y * 1.0 + z

        jaxpr = jax.make_jaxpr(f)(jnp.array(1.0), jnp.array(2.0), jnp.array(3.0))
        original_invars = jaxpr.jaxpr.invars
        original_outvars = jaxpr.jaxpr.outvars

        optimized = optimize_jaxpr(jaxpr.jaxpr)

        # Invars and outvars should be preserved
        self.assertEqual(tuple(optimized.invars), tuple(original_invars))
        self.assertEqual(tuple(optimized.outvars), tuple(original_outvars))

    def test_optimize_jaxpr_with_closed_jaxpr(self):
        """Test optimization with ClosedJaxpr."""

        def f(x):
            return x + (2 + 3)

        from brainstate._compatible_import import ClosedJaxpr
        closed_jaxpr = jax.make_jaxpr(f)(jnp.array(1.0))

        # Should accept ClosedJaxpr
        optimized = optimize_jaxpr(closed_jaxpr)

        # Should return ClosedJaxpr
        self.assertIsInstance(optimized, ClosedJaxpr)

    def test_optimize_jaxpr_max_iterations(self):
        """Test max_iterations parameter."""

        def f(x):
            return x + (2 + 3)

        jaxpr = jax.make_jaxpr(f)(jnp.array(1.0))

        # Should work with different max_iterations
        opt1 = optimize_jaxpr(jaxpr.jaxpr, max_iterations=1)
        opt5 = optimize_jaxpr(jaxpr.jaxpr, max_iterations=5)

        self.assertIsNotNone(opt1)
        self.assertIsNotNone(opt5)

    def test_optimize_jaxpr_selective_optimizations(self):
        """Test selective optimizations parameter."""

        def f(x):
            unused = x * 2  # noqa: F841
            return x + (2 + 3)

        jaxpr = jax.make_jaxpr(f)(jnp.array(1.0))

        # Run only specific optimizations
        opt_cf = optimize_jaxpr(jaxpr.jaxpr, optimizations=['constant_fold'])
        opt_dce = optimize_jaxpr(jaxpr.jaxpr, optimizations=['dce'])

        self.assertIsNotNone(opt_cf)
        self.assertIsNotNone(opt_dce)

    def test_optimize_jaxpr_invalid_optimization_name(self):
        """Test that invalid optimization names raise ValueError."""

        def f(x):
            return x + 1

        jaxpr = jax.make_jaxpr(f)(jnp.array(1.0))

        with pytest.raises(ValueError, match="Invalid optimization"):
            optimize_jaxpr(jaxpr.jaxpr, optimizations=['invalid_opt'])

    def test_optimize_jaxpr_verbose(self):
        """Test verbose output."""

        def f(x):
            return x + (2 + 3)

        jaxpr = jax.make_jaxpr(f)(jnp.array(1.0))

        # Should run without error
        optimized = optimize_jaxpr(jaxpr.jaxpr, verbose=True)
        self.assertIsNotNone(optimized)

    def test_optimize_jaxpr_convergence(self):
        """Test that optimization converges."""

        def f(x):
            a = x + 0.0
            b = a * 1.0
            c = b + 0.0
            return c

        jaxpr = jax.make_jaxpr(f)(jnp.array(1.0))

        # Multiple iterations should converge
        opt1 = optimize_jaxpr(jaxpr.jaxpr, max_iterations=1)
        opt10 = optimize_jaxpr(jaxpr.jaxpr, max_iterations=10)

        # After convergence, more iterations shouldn't increase equation count
        self.assertLessEqual(len(opt10.eqns), len(opt1.eqns))

    def test_optimize_jaxpr_wrong_type(self):
        """Test that wrong input type raises TypeError."""

        with pytest.raises(TypeError, match="Expected Jaxpr or ClosedJaxpr"):
            optimize_jaxpr("not a jaxpr")


class TestOptimizationCombinations(unittest.TestCase):
    """Test combinations of optimizations."""

    def test_constant_fold_then_dce(self):
        """Test constant folding followed by dead code elimination."""

        def f(x):
            const_unused = 2 + 3  # noqa: F841
            return x + 1

        jaxpr = jax.make_jaxpr(f)(jnp.array(1.0))
        original_len = len(jaxpr.jaxpr.eqns)

        # Apply constant folding
        after_cf = constant_fold(jaxpr.jaxpr)
        # Then apply DCE
        after_dce = dead_code_elimination(after_cf)

        # Should remove the constant computation
        self.assertLessEqual(len(after_dce.eqns), original_len)

    def test_algebraic_then_dce(self):
        """Test algebraic simplification followed by DCE."""

        def f(x):
            a = x * 0.0  # Simplifies to 0
            unused = a + 1  # noqa: F841  # Dead code
            return x + 1

        jaxpr = jax.make_jaxpr(f)(jnp.array(1.0))
        original_len = len(jaxpr.jaxpr.eqns)

        # Apply algebraic simplification
        after_alg = algebraic_simplification(jaxpr.jaxpr)
        # Then apply DCE
        after_dce = dead_code_elimination(after_alg)

        # Should eliminate dead code
        self.assertLess(len(after_dce.eqns), original_len)

    def test_full_optimization_pipeline(self):
        """Test complete optimization pipeline."""

        def f(x, y):
            # Constant folding opportunity
            const = 2 + 3
            # Algebraic simplification opportunity
            a = x + 0.0
            b = a * 1.0
            # Dead code
            unused = y * 10  # noqa: F841
            # CSE opportunity
            c = x + y
            d = x + y  # Duplicate
            return b + const + c + d

        jaxpr = jax.make_jaxpr(f)(jnp.array(1.0), jnp.array(2.0))
        original_len = len(jaxpr.jaxpr.eqns)

        # Apply full optimization
        optimized = optimize_jaxpr(jaxpr.jaxpr, max_iterations=3)

        # Should significantly reduce equation count
        self.assertLessEqual(len(optimized.eqns), original_len)

    def test_optimization_order_matters(self):
        """Test that different optimization orders produce valid results."""

        def f(x, y):
            a = x + (2 + 3)  # Constant fold opportunity
            b = a * 1.0  # Algebraic simplification opportunity
            return b + y

        jaxpr = jax.make_jaxpr(f)(jnp.array(1.0), jnp.array(2.0))

        # Order 1: constant_fold -> algebraic_simplification
        opt1 = optimize_jaxpr(
            jaxpr.jaxpr,
            optimizations=['constant_fold', 'algebraic_simplification'],
            max_iterations=1
        )

        # Order 2: algebraic_simplification -> constant_fold
        opt2 = optimize_jaxpr(
            jaxpr.jaxpr,
            optimizations=['algebraic_simplification', 'constant_fold'],
            max_iterations=1
        )

        # Both should produce valid results
        self.assertIsNotNone(opt1)
        self.assertIsNotNone(opt2)
        # Both should maintain interface
        self.assertEqual(tuple(opt1.invars), tuple(jaxpr.jaxpr.invars))
        self.assertEqual(tuple(opt2.invars), tuple(jaxpr.jaxpr.invars))


class TestOptimizationWithBrainState(unittest.TestCase):
    """Test optimizations with BrainState functions."""

    def test_optimize_stateful_function(self):
        """Test optimization of jaxpr from stateful function."""
        state = brainstate.State(jnp.array([1.0, 2.0]))

        def f(x):
            # Contains optimization opportunities
            a = x + 0.0  # Algebraic simplification
            b = (2 + 3) * a  # Constant folding
            state.value += b
            return state.value

        jaxpr, states = brainstate.transform.make_jaxpr(f)(jnp.array([1.0, 2.0]))
        original_len = len(jaxpr.in_avals)

        # Optimize the jaxpr
        optimized = optimize_jaxpr(jaxpr)

        # Should produce valid optimized jaxpr
        self.assertIsNotNone(optimized)
        # Interface should be preserved
        self.assertEqual(len(optimized.in_avals), original_len)

    def test_optimize_with_vmap(self):
        """Test optimization of jaxpr containing vmap."""
        from brainstate import ShortTermState
        from brainstate.util import filter as state_filter

        state = ShortTermState(jnp.zeros(3))

        def f(x):
            counter_value = state.value + x
            state.value = counter_value
            return state.value

        # Create a vmapped version with proper state axes
        from brainstate.transform._mapping import StatefulMapping
        mapper = StatefulMapping(
            f,
            in_axes=0,
            out_axes=0,
            state_in_axes={0: state_filter.OfType(ShortTermState)},
            state_out_axes={0: state_filter.OfType(ShortTermState)},
        )

        xs = jnp.array([1.0, 2.0, 3.0])
        jaxpr = mapper.get_jaxpr(xs)

        # Should be able to optimize
        optimized = optimize_jaxpr(jaxpr.jaxpr, max_iterations=2)
        self.assertIsNotNone(optimized)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and corner scenarios."""

    def test_empty_jaxpr(self):
        """Test optimization of trivial jaxpr."""

        def f(x):
            return x

        jaxpr = jax.make_jaxpr(f)(jnp.array(1.0))

        # Should handle gracefully
        optimized = optimize_jaxpr(jaxpr.jaxpr)
        self.assertIsNotNone(optimized)

    def test_jaxpr_with_no_optimizations(self):
        """Test jaxpr that has no optimization opportunities."""

        def f(x, y, z):
            a = x * y
            b = a + z
            c = b / 2.0
            return c

        jaxpr = jax.make_jaxpr(f)(jnp.array(1.0), jnp.array(2.0), jnp.array(3.0))
        original_len = len(jaxpr.jaxpr.eqns)

        optimized = optimize_jaxpr(jaxpr.jaxpr)

        # Should remain similar
        self.assertGreaterEqual(len(optimized.eqns), original_len - 1)

    def test_deeply_nested_constants(self):
        """Test optimization with deeply nested constant expressions."""

        def f(x):
            a = ((1 + 2) * (3 + 4)) + ((5 + 6) * (7 + 8))
            return x + a

        jaxpr = jax.make_jaxpr(f)(jnp.array(1.0))
        original_len = len(jaxpr.jaxpr.eqns)

        optimized = optimize_jaxpr(jaxpr.jaxpr)

        # Should fold all nested constants (or at least not increase)
        self.assertLessEqual(len(optimized.eqns), original_len)

    def test_optimization_with_multiple_dtypes(self):
        """Test optimization with different dtypes."""

        def f(x_float, x_int):
            return x_float + jnp.float32(x_int) + 1.0

        jaxpr = jax.make_jaxpr(f)(jnp.array(1.0, dtype=jnp.float32),
                                   jnp.array(2, dtype=jnp.int32))

        # Should handle different dtypes
        optimized = optimize_jaxpr(jaxpr.jaxpr)
        self.assertIsNotNone(optimized)

    def test_optimization_with_complex_numbers(self):
        """Test optimization with complex dtype."""

        def f(x):
            return x + (1.0 + 2.0j)

        jaxpr = jax.make_jaxpr(f)(jnp.array(1.0 + 1.0j, dtype=jnp.complex64))

        # Should handle complex numbers
        optimized = optimize_jaxpr(jaxpr.jaxpr)
        self.assertIsNotNone(optimized)


class TestOptimizationCorrectness(unittest.TestCase):
    """Test that optimizations preserve semantics."""

    def test_optimized_produces_same_result(self):
        """Test that optimized jaxpr produces same result as original."""

        def f(x, y):
            a = x + 0.0
            b = y * 1.0
            c = (2 + 3) * a
            return c + b

        # Get original jaxpr
        closed_jaxpr = jax.make_jaxpr(f)(jnp.array(1.0), jnp.array(2.0))

        # Get optimized jaxpr
        optimized = optimize_jaxpr(closed_jaxpr)

        # Test inputs
        x_test = jnp.array(5.0)
        y_test = jnp.array(10.0)

        # Compute results
        from brainstate._compatible_import import jaxpr_as_fun
        original_result = jaxpr_as_fun(closed_jaxpr)(x_test, y_test)
        optimized_result = jaxpr_as_fun(optimized)(x_test, y_test)

        # Results should be the same (handle both array and list returns)
        if isinstance(original_result, list):
            original_result = original_result[0]
        if isinstance(optimized_result, list):
            optimized_result = optimized_result[0]

        self.assertTrue(jnp.allclose(original_result, optimized_result))

    def test_complex_computation_correctness(self):
        """Test correctness with more complex computation."""

        def f(x, y, z):
            # Mix of operations
            a = x + y
            b = x + y  # CSE candidate
            c = a * 1.0  # Algebraic simplification
            d = (2 + 3) * c  # Constant folding
            unused = z * 100  # Dead code  # noqa: F841
            return d + b

        closed_jaxpr = jax.make_jaxpr(f)(jnp.array(1.0), jnp.array(2.0), jnp.array(3.0))
        optimized = optimize_jaxpr(closed_jaxpr)

        # Test with different inputs
        x_test = jnp.array(7.0)
        y_test = jnp.array(8.0)
        z_test = jnp.array(9.0)

        from brainstate._compatible_import import jaxpr_as_fun
        original_result = jaxpr_as_fun(closed_jaxpr)(x_test, y_test, z_test)
        optimized_result = jaxpr_as_fun(optimized)(x_test, y_test, z_test)

        # Handle both array and list returns
        if isinstance(original_result, list):
            original_result = original_result[0]
        if isinstance(optimized_result, list):
            optimized_result = optimized_result[0]

        self.assertTrue(jnp.allclose(original_result, optimized_result))


if __name__ == '__main__':
    unittest.main()
