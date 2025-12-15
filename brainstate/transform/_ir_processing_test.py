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

from brainstate._compatible_import import Jaxpr, ClosedJaxpr
from brainstate.transform._ir_processing import eqns_to_jaxpr, eqns_to_closed_jaxpr


class TestEqnsToJaxpr(unittest.TestCase):
    """Test eqns_to_jaxpr function."""

    def test_basic_conversion(self):
        """Test basic conversion of equations to Jaxpr."""
        # Create a simple jaxpr
        original_jaxpr = jax.make_jaxpr(lambda x, y: x + y)(1.0, 2.0)
        eqns = original_jaxpr.jaxpr.eqns

        # Convert equations back to Jaxpr
        reconstructed = eqns_to_jaxpr(eqns)

        # Verify it's a Jaxpr object
        self.assertIsInstance(reconstructed, Jaxpr)

        # Verify equations match
        self.assertEqual(len(reconstructed.eqns), len(eqns))
        self.assertEqual(reconstructed.eqns, list(eqns))

    def test_with_explicit_invars(self):
        """Test conversion with explicitly provided invars."""
        original_jaxpr = jax.make_jaxpr(lambda x, y: x + y)(1.0, 2.0)
        eqns = original_jaxpr.jaxpr.eqns
        invars = original_jaxpr.jaxpr.invars

        reconstructed = eqns_to_jaxpr(eqns, invars=invars)

        # Verify invars match
        self.assertEqual(reconstructed.invars, invars)

    def test_with_explicit_outvars(self):
        """Test conversion with explicitly provided outvars."""
        original_jaxpr = jax.make_jaxpr(lambda x, y: x + y)(1.0, 2.0)
        eqns = original_jaxpr.jaxpr.eqns
        outvars = original_jaxpr.jaxpr.outvars

        reconstructed = eqns_to_jaxpr(eqns, outvars=outvars)

        # Verify outvars match
        self.assertEqual(reconstructed.outvars, outvars)

    def test_with_all_explicit_vars(self):
        """Test conversion with all variables explicitly provided."""
        original_jaxpr = jax.make_jaxpr(lambda x, y: x + y)(1.0, 2.0)
        eqns = original_jaxpr.jaxpr.eqns
        invars = original_jaxpr.jaxpr.invars
        outvars = original_jaxpr.jaxpr.outvars
        constvars = original_jaxpr.jaxpr.constvars

        reconstructed = eqns_to_jaxpr(
            eqns,
            invars=invars,
            outvars=outvars,
            constvars=constvars
        )

        # Verify all variables match
        self.assertEqual(reconstructed.invars, invars)
        self.assertEqual(reconstructed.outvars, outvars)
        self.assertEqual(reconstructed.constvars, list(constvars))

    def test_multiple_operations(self):
        """Test conversion with multiple operations."""
        # Create a more complex jaxpr
        original_jaxpr = jax.make_jaxpr(lambda x, y: x * 2 + y * 3)(1.0, 2.0)
        eqns = original_jaxpr.jaxpr.eqns

        reconstructed = eqns_to_jaxpr(eqns)

        # Verify multiple equations are preserved
        self.assertEqual(len(reconstructed.eqns), len(eqns))
        self.assertGreater(len(reconstructed.eqns), 1)

    def test_empty_equations(self):
        """Test conversion with empty equations list."""
        reconstructed = eqns_to_jaxpr([])

        # Verify empty jaxpr is created
        self.assertEqual(len(reconstructed.eqns), 0)
        self.assertEqual(len(reconstructed.invars), 0)
        self.assertEqual(len(reconstructed.outvars), 0)
        self.assertEqual(len(reconstructed.constvars), 0)

    def test_infer_invars_from_equations(self):
        """Test automatic inference of invars from equations."""
        # Create jaxpr and use it to test inference
        original_jaxpr = jax.make_jaxpr(lambda x, y: x + y)(1.0, 2.0)
        eqns = original_jaxpr.jaxpr.eqns

        # Don't provide invars - let it be inferred
        reconstructed = eqns_to_jaxpr(eqns)

        # Should have inferred input variables
        self.assertGreater(len(reconstructed.invars), 0)

    def test_infer_outvars_from_last_equation(self):
        """Test automatic inference of outvars from last equation."""
        original_jaxpr = jax.make_jaxpr(lambda x, y: x + y)(1.0, 2.0)
        eqns = original_jaxpr.jaxpr.eqns

        # Don't provide outvars - let it be inferred
        reconstructed = eqns_to_jaxpr(eqns)

        # Should use last equation's output
        self.assertEqual(len(reconstructed.outvars), len(eqns[-1].outvars))
        self.assertEqual(reconstructed.outvars, list(eqns[-1].outvars))

    def test_with_constvars(self):
        """Test conversion with constant variables."""
        # Create jaxpr with constants
        const_val = jnp.array([1.0, 2.0, 3.0])
        original_jaxpr = jax.make_jaxpr(lambda x: x + const_val)(jnp.array([1.0, 2.0, 3.0]))
        eqns = original_jaxpr.jaxpr.eqns
        constvars = original_jaxpr.jaxpr.constvars

        reconstructed = eqns_to_jaxpr(eqns, constvars=constvars)

        # Verify constvars match
        self.assertEqual(reconstructed.constvars, list(constvars))

    def test_complex_function(self):
        """Test conversion with a complex function."""
        def complex_func(x, y, z):
            a = x * y
            b = y + z
            c = a - b
            return c * 2

        original_jaxpr = jax.make_jaxpr(complex_func)(1.0, 2.0, 3.0)
        eqns = original_jaxpr.jaxpr.eqns

        reconstructed = eqns_to_jaxpr(
            eqns,
            invars=original_jaxpr.jaxpr.invars,
            outvars=original_jaxpr.jaxpr.outvars
        )

        # Verify structure
        self.assertEqual(len(reconstructed.eqns), len(eqns))
        self.assertEqual(len(reconstructed.invars), 3)
        self.assertEqual(len(reconstructed.outvars), 1)


class TestEqnsToClosedJaxpr(unittest.TestCase):
    """Test eqns_to_closed_jaxpr function."""

    def test_basic_conversion(self):
        """Test basic conversion of equations to ClosedJaxpr."""
        original_jaxpr = jax.make_jaxpr(lambda x, y: x + y)(1.0, 2.0)
        eqns = original_jaxpr.jaxpr.eqns

        reconstructed = eqns_to_closed_jaxpr(eqns)

        # Verify it's a ClosedJaxpr object
        self.assertIsInstance(reconstructed, ClosedJaxpr)
        self.assertIsInstance(reconstructed.jaxpr, Jaxpr)

    def test_with_explicit_vars(self):
        """Test conversion with explicitly provided variables."""
        original_jaxpr = jax.make_jaxpr(lambda x, y: x * 2 + y)(1.0, 2.0)
        eqns = original_jaxpr.jaxpr.eqns
        invars = original_jaxpr.jaxpr.invars
        outvars = original_jaxpr.jaxpr.outvars

        reconstructed = eqns_to_closed_jaxpr(
            eqns,
            invars=invars,
            outvars=outvars
        )

        # Verify variables match
        self.assertEqual(reconstructed.jaxpr.invars, invars)
        self.assertEqual(reconstructed.jaxpr.outvars, outvars)

    def test_with_consts(self):
        """Test conversion with constants."""
        const_val = jnp.array([1.0, 2.0, 3.0])
        original_jaxpr = jax.make_jaxpr(lambda x: x + const_val)(jnp.array([1.0, 2.0, 3.0]))
        eqns = original_jaxpr.jaxpr.eqns
        constvars = original_jaxpr.jaxpr.constvars
        consts = original_jaxpr.consts

        reconstructed = eqns_to_closed_jaxpr(
            eqns,
            invars=original_jaxpr.jaxpr.invars,
            outvars=original_jaxpr.jaxpr.outvars,
            constvars=constvars,
            consts=consts
        )

        # Verify consts match
        self.assertEqual(len(reconstructed.consts), len(consts))
        self.assertEqual(reconstructed.jaxpr.constvars, list(constvars))

    def test_empty_consts(self):
        """Test conversion with empty consts."""
        original_jaxpr = jax.make_jaxpr(lambda x, y: x + y)(1.0, 2.0)
        eqns = original_jaxpr.jaxpr.eqns

        reconstructed = eqns_to_closed_jaxpr(eqns)

        # Verify empty consts
        self.assertEqual(len(reconstructed.consts), 0)

    def test_execution_compatibility(self):
        """Test that reconstructed ClosedJaxpr can be executed."""
        # Create original function and jaxpr
        def test_func(x, y):
            return x * 2 + y * 3

        original_jaxpr = jax.make_jaxpr(test_func)(1.0, 2.0)
        eqns = original_jaxpr.jaxpr.eqns

        # Reconstruct ClosedJaxpr
        reconstructed = eqns_to_closed_jaxpr(
            eqns,
            invars=original_jaxpr.jaxpr.invars,
            outvars=original_jaxpr.jaxpr.outvars,
            constvars=original_jaxpr.jaxpr.constvars,
            consts=original_jaxpr.consts
        )

        # Execute both and compare results
        from jax._src.core import jaxpr_as_fun
        original_result = jaxpr_as_fun(original_jaxpr)(1.0, 2.0)
        reconstructed_result = jaxpr_as_fun(reconstructed)(1.0, 2.0)

        # Results should match (convert to arrays if needed)
        original_arr = jnp.array(original_result) if isinstance(original_result, list) else original_result
        reconstructed_arr = jnp.array(reconstructed_result) if isinstance(reconstructed_result, list) else reconstructed_result
        self.assertTrue(jnp.allclose(original_arr, reconstructed_arr))

    def test_multiple_consts(self):
        """Test conversion with multiple constants."""
        const1 = jnp.array([1.0, 2.0])
        const2 = jnp.array([3.0, 4.0])

        def func(x):
            return x + const1 + const2

        original_jaxpr = jax.make_jaxpr(func)(jnp.array([5.0, 6.0]))
        eqns = original_jaxpr.jaxpr.eqns

        reconstructed = eqns_to_closed_jaxpr(
            eqns,
            invars=original_jaxpr.jaxpr.invars,
            outvars=original_jaxpr.jaxpr.outvars,
            constvars=original_jaxpr.jaxpr.constvars,
            consts=original_jaxpr.consts
        )

        # Verify multiple consts are preserved
        self.assertEqual(len(reconstructed.consts), len(original_jaxpr.consts))

    def test_nested_operations(self):
        """Test conversion with nested operations."""
        def nested_func(x, y):
            a = jnp.sin(x)
            b = jnp.cos(y)
            c = a * b
            d = jnp.exp(c)
            return d + 1.0

        original_jaxpr = jax.make_jaxpr(nested_func)(1.0, 2.0)
        eqns = original_jaxpr.jaxpr.eqns

        reconstructed = eqns_to_closed_jaxpr(
            eqns,
            invars=original_jaxpr.jaxpr.invars,
            outvars=original_jaxpr.jaxpr.outvars,
            constvars=original_jaxpr.jaxpr.constvars,
            consts=original_jaxpr.consts
        )

        # Verify complex structure is preserved
        self.assertEqual(len(reconstructed.jaxpr.eqns), len(eqns))
        self.assertGreater(len(reconstructed.jaxpr.eqns), 3)

    def test_array_operations(self):
        """Test conversion with array operations."""
        def array_func(x, y):
            return jnp.dot(x, y)

        x_val = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        y_val = jnp.array([[5.0, 6.0], [7.0, 8.0]])

        original_jaxpr = jax.make_jaxpr(array_func)(x_val, y_val)
        eqns = original_jaxpr.jaxpr.eqns

        reconstructed = eqns_to_closed_jaxpr(
            eqns,
            invars=original_jaxpr.jaxpr.invars,
            outvars=original_jaxpr.jaxpr.outvars
        )

        # Verify array operations work
        from jax._src.core import jaxpr_as_fun
        original_result = jaxpr_as_fun(original_jaxpr)(x_val, y_val)
        reconstructed_result = jaxpr_as_fun(reconstructed)(x_val, y_val)

        # Convert results to arrays if needed
        original_arr = jnp.array(original_result) if isinstance(original_result, list) else original_result
        reconstructed_arr = jnp.array(reconstructed_result) if isinstance(reconstructed_result, list) else reconstructed_result
        self.assertTrue(jnp.allclose(original_arr, reconstructed_arr))


if __name__ == '__main__':
    unittest.main()
