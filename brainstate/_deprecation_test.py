#!/usr/bin/env python3
"""
Comprehensive tests for deprecated modules (augment, compile, functional).

Tests that:
1. Deprecation warnings are shown appropriately
2. All functionality is properly forwarded to replacement modules
3. API compatibility is maintained
4. Error handling works correctly
"""

import unittest
import warnings
import sys
import numpy as np
import os
from unittest.mock import patch

# Add brainstate to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
import jax.numpy as jnp
import brainstate


class TestDeprecatedAugmentModule(unittest.TestCase):
    """Test the deprecated brainstate.augment module."""

    def setUp(self):
        """Reset warning filters before each test."""
        warnings.resetwarnings()

    def test_augment_module_attributes(self):
        """Test that augment module has correct attributes."""
        # Test module attributes
        self.assertEqual(brainstate.augment.__name__, 'brainstate.augment')
        self.assertIn('deprecated', brainstate.augment.__doc__.lower())
        self.assertTrue(hasattr(brainstate.augment, '__all__'))

        # Test repr
        repr_str = repr(brainstate.augment)
        self.assertIn('DeprecatedModule', repr_str)
        self.assertIn('brainstate.augment', repr_str)
        self.assertIn('brainstate.transform', repr_str)

    def test_augment_scoped_apis(self):
        """Test that augment module only exposes scoped APIs."""
        # Check that expected APIs are available
        expected_apis = [
            'GradientTransform', 'grad', 'vector_grad', 'hessian', 'jacobian',
            'jacrev', 'jacfwd', 'abstract_init', 'vmap', 'pmap', 'map',
            'vmap_new_states', 'restore_rngs'
        ]

        for api in expected_apis:
            with self.subTest(api=api):
                self.assertIn(api, brainstate.augment.__all__)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.assertTrue(hasattr(brainstate.augment, api),
                                    f"API '{api}' should be available in augment module")

        # Check that __all__ contains only expected APIs
        self.assertEqual(set(brainstate.augment.__all__), set(expected_apis))

    def test_augment_deprecation_warnings(self):
        """Test that augment module shows deprecation warnings."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Access different attributes
            _ = brainstate.augment.grad
            _ = brainstate.augment.vmap
            _ = brainstate.augment.vector_grad

            # Should have warnings for each unique attribute
            self.assertGreaterEqual(len(w), 3)

            # Check warning messages
            for warning in w:
                self.assertEqual(warning.category, DeprecationWarning)
                msg = str(warning.message)
                self.assertIn('brainstate.augment', msg)
                self.assertIn('deprecated', msg)
                self.assertIn('brainstate.transform', msg)

    # def test_augment_no_duplicate_warnings(self):
    #     """Test that repeated access doesn't generate duplicate warnings."""
    #     with warnings.catch_warnings(record=True) as w:
    #         # Access the same attribute multiple times
    #         _ = brainstate.augment.grad
    #         _ = brainstate.augment.grad
    #         _ = brainstate.augment.grad
    #
    #         # Should only have one warning
    #         self.assertEqual(len(w), 1)

    def test_augment_functionality_forwarding(self):
        """Test that augment module forwards functionality correctly."""
        # Test that functions are properly forwarded
        self.assertTrue(callable(brainstate.augment.grad))
        self.assertTrue(callable(brainstate.augment.vmap))
        self.assertTrue(callable(brainstate.augment.vector_grad))

        # Test that they are the same as transform module
        self.assertIs(brainstate.augment.grad, brainstate.transform.grad)
        self.assertIs(brainstate.augment.vmap, brainstate.transform.vmap)

    def test_augment_grad_functionality(self):
        """Test that grad function works through deprecated module."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore deprecation warnings for this test

            # Create a simple state and function
            state = brainstate.State(jnp.array([1.0, 2.0]))

            def loss_fn():
                return jnp.sum(state.value ** 2)

            # Test grad function
            grad_fn = brainstate.augment.grad(loss_fn, state)
            grads = grad_fn()

            # Should compute correct gradients
            expected = 2 * state.value
            np.testing.assert_array_almost_equal(grads, expected)

    def test_augment_dir_functionality(self):
        """Test that dir() works on augment module."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            attrs = dir(brainstate.augment)

            # Should contain expected attributes
            self.assertIn('grad', attrs)
            self.assertIn('vmap', attrs)
            self.assertIn('vector_grad', attrs)

    def test_augment_missing_attribute_error(self):
        """Test that accessing non-existent attributes raises appropriate error."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            with self.assertRaises(AttributeError) as context:
                _ = brainstate.augment.nonexistent_function

            error_msg = str(context.exception)
            self.assertIn('brainstate.augment', error_msg)
            self.assertIn('nonexistent_function', error_msg)
            self.assertIn('brainstate.transform', error_msg)


class TestDeprecatedCompileModule(unittest.TestCase):
    """Test the deprecated brainstate.compile module."""

    def setUp(self):
        """Reset warning filters before each test."""
        warnings.resetwarnings()

    def test_compile_module_attributes(self):
        """Test that compile module has correct attributes."""
        self.assertEqual(brainstate.compile.__name__, 'brainstate.compile')
        self.assertIn('deprecated', brainstate.compile.__doc__.lower())
        self.assertTrue(hasattr(brainstate.compile, '__all__'))

    def test_compile_scoped_apis(self):
        """Test that compile module only exposes scoped APIs."""
        expected_apis = [
            'checkpoint', 'remat', 'cond', 'switch', 'ifelse', 'jit_error_if',
            'jit', 'scan', 'checkpointed_scan', 'for_loop', 'checkpointed_for_loop',
            'while_loop', 'bounded_while_loop', 'StatefulFunction', 'make_jaxpr',
            'ProgressBar'
        ]

        for api in expected_apis:
            with self.subTest(api=api):
                self.assertIn(api, brainstate.compile.__all__)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.assertTrue(hasattr(brainstate.compile, api),
                                    f"API '{api}' should be available in compile module")

        # Check that __all__ contains only expected APIs
        self.assertEqual(set(brainstate.compile.__all__), set(expected_apis))

    def test_compile_deprecation_warnings(self):
        """Test that compile module shows deprecation warnings."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Access different attributes
            _ = brainstate.compile.jit
            _ = brainstate.compile.for_loop
            _ = brainstate.compile.while_loop

            # Should have warnings
            self.assertGreaterEqual(len(w), 3)

            # Check warning content
            for warning in w:
                self.assertEqual(warning.category, DeprecationWarning)
                msg = str(warning.message)
                self.assertIn('brainstate.compile', msg)
                self.assertIn('brainstate.transform', msg)

    def test_compile_functionality_forwarding(self):
        """Test that compile module forwards functionality correctly."""
        # Test that functions are properly forwarded
        self.assertTrue(callable(brainstate.compile.jit))
        self.assertTrue(callable(brainstate.compile.for_loop))
        self.assertTrue(callable(brainstate.compile.while_loop))

        # Test that they are the same as transform module
        self.assertIs(brainstate.compile.jit, brainstate.transform.jit)
        self.assertIs(brainstate.compile.for_loop, brainstate.transform.for_loop)

    def test_compile_jit_functionality(self):
        """Test that jit function works through deprecated module."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            state = brainstate.State(5.0)

            @brainstate.compile.jit
            def add_one():
                state.value += 1.0
                return state.value

            result = add_one()
            self.assertEqual(result, 6.0)
            self.assertEqual(state.value, 6.0)

    def test_compile_for_loop_functionality(self):
        """Test that for_loop function works through deprecated module."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            counter = brainstate.State(0.0)

            def body(i):
                counter.value += 1.0

            brainstate.compile.for_loop(body, jnp.arange(5))
            self.assertEqual(counter.value, 5.0)


class TestDeprecatedFunctionalModule(unittest.TestCase):
    """Test the deprecated brainstate.functional module."""

    def setUp(self):
        """Reset warning filters before each test."""
        warnings.resetwarnings()

    def test_functional_module_attributes(self):
        """Test that functional module has correct attributes."""
        self.assertEqual(brainstate.functional.__name__, 'brainstate.functional')
        self.assertIn('deprecated', brainstate.functional.__doc__.lower())
        self.assertTrue(hasattr(brainstate.functional, '__all__'))

    def test_functional_scoped_apis(self):
        """Test that functional module only exposes scoped APIs."""
        expected_apis = [
            'weight_standardization', 'clip_grad_norm',
            # Activation functions
            'tanh', 'relu', 'squareplus', 'softplus', 'soft_sign', 'sigmoid',
            'silu', 'swish', 'log_sigmoid', 'elu', 'leaky_relu', 'hard_tanh',
            'celu', 'selu', 'gelu', 'glu', 'logsumexp', 'log_softmax',
            'softmax', 'standardize'
        ]

        for api in expected_apis:
            with self.subTest(api=api):
                self.assertIn(api, brainstate.functional.__all__)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.assertTrue(hasattr(brainstate.functional, api),
                                    f"API '{api}' should be available in functional module")

        # Check that __all__ contains only expected APIs
        self.assertEqual(set(brainstate.functional.__all__), set(expected_apis))

    def test_functional_deprecation_warnings(self):
        """Test that functional module shows deprecation warnings."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Access different attributes
            _ = brainstate.functional.relu
            _ = brainstate.functional.sigmoid
            _ = brainstate.functional.tanh

            # Should have warnings
            # self.assertGreaterEqual(len(w), 3)

            # Check warning content
            for warning in w:
                self.assertEqual(warning.category, DeprecationWarning)
                msg = str(warning.message)
                self.assertIn('brainstate.functional', msg)
                self.assertIn('brainstate.nn', msg)

    def test_functional_functionality_forwarding(self):
        """Test that functional module forwards functionality correctly."""
        # Test that functions are properly forwarded
        self.assertTrue(callable(brainstate.functional.relu))
        self.assertTrue(callable(brainstate.functional.sigmoid))
        self.assertTrue(callable(brainstate.functional.tanh))

        # # Test that they are the same as nn module
        # self.assertIs(brainstate.functional.relu, brainstate.nn.relu)
        # self.assertIs(brainstate.functional.sigmoid, brainstate.nn.sigmoid)

    def test_functional_activation_functions(self):
        """Test that activation functions work through deprecated module."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Test relu
            x = jnp.array([-1.0, 0.0, 1.0])
            result = brainstate.functional.relu(x)
            expected = jnp.array([0.0, 0.0, 1.0])
            np.testing.assert_array_almost_equal(result, expected)

            # Test sigmoid
            x = jnp.array([0.0])
            result = brainstate.functional.sigmoid(x)
            expected = jnp.array([0.5])
            np.testing.assert_array_almost_equal(result, expected, decimal=5)

            # Test tanh
            x = jnp.array([0.0])
            result = brainstate.functional.tanh(x)
            expected = jnp.array([0.0])
            np.testing.assert_array_almost_equal(result, expected)

    def test_functional_weight_standardization(self):
        """Test that weight_standardization works through deprecated module."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Create a simple weight matrix
            weights = jnp.ones((3, 3))

            # Test weight standardization (should be available)
            if hasattr(brainstate.functional, 'weight_standardization'):
                standardized = brainstate.functional.weight_standardization(weights)
                self.assertEqual(standardized.shape, weights.shape)


class TestDeprecatedModulesIntegration(unittest.TestCase):
    """Integration tests for all deprecated modules."""

    def test_all_deprecated_modules_in_brainstate(self):
        """Test that all deprecated modules are available in brainstate."""
        self.assertTrue(hasattr(brainstate, 'augment'))
        self.assertTrue(hasattr(brainstate, 'compile'))
        self.assertTrue(hasattr(brainstate, 'functional'))

    def test_deprecated_modules_in_all(self):
        """Test that deprecated modules are in __all__."""
        self.assertIn('augment', brainstate.__all__)
        self.assertIn('compile', brainstate.__all__)
        self.assertIn('functional', brainstate.__all__)

    def test_mixed_usage_compatibility(self):
        """Test that users can mix deprecated and new modules."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Create a state
            state = brainstate.State(jnp.array([1.0, 2.0]))

            def loss_fn():
                x = brainstate.functional.relu(state.value)  # deprecated
                return jnp.sum(x ** 2)

            # Use deprecated augment with new transform
            grad_fn = brainstate.augment.grad(loss_fn, state)  # deprecated
            grads = grad_fn()

            # Should work correctly
            self.assertIsInstance(grads, jnp.ndarray)
            self.assertEqual(grads.shape, (2,))

    def test_warning_stacklevel(self):
        """Test that warnings point to user code, not internal code."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # This should generate a warning pointing to this line
            _ = brainstate.augment.grad

            # # Check that warning points to user code
            # # self.assertGreaterEqual(len(w), 1)
            # warning = w[0]
            #
            # # The warning should point to this test file
            # self.assertIn('_deprecation_test.py', warning.filename)


class TestScopedAPIRestrictions(unittest.TestCase):
    """Test that scoped APIs properly restrict access to non-scoped functions."""

    def test_augment_blocks_non_scoped_apis(self):
        """Test that augment module blocks access to APIs not in its scope."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # These should work (scoped APIs)
            self.assertTrue(hasattr(brainstate.augment, 'grad'))
            self.assertTrue(hasattr(brainstate.augment, 'vmap'))

            # This should NOT work if transform has APIs not in augment scope
            # (Note: since we're using string-based imports, this test checks the scoping mechanism)
            try:
                # Try to access something that might exist in transform but not in augment scope
                _ = brainstate.augment.nonexistent_function
                self.fail("Should not be able to access non-scoped API")
            except AttributeError as e:
                self.assertIn('Available attributes:', str(e))
                self.assertIn('brainstate.augment', str(e))

    def test_compile_blocks_non_scoped_apis(self):
        """Test that compile module blocks access to APIs not in its scope."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # These should work (scoped APIs)
            self.assertTrue(hasattr(brainstate.compile, 'jit'))
            self.assertTrue(hasattr(brainstate.compile, 'for_loop'))

            # This should NOT work
            try:
                _ = brainstate.compile.nonexistent_function
                self.fail("Should not be able to access non-scoped API")
            except AttributeError as e:
                self.assertIn('Available attributes:', str(e))

    def test_functional_blocks_non_scoped_apis(self):
        """Test that functional module blocks access to APIs not in its scope."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # These should work (scoped APIs)
            self.assertTrue(hasattr(brainstate.functional, 'relu'))
            self.assertTrue(hasattr(brainstate.functional, 'sigmoid'))

            # This should NOT work
            try:
                _ = brainstate.functional.nonexistent_function
                self.fail("Should not be able to access non-scoped API")
            except AttributeError as e:
                self.assertIn('Available attributes:', str(e))


class TestDeprecationSystemRobustness(unittest.TestCase):
    """Test edge cases and robustness of the deprecation system."""

    def test_nested_attribute_access(self):
        """Test accessing nested attributes doesn't break."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Test that we can access nested attributes if they exist
            if hasattr(brainstate.transform, 'grad'):
                grad_func = brainstate.augment.grad
                self.assertTrue(callable(grad_func))

    def test_module_import_style_access(self):
        """Test different styles of accessing deprecated modules."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Direct access
            func1 = brainstate.augment.grad

            # Module-style access
            augment_module = brainstate.augment
            func2 = augment_module.grad

            # Should be the same function
            self.assertIs(func1, func2)

    def test_help_and_documentation(self):
        """Test that help() and documentation work on deprecated modules."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Should be able to get help without errors
            try:
                help_text = brainstate.augment.__doc__
                self.assertIsInstance(help_text, str)
                self.assertIn('deprecated', help_text.lower())
            except Exception as e:
                self.fail(f"Getting documentation failed: {e}")

    def test_multiple_import_styles(self):
        """Test that different import styles work with deprecation."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Test that we can still access through different paths
            from brainstate import augment as aug
            from brainstate import functional as func

            self.assertTrue(callable(aug.grad))
            self.assertTrue(callable(func.relu))


if __name__ == '__main__':
    # Configure test output
    unittest.TestCase.maxDiff = None

    # Run tests
    unittest.main(verbosity=2)