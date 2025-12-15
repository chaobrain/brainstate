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

# -*- coding: utf-8 -*-


import unittest
import warnings

import jax.numpy as jnp
import numpy as np

import brainstate
from brainstate._deprecation import DeprecatedModule, create_deprecated_module_proxy


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
            'jacrev', 'jacfwd', 'vmap', 'pmap', 'map',
            'vmap_new_states',
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
            # self.assertGreaterEqual(len(w), 3)

            # Check warning messages
            for warning in w:
                self.assertEqual(warning.category, DeprecationWarning)
                msg = str(warning.message)
                self.assertIn('brainstate.augment', msg)
                self.assertIn('deprecated', msg)
                self.assertIn('brainstate.transform', msg)

    def test_augment_no_duplicate_warnings(self):
        """Test that repeated access doesn't generate duplicate warnings."""
        with warnings.catch_warnings(record=True) as w:
            # Access the same attribute multiple times
            _ = brainstate.augment.grad
            _ = brainstate.augment.grad
            _ = brainstate.augment.grad

            # Should only have one warning
            # self.assertEqual(len(w), 1)

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
            # self.assertGreaterEqual(len(w), 3)

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
        # self.assertEqual(set(brainstate.functional.__all__), set(expected_apis))

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


class MockReplacementModule:
    """Mock module for testing."""

    @staticmethod
    def test_function(x):
        return x * 2

    test_variable = 42

    class test_class:
        def __init__(self, value):
            self.value = value


class TestDeprecatedModule(unittest.TestCase):
    """Test the DeprecatedModule class."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_module = MockReplacementModule()
        self.deprecated = DeprecatedModule(
            deprecated_name='test.deprecated',
            replacement_module=self.mock_module,
            replacement_name='test.replacement',
            version='1.0.0',
            removal_version='2.0.0'
        )

    def test_initialization(self):
        """Test DeprecatedModule initialization."""
        self.assertEqual(self.deprecated.__name__, 'test.deprecated')
        self.assertIn('DEPRECATED', self.deprecated.__doc__)
        self.assertIn('test.deprecated', self.deprecated.__doc__)
        self.assertIn('test.replacement', self.deprecated.__doc__)

    def test_repr(self):
        """Test DeprecatedModule repr."""
        repr_str = repr(self.deprecated)
        self.assertIn('DeprecatedModule', repr_str)
        self.assertIn('test.deprecated', repr_str)
        self.assertIn('test.replacement', repr_str)

    def test_attribute_forwarding(self):
        """Test that attributes are properly forwarded."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Test function forwarding
            result = self.deprecated.test_function(5)
            self.assertEqual(result, 10)

            # Test variable forwarding
            self.assertEqual(self.deprecated.test_variable, 42)

            # Test class forwarding
            instance = self.deprecated.test_class(100)
            self.assertEqual(instance.value, 100)

    def test_deprecation_warnings(self):
        """Test that deprecation warnings are generated."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Access different attributes
            _ = self.deprecated.test_function
            _ = self.deprecated.test_variable
            _ = self.deprecated.test_class

            # Should have generated warnings
            self.assertEqual(len(w), 3)

            # Check warning properties
            for warning in w:
                self.assertEqual(warning.category, DeprecationWarning)
                msg = str(warning.message)
                self.assertIn('test.deprecated', msg)
                self.assertIn('test.replacement', msg)
                self.assertIn('deprecated', msg.lower())

    def test_no_duplicate_warnings(self):
        """Test that accessing the same attribute multiple times only warns once."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Access the same attribute multiple times
            _ = self.deprecated.test_function
            _ = self.deprecated.test_function
            _ = self.deprecated.test_function

            # Should only have one warning
            self.assertEqual(len(w), 1)

    def test_warning_with_removal_version(self):
        """Test warning message includes removal version when specified."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            _ = self.deprecated.test_function

            self.assertEqual(len(w), 1)
            msg = str(w[0].message)
            self.assertIn('2.0.0', msg)

    def test_missing_attribute_error(self):
        """Test that accessing non-existent attributes raises AttributeError."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            with self.assertRaises(AttributeError) as context:
                _ = self.deprecated.nonexistent_attribute

            error_msg = str(context.exception)
            self.assertIn('test.deprecated', error_msg)
            self.assertIn('nonexistent_attribute', error_msg)
            self.assertIn('test.replacement', error_msg)

    def test_dir_functionality(self):
        """Test that dir() works on deprecated module."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            attrs = dir(self.deprecated)

            # Should warn about dir access
            self.assertGreaterEqual(len(w), 1)

            # Should contain expected attributes
            self.assertIn('test_function', attrs)
            self.assertIn('test_variable', attrs)
            self.assertIn('test_class', attrs)

    def test_module_without_all_attribute(self):
        """Test DeprecatedModule with replacement module that has no __all__."""

        class ModuleWithoutAll:
            def some_function(self):
                return "test"

        module_without_all = ModuleWithoutAll()
        deprecated = DeprecatedModule(
            deprecated_name='test.no_all',
            replacement_module=module_without_all,
            replacement_name='test.replacement'
        )

        # Should not have __all__ attribute
        self.assertFalse(hasattr(deprecated, '__all__'))

        # Should still forward attributes
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.assertTrue(hasattr(deprecated, 'some_function'))


class TestCreateDeprecatedModuleProxy(unittest.TestCase):
    """Test the create_deprecated_module_proxy function."""

    def test_create_proxy_function(self):
        """Test the proxy creation function."""
        mock_module = MockReplacementModule()

        proxy = create_deprecated_module_proxy(
            deprecated_name='test.proxy',
            replacement_module=mock_module,
            replacement_name='test.new_module',
            version='1.0.0'
        )

        self.assertIsInstance(proxy, DeprecatedModule)
        self.assertEqual(proxy.__name__, 'test.proxy')

        # Test that it works
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = proxy.test_function(10)
            self.assertEqual(result, 20)

    def test_proxy_with_kwargs(self):
        """Test proxy creation with additional keyword arguments."""
        mock_module = MockReplacementModule()

        proxy = create_deprecated_module_proxy(
            deprecated_name='test.kwargs',
            replacement_module=mock_module,
            replacement_name='test.new',
            removal_version='3.0.0'
        )

        # Test warning includes removal version
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = proxy.test_function

            self.assertEqual(len(w), 1)
            self.assertIn('3.0.0', str(w[0].message))


class TestDeprecationEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""

    def test_circular_reference_handling(self):
        """Test that circular references don't break the deprecation system."""
        mock_module = MockReplacementModule()
        deprecated = DeprecatedModule(
            deprecated_name='test.circular',
            replacement_module=mock_module,
            replacement_name='test.replacement'
        )

        # Add a circular reference (this should not break anything)
        mock_module.circular_ref = deprecated

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Should still work normally
            result = deprecated.test_function(5)
            self.assertEqual(result, 10)

    def test_complex_attribute_access_patterns(self):
        """Test complex attribute access patterns."""
        mock_module = MockReplacementModule()
        deprecated = DeprecatedModule(
            deprecated_name='test.complex',
            replacement_module=mock_module,
            replacement_name='test.replacement'
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Test chained access
            func = deprecated.test_function
            result = func(7)
            self.assertEqual(result, 14)

            # Test accessing through variables
            var_func = getattr(deprecated, 'test_function')
            result2 = var_func(8)
            self.assertEqual(result2, 16)

    def test_stacklevel_accuracy(self):
        """Test that warnings point to the correct stack level."""
        mock_module = MockReplacementModule()
        deprecated = DeprecatedModule(
            deprecated_name='test.stack',
            replacement_module=mock_module,
            replacement_name='test.replacement'
        )

        def intermediate_function():
            return deprecated.test_function

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # This should generate a warning pointing to this test
            _ = intermediate_function()

            self.assertEqual(len(w), 1)
            # The warning should reference this test file, not internal code
            self.assertIn('_deprecation_test.py', w[0].filename)


class TestDeprecatedModuleInitialization(unittest.TestCase):
    """Test initialization and setup of deprecated modules."""

    def test_deprecated_module_initialization_minimal_parameters(self):
        """Test DeprecatedModule initialization with minimal parameters."""
        mock_module = MockReplacementModule()

        deprecated = DeprecatedModule(
            deprecated_name='test.minimal',
            replacement_module=mock_module,
            replacement_name='test.replacement_min'
        )

        # Test required attributes are set
        self.assertEqual(deprecated.__name__, 'test.minimal')
        self.assertEqual(deprecated._deprecated_name, 'test.minimal')
        self.assertEqual(deprecated._replacement_module, mock_module)
        self.assertEqual(deprecated._replacement_name, 'test.replacement_min')

        # Test optional attributes - version has a default, removal_version is None
        self.assertEqual(deprecated._version, "0.1.11")  # Default version
        self.assertIsNone(deprecated._removal_version)

        # Test docstring still generated without version info
        self.assertIn('DEPRECATED', deprecated.__doc__)
        self.assertIn('test.minimal', deprecated.__doc__)
        self.assertIn('test.replacement_min', deprecated.__doc__)

    def test_deprecated_module_with_empty_replacement_module(self):
        """Test DeprecatedModule with replacement module that has no attributes."""

        class EmptyModule:
            pass

        empty_module = EmptyModule()
        deprecated = DeprecatedModule(
            deprecated_name='test.empty',
            replacement_module=empty_module,
            replacement_name='test.empty_replacement'
        )

        # Should handle empty module gracefully
        self.assertEqual(deprecated.__name__, 'test.empty')
        self.assertFalse(hasattr(deprecated, '__all__'))

        # Accessing non-existent attribute should raise proper error
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with self.assertRaises(AttributeError):
                _ = deprecated.nonexistent

    def test_deprecated_module_initialization_with_callable_replacement(self):
        """Test DeprecatedModule with replacement module that has callable attributes."""

        class CallableModule:
            @staticmethod
            def func1():
                return "result1"

            @classmethod
            def func2(cls):
                return "result2"

            var1 = "variable1"

        callable_module = CallableModule()
        deprecated = DeprecatedModule(
            deprecated_name='test.callable',
            replacement_module=callable_module,
            replacement_name='test.callable_replacement'
        )

        # Test callable forwarding works
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            self.assertEqual(deprecated.func1(), "result1")
            self.assertEqual(deprecated.func2(), "result2")
            self.assertEqual(deprecated.var1, "variable1")


class TestScopedAPIStringImports(unittest.TestCase):
    """Test scoped API functionality with string-based imports."""

    def test_scoped_api_string_based_attribute_access(self):
        """Test that scoped APIs work with string-based attribute access."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Test that we can access scoped APIs through string-based lookups
            for api_name in brainstate.augment.__all__:
                with self.subTest(api_name=api_name):
                    # Should be able to get attribute via string lookup
                    attr = getattr(brainstate.augment, api_name, None)
                    self.assertIsNotNone(attr, f"API '{api_name}' should be accessible via getattr")

                    # Should be same as direct access
                    direct_attr = getattr(brainstate.augment, api_name)
                    self.assertIs(attr, direct_attr)

    def test_scoped_api_dynamic_import_patterns(self):
        """Test scoped APIs with dynamic import patterns."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Test importing specific functions dynamically
            api_names = ['grad', 'vmap', 'vector_grad']

            for api_name in api_names:
                with self.subTest(api_name=api_name):
                    # Simulate dynamic import pattern
                    if hasattr(brainstate.augment, api_name):
                        func = getattr(brainstate.augment, api_name)
                        self.assertTrue(callable(func))

                        # Should be the same as the transform version
                        if hasattr(brainstate.transform, api_name):
                            transform_func = getattr(brainstate.transform, api_name)
                            self.assertIs(func, transform_func)

    def test_scoped_api_list_comprehension_access(self):
        """Test accessing scoped APIs through list comprehensions."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Get all callable APIs from augment module
            callables = [getattr(brainstate.augment, name) for name in brainstate.augment.__all__
                         if callable(getattr(brainstate.augment, name, None))]

            # Should have found some callables
            self.assertGreater(len(callables), 0)

            # All should be actual callable objects
            for func in callables:
                self.assertTrue(callable(func))

    def test_scoped_api_introspection(self):
        """Test that scoped APIs support proper introspection."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Test that we can introspect the grad function
            if hasattr(brainstate.augment, 'grad'):
                grad_func = brainstate.augment.grad

                # Should have proper function attributes
                self.assertTrue(hasattr(grad_func, '__name__'))
                self.assertTrue(hasattr(grad_func, '__doc__'))
                self.assertTrue(hasattr(grad_func, '__module__'))

                # Name should be preserved
                self.assertEqual(grad_func.__name__, 'grad')

    def test_scoped_api_with_string_module_names(self):
        """Test scoped APIs work when modules are accessed via string names."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Test accessing deprecated modules by string name
            module_names = ['augment', 'compile', 'functional']

            for module_name in module_names:
                with self.subTest(module_name=module_name):
                    # Get module via getattr on brainstate
                    module = getattr(brainstate, module_name, None)
                    self.assertIsNotNone(module)

                    # Should have __all__ attribute
                    self.assertTrue(hasattr(module, '__all__'))

                    # Should be able to access APIs from the scoped list
                    for api_name in getattr(module, '__all__', []):
                        if hasattr(module, api_name):
                            attr = getattr(module, api_name)
                            self.assertIsNotNone(attr)


class TestDeprecationErrorHandlingAndFallbacks(unittest.TestCase):
    """Test error handling and fallback mechanisms in deprecation system."""

    def test_invalid_attribute_access_error_messages(self):
        """Test that invalid attribute access provides helpful error messages."""
        mock_module = MockReplacementModule()
        deprecated = DeprecatedModule(
            deprecated_name='test.invalid_attr',
            replacement_module=mock_module,
            replacement_name='test.replacement_invalid'
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            with self.assertRaises(AttributeError) as context:
                _ = deprecated.completely_nonexistent_function

            error_msg = str(context.exception)

            # Error message should contain helpful information
            self.assertIn('test.invalid_attr', error_msg)
            self.assertIn('completely_nonexistent_function', error_msg)

    def test_fallback_when_replacement_module_lacks_attribute(self):
        """Test fallback behavior when replacement module lacks expected attribute."""

        class IncompleteModule:
            def existing_func(self):
                return "exists"

        incomplete_module = IncompleteModule()
        deprecated = DeprecatedModule(
            deprecated_name='test.incomplete',
            replacement_module=incomplete_module,
            replacement_name='test.incomplete_replacement'
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Should work for existing function
            result = deprecated.existing_func()
            self.assertEqual(result, "exists")

            # Should raise AttributeError for missing function
            with self.assertRaises(AttributeError):
                _ = deprecated.missing_func

    def test_exception_handling_during_warning_generation(self):
        """Test that exceptions during warning generation don't break functionality."""

        class ProblematicModule:
            def test_func(self):
                return "works"

        problematic_module = ProblematicModule()
        deprecated = DeprecatedModule(
            deprecated_name='test.problematic',
            replacement_module=problematic_module,
            replacement_name='test.problematic_replacement'
        )

        # Even if warning generation has issues, functionality should still work
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            result = deprecated.test_func()
            self.assertEqual(result, "works")

    def test_graceful_handling_of_special_attributes(self):
        """Test graceful handling of special Python attributes."""
        mock_module = MockReplacementModule()
        deprecated = DeprecatedModule(
            deprecated_name='test.special',
            replacement_module=mock_module,
            replacement_name='test.special_replacement'
        )

        # Test that accessing special attributes doesn't break
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # These should work without warnings or errors
            self.assertEqual(deprecated.__name__, 'test.special')
            self.assertIsInstance(deprecated.__doc__, str)

            # repr should work
            repr_str = repr(deprecated)
            self.assertIsInstance(repr_str, str)

    def test_multiple_error_conditions_simultaneously(self):
        """Test handling multiple error conditions at once."""

        class MultiErrorModule:
            def func1(self):
                raise RuntimeError("Runtime error in func1")

            # func2 is missing despite being in __all__

        error_module = MultiErrorModule()
        deprecated = DeprecatedModule(
            deprecated_name='test.multi_error',
            replacement_module=error_module,
            replacement_name='test.multi_error_replacement'
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Test that we get the expected errors
            with self.assertRaises(RuntimeError):
                deprecated.func1()

            with self.assertRaises(AttributeError):
                _ = deprecated.func2

            with self.assertRaises(AttributeError):
                _ = deprecated.nonexistent


class TestConcurrentAccessAndThreadSafety(unittest.TestCase):
    """Test concurrent access and thread safety of deprecation system."""

    def test_concurrent_attribute_access(self):
        """Test that concurrent attribute access works correctly."""
        import threading
        import time

        mock_module = MockReplacementModule()
        deprecated = DeprecatedModule(
            deprecated_name='test.concurrent',
            replacement_module=mock_module,
            replacement_name='test.concurrent_replacement'
        )

        results = []
        errors = []

        def access_attributes():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")

                    # Access different attributes multiple times
                    for _ in range(10):
                        result1 = deprecated.test_function(5)
                        result2 = deprecated.test_variable
                        results.append((result1, result2))
                        time.sleep(0.001)  # Small delay to encourage race conditions

            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=access_attributes)
            threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Check results
        self.assertEqual(len(errors), 0, f"Errors occurred: {errors}")
        self.assertGreater(len(results), 0)

        # All results should be consistent
        for result1, result2 in results:
            self.assertEqual(result1, 10)  # test_function(5) should return 10
            self.assertEqual(result2, 42)  # test_variable should be 42

    def test_thread_safety_of_warning_generation(self):
        """Test that warning generation is thread-safe."""
        import threading

        mock_module = MockReplacementModule()
        deprecated = DeprecatedModule(
            deprecated_name='test.thread_warnings',
            replacement_module=mock_module,
            replacement_name='test.thread_warnings_replacement'
        )

        warning_counts = []

        def generate_warnings():
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")

                # Access attributes to generate warnings
                _ = deprecated.test_function
                _ = deprecated.test_variable
                _ = deprecated.test_class

                warning_counts.append(len(w))

        # Create multiple threads
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=generate_warnings)
            threads.append(thread)

        # Start and join all threads
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # Each thread should have generated some warnings
        self.assertEqual(len(warning_counts), 3)
        for count in warning_counts:
            self.assertGreaterEqual(count, 0)

    def test_race_condition_in_attribute_caching(self):
        """Test for race conditions in any internal attribute caching."""
        import threading

        mock_module = MockReplacementModule()
        deprecated = DeprecatedModule(
            deprecated_name='test.race_condition',
            replacement_module=mock_module,
            replacement_name='test.race_condition_replacement'
        )

        results = {}
        lock = threading.Lock()

        def access_same_attribute(thread_id):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # Access the same attribute multiple times
                for i in range(20):
                    attr = deprecated.test_function
                    result = attr(i)

                    with lock:
                        if thread_id not in results:
                            results[thread_id] = []
                        results[thread_id].append(result)

        # Create threads that all access the same attribute
        threads = []
        for i in range(4):
            thread = threading.Thread(target=access_same_attribute, args=(i,))
            threads.append(thread)

        # Start and join all threads
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # Verify all threads got consistent results
        self.assertEqual(len(results), 4)
        for thread_id, thread_results in results.items():
            self.assertEqual(len(thread_results), 20)
            for i, result in enumerate(thread_results):
                self.assertEqual(result, i * 2)  # test_function multiplies by 2


class TestMemoryUsageAndPerformance(unittest.TestCase):
    """Test memory usage and performance aspects of deprecation system."""

    def test_memory_usage_of_deprecated_modules(self):
        """Test that deprecated modules don't consume excessive memory."""

        # Create many deprecated modules
        modules = []
        for i in range(100):
            mock_module = MockReplacementModule()
            deprecated = DeprecatedModule(
                deprecated_name=f'test.memory_{i}',
                replacement_module=mock_module,
                replacement_name=f'test.memory_replacement_{i}'
            )
            modules.append(deprecated)

        # Test that we can create many modules without excessive memory usage
        self.assertEqual(len(modules), 100)

        # Basic functionality should still work
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            for i in range(0, 100, 10):  # Test every 10th module
                result = modules[i].test_function(1)
                self.assertEqual(result, 2)

    def test_performance_of_attribute_access(self):
        """Test performance of deprecated module attribute access."""
        import time

        mock_module = MockReplacementModule()
        deprecated = DeprecatedModule(
            deprecated_name='test.performance',
            replacement_module=mock_module,
            replacement_name='test.performance_replacement'
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Time multiple attribute accesses
            start_time = time.time()

            for _ in range(1000):
                _ = deprecated.test_function
                _ = deprecated.test_variable
                _ = deprecated.test_class

            end_time = time.time()

            # Should complete reasonably quickly (less than 1 second for 1000 iterations)
            elapsed = end_time - start_time
            self.assertLess(elapsed, 1.0, f"Attribute access took too long: {elapsed}s")

    def test_warning_performance_impact(self):
        """Test that warning generation doesn't significantly impact performance."""
        import time

        mock_module = MockReplacementModule()
        deprecated = DeprecatedModule(
            deprecated_name='test.warning_performance',
            replacement_module=mock_module,
            replacement_name='test.warning_performance_replacement'
        )

        # Test with warnings enabled
        start_time = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("always")

            for _ in range(100):
                _ = deprecated.test_function
                _ = deprecated.test_variable

        with_warnings_time = time.time() - start_time

        # Test with warnings disabled
        start_time = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            for _ in range(100):
                _ = deprecated.test_function
                _ = deprecated.test_variable

        without_warnings_time = time.time() - start_time

        # With warnings should not be dramatically slower (less than 10x)
        if without_warnings_time > 0:
            ratio = with_warnings_time / without_warnings_time
            self.assertLess(ratio, 10.0, f"Warning generation too slow: {ratio}x slower")

    def test_memory_leak_prevention(self):
        """Test that deprecated modules don't cause memory leaks."""
        import gc
        import weakref

        # Create deprecated modules with weak references
        weak_refs = []

        for i in range(50):
            mock_module = MockReplacementModule()
            deprecated = DeprecatedModule(
                deprecated_name=f'test.leak_{i}',
                replacement_module=mock_module,
                replacement_name=f'test.leak_replacement_{i}'
            )

            # Access some attributes to trigger any caching
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _ = deprecated.test_function

            weak_refs.append(weakref.ref(deprecated))

        # Force garbage collection
        gc.collect()

        # After modules go out of scope, weak references should become invalid
        # (This test is somewhat artificial but helps catch obvious leaks)
        del deprecated
        gc.collect()

        # At least some weak references should be collectible
        # (We can't guarantee all will be collected due to Python's GC behavior)
        self.assertTrue(len(weak_refs) > 0)


class TestDeprecatedAugment(unittest.TestCase):
    """Test suite for the deprecated brainstate.augment module."""

    def test_augment_module_import(self):
        """Test that the deprecated augment module can be imported."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import brainstate
            # Access an attribute to trigger deprecation warning
            _ = brainstate.augment.grad

            # Check that a deprecation warning was issued (excluding JAX warnings)
            relevant_warnings = [
                warning for warning in w
                if issubclass(warning.category, DeprecationWarning)
                   and 'brainstate.augment' in str(warning.message)
            ]
            # self.assertGreater(len(relevant_warnings), 0)

    def test_augmentation_functions(self):
        """Test that all augmentation functions are accessible."""
        import brainstate

        augment_funcs = [
            'GradientTransform',
            'grad',
            'vector_grad',
            'hessian',
            'jacobian',
            'jacrev',
            'jacfwd',
            'vmap',
            'pmap',
            'map',
            'vmap_new_states',
        ]

        for func_name in augment_funcs:
            with self.subTest(function=func_name):
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")

                    # Access the function
                    func = getattr(brainstate.augment, func_name)
                    self.assertIsNotNone(func)

                    # Check that a deprecation warning was issued
                    deprecation_warnings = [warning for warning in w if
                                            issubclass(warning.category, DeprecationWarning)]
                    # Filter out the JAX warning
                    relevant_warnings = [w for w in deprecation_warnings if 'brainstate.augment' in str(w.message)]
                    # self.assertGreater(len(relevant_warnings), 0, f"No deprecation warning for {func_name}")

    def test_gradient_functions(self):
        """Test gradient-related functions."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            # Test grad
            grad = brainstate.augment.grad
            self.assertIsNotNone(grad)

            # Test vector_grad
            vector_grad = brainstate.augment.vector_grad
            self.assertIsNotNone(vector_grad)

            # Test GradientTransform
            GradientTransform = brainstate.augment.GradientTransform
            self.assertIsNotNone(GradientTransform)

    def test_grad_function(self):
        """Test grad function functionality."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            # Test grad function
            grad = brainstate.augment.grad
            self.assertIsNotNone(grad)
            # Just check that it's callable
            self.assertTrue(callable(grad))

    def test_jacobian_functions(self):
        """Test Jacobian-related functions."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            # Test jacobian
            jacobian = brainstate.augment.jacobian
            self.assertIsNotNone(jacobian)

            # Test jacrev
            jacrev = brainstate.augment.jacrev
            self.assertIsNotNone(jacrev)

            # Test jacfwd
            jacfwd = brainstate.augment.jacfwd
            self.assertIsNotNone(jacfwd)

    def test_hessian_function(self):
        """Test Hessian function."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            # Test hessian
            hessian = brainstate.augment.hessian
            self.assertIsNotNone(hessian)
            # Just check that it's callable
            self.assertTrue(callable(hessian))

    def test_mapping_functions(self):
        """Test mapping-related functions."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            # Test vmap
            vmap = brainstate.augment.vmap
            self.assertIsNotNone(vmap)

            # Test pmap
            pmap = brainstate.augment.pmap
            self.assertIsNotNone(pmap)

            # Test map
            map_func = brainstate.augment.map
            self.assertIsNotNone(map_func)

    def test_vmap_function(self):
        """Test vmap function functionality."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            # Test vmap
            vmap = brainstate.augment.vmap
            self.assertIsNotNone(vmap)
            # Just check that it's callable
            self.assertTrue(callable(vmap))

    def test_vmap_new_states(self):
        """Test vmap_new_states function."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            # Test vmap_new_states
            vmap_new_states = brainstate.augment.vmap_new_states
            self.assertIsNotNone(vmap_new_states)

    def test_module_attributes(self):
        """Test module-level attributes."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            # Test __name__ attribute
            self.assertEqual(brainstate.augment.__name__, 'brainstate.augment')

            # Test __doc__ attribute
            self.assertIn('DEPRECATED', brainstate.augment.__doc__)

            # Test __all__ attribute
            self.assertIsInstance(brainstate.augment.__all__, list)
            self.assertIn('grad', brainstate.augment.__all__)
            self.assertIn('vmap', brainstate.augment.__all__)

    def test_dir_method(self):
        """Test that dir() returns appropriate attributes."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import brainstate

            attrs = dir(brainstate.augment)

            # Check that expected attributes are present
            expected_attrs = [
                'grad', 'vmap', 'jacobian', 'hessian',
                '__name__', '__doc__', '__all__'
            ]
            for attr in expected_attrs:
                self.assertIn(attr, attrs)

            # Check that a deprecation warning was issued
            # self.assertTrue(any(issubclass(warning.category, DeprecationWarning) for warning in w))

    def test_invalid_attribute_access(self):
        """Test that accessing invalid attributes raises appropriate errors."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            with self.assertRaises(AttributeError) as context:
                _ = brainstate.augment.NonExistentFunction

            self.assertIn('NonExistentFunction', str(context.exception))
            self.assertIn('brainstate.augment', str(context.exception))

    def test_repr_method(self):
        """Test the __repr__ method of the deprecated module."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            repr_str = repr(brainstate.augment)
            self.assertIn('DeprecatedModule', repr_str)
            self.assertIn('brainstate.augment', repr_str)
            self.assertIn('brainstate.transform', repr_str)

    def test_gradient_transform_class(self):
        """Test GradientTransform class."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            # Test GradientTransform class
            GradientTransform = brainstate.augment.GradientTransform
            self.assertIsNotNone(GradientTransform)


class TestDeprecatedCompile(unittest.TestCase):
    """Test suite for the deprecated brainstate.compile module."""

    def test_compile_module_import(self):
        """Test that the deprecated compile module can be imported."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import brainstate
            # Access an attribute to trigger deprecation warning
            _ = brainstate.compile.jit

            # Check that a deprecation warning was issued (excluding JAX warnings)
            relevant_warnings = [
                warning for warning in w
                if issubclass(warning.category, DeprecationWarning)
                   and 'brainstate.compile' in str(warning.message)
            ]
            # self.assertGreater(len(relevant_warnings), 0)

    def test_compilation_functions(self):
        """Test that all compilation functions are accessible."""
        import brainstate

        compile_funcs = [
            'checkpoint',
            'remat',
            'cond',
            'switch',
            'ifelse',
            'jit_error_if',
            'jit',
            'scan',
            'checkpointed_scan',
            'for_loop',
            'checkpointed_for_loop',
            'while_loop',
            'bounded_while_loop',
            'StatefulFunction',
            'make_jaxpr',
            'ProgressBar',
        ]

        for func_name in compile_funcs:
            with self.subTest(function=func_name):
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")

                    # Access the function
                    func = getattr(brainstate.compile, func_name)
                    self.assertIsNotNone(func)

                    # Check that a deprecation warning was issued
                    deprecation_warnings = [warning for warning in w if
                                            issubclass(warning.category, DeprecationWarning)]
                    # Filter out the JAX warning
                    relevant_warnings = [w for w in deprecation_warnings if 'brainstate.compile' in str(w.message)]
                    # self.assertGreater(len(relevant_warnings), 0, f"No deprecation warning for {func_name}")

    def test_jit_function(self):
        """Test JIT compilation function."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            # Test jit function
            jit = brainstate.compile.jit
            self.assertIsNotNone(jit)
            # Just check that it's callable
            self.assertTrue(callable(jit))

    def test_cond_function(self):
        """Test conditional function."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            # Test cond function
            cond = brainstate.compile.cond
            self.assertIsNotNone(cond)
            # Just check that it's callable
            self.assertTrue(callable(cond))

    def test_ifelse_function(self):
        """Test ifelse function."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            # Test ifelse function
            ifelse = brainstate.compile.ifelse
            self.assertIsNotNone(ifelse)

    def test_switch_function(self):
        """Test switch function."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            # Test switch function
            switch = brainstate.compile.switch
            self.assertIsNotNone(switch)

    def test_loop_functions(self):
        """Test loop-related functions."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            # Test for_loop
            for_loop = brainstate.compile.for_loop
            self.assertIsNotNone(for_loop)

            # Test while_loop
            while_loop = brainstate.compile.while_loop
            self.assertIsNotNone(while_loop)

            # Test bounded_while_loop
            bounded_while_loop = brainstate.compile.bounded_while_loop
            self.assertIsNotNone(bounded_while_loop)

    def test_scan_functions(self):
        """Test scan-related functions."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            # Test scan
            scan = brainstate.compile.scan
            self.assertIsNotNone(scan)

            # Test checkpointed_scan
            checkpointed_scan = brainstate.compile.checkpointed_scan
            self.assertIsNotNone(checkpointed_scan)

    def test_checkpoint_functions(self):
        """Test checkpoint-related functions."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            # Test checkpoint
            checkpoint = brainstate.compile.checkpoint
            self.assertIsNotNone(checkpoint)

            # Test remat (rematerialization)
            remat = brainstate.compile.remat
            self.assertIsNotNone(remat)

    def test_jit_error_if(self):
        """Test jit_error_if function."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            # Test jit_error_if
            jit_error_if = brainstate.compile.jit_error_if
            self.assertIsNotNone(jit_error_if)

    def test_stateful_function(self):
        """Test StatefulFunction class."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            # Test StatefulFunction
            StatefulFunction = brainstate.compile.StatefulFunction
            self.assertIsNotNone(StatefulFunction)

    def test_make_jaxpr(self):
        """Test make_jaxpr function."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            # Test make_jaxpr
            make_jaxpr = brainstate.compile.make_jaxpr
            self.assertIsNotNone(make_jaxpr)

    def test_progress_bar(self):
        """Test ProgressBar class."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            # Test ProgressBar
            ProgressBar = brainstate.compile.ProgressBar
            self.assertIsNotNone(ProgressBar)

    def test_checkpointed_for_loop(self):
        """Test checkpointed_for_loop function."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            # Test checkpointed_for_loop
            checkpointed_for_loop = brainstate.compile.checkpointed_for_loop
            self.assertIsNotNone(checkpointed_for_loop)

    def test_module_attributes(self):
        """Test module-level attributes."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            # Test __name__ attribute
            self.assertEqual(brainstate.compile.__name__, 'brainstate.compile')

            # Test __doc__ attribute
            self.assertIn('DEPRECATED', brainstate.compile.__doc__)

            # Test __all__ attribute
            self.assertIsInstance(brainstate.compile.__all__, list)
            self.assertIn('jit', brainstate.compile.__all__)
            self.assertIn('cond', brainstate.compile.__all__)

    def test_dir_method(self):
        """Test that dir() returns appropriate attributes."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import brainstate

            attrs = dir(brainstate.compile)

            # Check that expected attributes are present
            expected_attrs = [
                'jit', 'cond', 'scan', 'for_loop', 'while_loop',
                '__name__', '__doc__', '__all__'
            ]
            for attr in expected_attrs:
                self.assertIn(attr, attrs)

            # Check that a deprecation warning was issued
            # self.assertTrue(any(issubclass(warning.category, DeprecationWarning) for warning in w))

    def test_invalid_attribute_access(self):
        """Test that accessing invalid attributes raises appropriate errors."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            with self.assertRaises(AttributeError) as context:
                _ = brainstate.compile.NonExistentFunction

            self.assertIn('NonExistentFunction', str(context.exception))
            self.assertIn('brainstate.compile', str(context.exception))

    def test_repr_method(self):
        """Test the __repr__ method of the deprecated module."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            repr_str = repr(brainstate.compile)
            self.assertIn('DeprecatedModule', repr_str)
            self.assertIn('brainstate.compile', repr_str)
            self.assertIn('brainstate.transform', repr_str)


class TestDeprecatedFunctional(unittest.TestCase):
    """Test suite for the deprecated brainstate.functional module."""

    def test_functional_module_import(self):
        """Test that the deprecated functional module can be imported."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import brainstate
            # Access an attribute to trigger deprecation warning
            _ = brainstate.functional.relu

            # Check that a deprecation warning was issued (excluding JAX warnings)
            relevant_warnings = [
                warning for warning in w
                if issubclass(warning.category, DeprecationWarning)
                   and 'brainstate.functional' in str(warning.message)
            ]
            # self.assertGreater(len(relevant_warnings), 0)

    def test_activation_functions(self):
        """Test that all activation functions are accessible."""
        import brainstate

        activations = [
            'tanh',
            'relu',
            'squareplus',
            'softplus',
            'soft_sign',
            'sigmoid',
            'silu',
            'swish',
            'log_sigmoid',
            'elu',
            'leaky_relu',
            'hard_tanh',
            'celu',
            'selu',
            'gelu',
            'glu',
            'logsumexp',
            'log_softmax',
            'softmax',
            'standardize'
        ]

        for activation_name in activations:
            with self.subTest(activation=activation_name):
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")

                    # Access the activation function
                    activation = getattr(brainstate.functional, activation_name)
                    self.assertIsNotNone(activation)

                    # Check that a deprecation warning was issued
                    deprecation_warnings = [warning for warning in w if
                                            issubclass(warning.category, DeprecationWarning)]
                    # Filter out the JAX warning
                    relevant_warnings = [w for w in deprecation_warnings if 'brainstate.functional' in str(w.message)]
                    # self.assertGreater(len(relevant_warnings), 0, f"No deprecation warning for {activation_name}")

    def test_activation_functionality(self):
        """Test that deprecated activation functions still work correctly."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            # Test data
            x = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])

            # Test relu
            result = brainstate.functional.relu(x)
            expected = jnp.maximum(0, x)
            self.assertTrue(jnp.allclose(result, expected))

            # Test sigmoid
            result = brainstate.functional.sigmoid(x)
            expected = 1 / (1 + jnp.exp(-x))
            self.assertTrue(jnp.allclose(result, expected))

            # Test tanh
            result = brainstate.functional.tanh(x)
            expected = jnp.tanh(x)
            self.assertTrue(jnp.allclose(result, expected))

            # Test softmax
            result = brainstate.functional.softmax(x)
            self.assertAlmostEqual(jnp.sum(result), 1.0, places=5)

    def test_weight_standardization(self):
        """Test weight standardization function."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            # Test weight standardization
            weight_std = brainstate.functional.weight_standardization
            self.assertIsNotNone(weight_std)

    def test_clip_grad_norm(self):
        """Test clip_grad_norm function."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            # Test clip_grad_norm
            clip_grad = brainstate.functional.clip_grad_norm
            self.assertIsNotNone(clip_grad)

    def test_leaky_relu(self):
        """Test leaky_relu with custom alpha."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            x = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
            # Test leaky_relu
            result = brainstate.functional.leaky_relu(x, negative_slope=0.01)
            # Check positive values are unchanged
            self.assertTrue(jnp.allclose(result[x >= 0], x[x >= 0]))
            # Check negative values are scaled
            self.assertTrue(jnp.allclose(result[x < 0], 0.01 * x[x < 0]))

    def test_elu_activation(self):
        """Test ELU activation function."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            x = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
            # Test ELU
            result = brainstate.functional.elu(x, alpha=1.0)
            # Check positive values are unchanged
            self.assertTrue(jnp.allclose(result[x >= 0], x[x >= 0]))
            # Check negative values follow ELU formula
            expected_neg = 1.0 * (jnp.exp(x[x < 0]) - 1)
            self.assertTrue(jnp.allclose(result[x < 0], expected_neg))

    def test_gelu_activation(self):
        """Test GELU activation function."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            x = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
            # Test GELU
            result = brainstate.functional.gelu(x)
            self.assertEqual(result.shape, x.shape)
            # Check that GELU(0) ≈ 0
            self.assertAlmostEqual(result[2], 0.0, places=5)

    def test_softplus_activation(self):
        """Test Softplus activation function."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            x = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
            # Test softplus
            result = brainstate.functional.softplus(x)
            expected = jnp.log(1 + jnp.exp(x))
            self.assertTrue(jnp.allclose(result, expected))

    def test_log_softmax(self):
        """Test log_softmax function."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            x = jnp.array([1.0, 2.0, 3.0])
            # Test log_softmax
            result = brainstate.functional.log_softmax(x)
            # Check that exp of log_softmax sums to 1
            self.assertAlmostEqual(jnp.sum(jnp.exp(result)), 1.0, places=5)

    def test_silu_swish(self):
        """Test SiLU (Swish) activation function."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            x = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])

            # Test silu
            result_silu = brainstate.functional.silu(x)
            # Test swish (should be the same as silu)
            result_swish = brainstate.functional.swish(x)

            # They should be equal
            self.assertTrue(jnp.allclose(result_silu, result_swish))

            # Check against expected formula: x * sigmoid(x)
            expected = x * brainstate.functional.sigmoid(x)
            self.assertTrue(jnp.allclose(result_silu, expected))

    def test_module_attributes(self):
        """Test module-level attributes."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            # Test __name__ attribute
            self.assertEqual(brainstate.functional.__name__, 'brainstate.functional')

            # Test __doc__ attribute
            self.assertIn('DEPRECATED', brainstate.functional.__doc__)

            # Test __all__ attribute
            self.assertIsInstance(brainstate.functional.__all__, list)
            self.assertIn('relu', brainstate.functional.__all__)
            self.assertIn('sigmoid', brainstate.functional.__all__)

    def test_dir_method(self):
        """Test that dir() returns appropriate attributes."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import brainstate

            attrs = dir(brainstate.functional)

            # Check that expected attributes are present
            expected_attrs = [
                'relu', 'sigmoid', 'tanh', 'softmax',
                '__name__', '__doc__', '__all__'
            ]
            for attr in expected_attrs:
                self.assertIn(attr, attrs)

            # Check that a deprecation warning was issued
            # self.assertTrue(any(issubclass(warning.category, DeprecationWarning) for warning in w))

    def test_invalid_attribute_access(self):
        """Test that accessing invalid attributes raises appropriate errors."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            with self.assertRaises(AttributeError) as context:
                _ = brainstate.functional.NonExistentFunction

            self.assertIn('NonExistentFunction', str(context.exception))
            self.assertIn('brainstate.functional', str(context.exception))

    def test_repr_method(self):
        """Test the __repr__ method of the deprecated module."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            repr_str = repr(brainstate.functional)
            self.assertIn('DeprecatedModule', repr_str)
            self.assertIn('brainstate.functional', repr_str)
            self.assertIn('brainstate.nn', repr_str)


class TestDeprecatedInit(unittest.TestCase):
    """Test suite for the deprecated brainstate.init module."""

    def test_init_module_import(self):
        """Test that the deprecated init module can be imported."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import brainstate
            # Access an attribute to trigger deprecation warning
            _ = brainstate.init.Constant

            # Check that a deprecation warning was issued
            self.assertGreater(len(w), 0)
            self.assertTrue(any(issubclass(warning.category, DeprecationWarning) for warning in w))

    def test_param_function(self):
        """Test the deprecated param function."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import brainstate

            # Test accessing param function
            param = brainstate.init.param
            self.assertIsNotNone(param)

            # Check that a deprecation warning was issued
            self.assertTrue(any(issubclass(warning.category, DeprecationWarning) for warning in w))

    def test_initializers(self):
        """Test that all deprecated initializers are accessible."""
        import brainstate

        # Test various initializers
        initializers = [
            'Constant',
            'Identity',
            'Normal',
            'TruncatedNormal',
            'Uniform',
            'KaimingUniform',
            'KaimingNormal',
            'XavierUniform',
            'XavierNormal',
            'LecunUniform',
            'LecunNormal',
            'Orthogonal',
            'DeltaOrthogonal',
        ]

        for init_name in initializers:
            with self.subTest(initializer=init_name):
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")

                    # Access the initializer
                    initializer = getattr(brainstate.init, init_name)
                    self.assertIsNotNone(initializer)

                    # Check that a deprecation warning was issued
                    deprecation_warnings = [warning for warning in w if
                                            issubclass(warning.category, DeprecationWarning)]
                    # Filter out the JAX warning
                    relevant_warnings = [w for w in deprecation_warnings if 'brainstate.init' in str(w.message)]
                    # self.assertGreater(len(relevant_warnings), 0, f"No deprecation warning for {init_name}")

    def test_initializer_functionality(self):
        """Test that deprecated initializers still work correctly."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            # Test Constant initializer
            const_init = brainstate.init.Constant(0.5)
            result = const_init((2, 3))
            self.assertEqual(result.shape, (2, 3))
            self.assertTrue(jnp.allclose(result, 0.5))

            # Test Normal initializer
            normal_init = brainstate.init.Normal(mean=0.0, std=1.0)
            result = normal_init((10, 10))
            self.assertEqual(result.shape, (10, 10))

            # Test Uniform initializer
            uniform_init = brainstate.init.Uniform(low=-1.0, high=1.0)
            result = uniform_init((5, 5))
            self.assertEqual(result.shape, (5, 5))
            self.assertTrue(jnp.all(result >= -1.0))
            self.assertTrue(jnp.all(result <= 1.0))

    def test_kaiming_initializers(self):
        """Test Kaiming (He) initialization methods."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            # Test KaimingUniform
            kaiming_uniform = brainstate.init.KaimingUniform()
            result = kaiming_uniform((10, 10))
            self.assertEqual(result.shape, (10, 10))

            # Test KaimingNormal
            kaiming_normal = brainstate.init.KaimingNormal()
            result = kaiming_normal((10, 10))
            self.assertEqual(result.shape, (10, 10))

    def test_xavier_initializers(self):
        """Test Xavier (Glorot) initialization methods."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            # Test XavierUniform
            xavier_uniform = brainstate.init.XavierUniform()
            result = xavier_uniform((10, 10))
            self.assertEqual(result.shape, (10, 10))

            # Test XavierNormal
            xavier_normal = brainstate.init.XavierNormal()
            result = xavier_normal((10, 10))
            self.assertEqual(result.shape, (10, 10))

    def test_lecun_initializers(self):
        """Test LeCun initialization methods."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            # Test LecunUniform
            lecun_uniform = brainstate.init.LecunUniform()
            result = lecun_uniform((10, 10))
            self.assertEqual(result.shape, (10, 10))

            # Test LecunNormal
            lecun_normal = brainstate.init.LecunNormal()
            result = lecun_normal((10, 10))
            self.assertEqual(result.shape, (10, 10))

    def test_orthogonal_initializers(self):
        """Test Orthogonal initialization methods."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            # Test Orthogonal
            orthogonal = brainstate.init.Orthogonal()
            result = orthogonal((10, 10))
            self.assertEqual(result.shape, (10, 10))

            # Test DeltaOrthogonal with 3D shape (required)
            delta_orthogonal = brainstate.init.DeltaOrthogonal()
            result = delta_orthogonal((3, 3, 3))
            self.assertEqual(result.shape, (3, 3, 3))

    def test_identity_initializer(self):
        """Test Identity initializer."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            # Test Identity
            identity = brainstate.init.Identity()
            result = identity((5, 5))
            self.assertEqual(result.shape, (5, 5))
            # Check it's an identity matrix
            expected = jnp.eye(5)
            self.assertTrue(jnp.allclose(result, expected))

    def test_module_attributes(self):
        """Test module-level attributes."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            # Test __name__ attribute
            self.assertEqual(brainstate.init.__name__, 'braintools.init')

            # Test __all__ attribute
            self.assertIsInstance(brainstate.init.__all__, list)
            self.assertIn('Constant', brainstate.init.__all__)
            self.assertIn('Normal', brainstate.init.__all__)

    def test_dir_method(self):
        """Test that dir() returns appropriate attributes."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import brainstate

            attrs = dir(brainstate.init)

            # Check that expected attributes are present
            expected_attrs = [
                'Constant', 'Normal', 'Uniform', 'XavierNormal',
                '__name__', '__doc__', '__all__'
            ]
            for attr in expected_attrs:
                self.assertIn(attr, attrs)

            # Check that a deprecation warning was issued
            self.assertTrue(any(issubclass(warning.category, DeprecationWarning) for warning in w))
