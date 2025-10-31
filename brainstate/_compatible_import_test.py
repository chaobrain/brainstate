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

"""
Comprehensive test suite for the _compatible_import module.

This test module provides extensive coverage of the compatibility layer
functionality, including:
- JAX version-dependent imports and compatibility
- Utility functions (safe_map, safe_zip, unzip2)
- Function wrapping and metadata handling
- Type safety and error handling
- Edge cases and boundary conditions
"""

import unittest
from functools import partial
from unittest.mock import Mock

import jax
import jax.numpy as jnp
import numpy as np

from brainstate import _compatible_import as compat


class TestJAXVersionCompatibility(unittest.TestCase):
    """Test JAX version-dependent imports and compatibility."""

    def setUp(self):
        """Set up test environment."""
        self.original_version = jax.__version_info__

    def tearDown(self):
        """Clean up after tests."""
        # Restore original version info
        jax.__version_info__ = self.original_version

    def test_device_import_compatibility(self):
        """Test Device import works across JAX versions."""
        # Test that Device is available and importable
        self.assertTrue(hasattr(compat, 'Device'))
        self.assertIsNotNone(compat.Device)

        # Test Device can be used for type checking
        device = jax.devices()[0]
        self.assertIsInstance(device, compat.Device)

    def test_core_imports_availability(self):
        """Test core JAX imports are available."""
        # Core types should be available
        core_types = [
            'ClosedJaxpr', 'Primitive', 'Var', 'JaxprEqn',
            'Jaxpr', 'Literal', 'Tracer'
        ]

        for type_name in core_types:
            self.assertTrue(hasattr(compat, type_name),
                            f"{type_name} should be available")
            self.assertIsNotNone(getattr(compat, type_name))

    def test_function_imports_availability(self):
        """Test function imports are available."""
        functions = [
            'jaxpr_as_fun', 'get_aval', 'to_concrete_aval',
            'extend_axis_env_nd'
        ]

        for func_name in functions:
            self.assertTrue(hasattr(compat, func_name),
                            f"{func_name} should be available")
            self.assertTrue(callable(getattr(compat, func_name)),
                            f"{func_name} should be callable")

    def test_extend_axis_env_nd_functionality(self):
        """Test extend_axis_env_nd context manager."""
        # Test basic functionality
        with compat.extend_axis_env_nd([('test_axis', 10)]):
            # Context should execute without error
            pass

        # Test with multiple axes
        with compat.extend_axis_env_nd([('batch', 32), ('seq', 128)]):
            pass

        # Test with empty axes
        with compat.extend_axis_env_nd([]):
            pass

    def test_get_aval_functionality(self):
        """Test get_aval function works correctly."""
        # Test with JAX array
        arr = jnp.array([1, 2, 3])
        aval = compat.get_aval(arr)
        self.assertIsNotNone(aval)

        # Test with scalar
        scalar = jnp.float32(3.14)
        scalar_aval = compat.get_aval(scalar)
        self.assertIsNotNone(scalar_aval)

    def test_to_concrete_aval_functionality(self):
        """Test to_concrete_aval function."""
        # Test with concrete array
        arr = jnp.array([1, 2, 3])
        result = compat.to_concrete_aval(arr)
        self.assertIsNotNone(result)

        # # Test with scalar
        # scalar = 42.0
        # result = compat.to_concrete_aval(scalar)
        # self.assertEqual(result, scalar)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions like safe_map, safe_zip, unzip2."""

    def setUp(self):
        """Set up test environment."""
        pass

    def tearDown(self):
        """Clean up after tests."""
        pass

    def test_safe_map_basic(self):
        """Test basic safe_map functionality."""
        # Single argument function
        result = compat.safe_map(lambda x: x * 2, [1, 2, 3])
        self.assertEqual(result, [2, 4, 6])

        # Multiple argument function
        result = compat.safe_map(lambda x, y: x + y, [1, 2, 3], [4, 5, 6])
        self.assertEqual(result, [5, 7, 9])

        # String function
        result = compat.safe_map(str.upper, ['a', 'b', 'c'])
        self.assertEqual(result, ['A', 'B', 'C'])

    def test_safe_map_empty_inputs(self):
        """Test safe_map with empty inputs."""
        result = compat.safe_map(lambda x: x, [])
        self.assertEqual(result, [])

        result = compat.safe_map(lambda x, y: x + y, [], [])
        self.assertEqual(result, [])

    def test_safe_map_length_mismatch(self):
        """Test safe_map raises error on length mismatch."""
        with self.assertRaises(AssertionError) as context:
            compat.safe_map(lambda x, y: x + y, [1, 2, 3], [4, 5])

        self.assertIn('length mismatch', str(context.exception))

    def test_safe_map_complex_functions(self):
        """Test safe_map with complex functions."""
        # Lambda with multiple operations
        result = compat.safe_map(lambda x: x ** 2 + 1, [1, 2, 3])
        self.assertEqual(result, [2, 5, 10])

        # Function that returns tuples
        result = compat.safe_map(lambda x, y: (x, y), [1, 2], ['a', 'b'])
        self.assertEqual(result, [(1, 'a'), (2, 'b')])

    def test_safe_zip_basic(self):
        """Test basic safe_zip functionality."""
        # Two sequences
        result = compat.safe_zip([1, 2, 3], ['a', 'b', 'c'])
        expected = [(1, 'a'), (2, 'b'), (3, 'c')]
        self.assertEqual(result, expected)

        # Three sequences
        result = compat.safe_zip([1, 2], [3, 4], [5, 6])
        expected = [(1, 3, 5), (2, 4, 6)]
        self.assertEqual(result, expected)

    def test_safe_zip_empty_inputs(self):
        """Test safe_zip with empty inputs."""
        result = compat.safe_zip([], [])
        self.assertEqual(result, [])

        result = compat.safe_zip([], [], [])
        self.assertEqual(result, [])

    def test_safe_zip_length_mismatch(self):
        """Test safe_zip raises error on length mismatch."""
        with self.assertRaises(AssertionError) as context:
            compat.safe_zip([1, 2, 3], [4, 5])

        self.assertIn('length mismatch', str(context.exception))

    def test_safe_zip_single_sequence(self):
        """Test safe_zip with single sequence."""
        result = compat.safe_zip([1, 2, 3])
        expected = [(1,), (2,), (3,)]
        self.assertEqual(result, expected)

    def test_safe_zip_mixed_types(self):
        """Test safe_zip with mixed data types."""
        result = compat.safe_zip([1, 2], ['a', 'b'], [True, False])
        expected = [(1, 'a', True), (2, 'b', False)]
        self.assertEqual(result, expected)

    def test_unzip2_basic(self):
        """Test basic unzip2 functionality."""
        pairs = [(1, 'a'), (2, 'b'), (3, 'c')]
        first, second = compat.unzip2(pairs)

        self.assertEqual(first, (1, 2, 3))
        self.assertEqual(second, ('a', 'b', 'c'))

    def test_unzip2_empty(self):
        """Test unzip2 with empty input."""
        first, second = compat.unzip2([])
        self.assertEqual(first, ())
        self.assertEqual(second, ())

    def test_unzip2_single_pair(self):
        """Test unzip2 with single pair."""
        first, second = compat.unzip2([(42, 'answer')])
        self.assertEqual(first, (42,))
        self.assertEqual(second, ('answer',))

    def test_unzip2_mixed_types(self):
        """Test unzip2 with mixed data types."""
        pairs = [(1, 'a'), (2.5, 'b'), (None, 'c')]
        first, second = compat.unzip2(pairs)

        self.assertEqual(first, (1, 2.5, None))
        self.assertEqual(second, ('a', 'b', 'c'))

    def test_unzip2_return_types(self):
        """Test unzip2 returns proper tuple types."""
        pairs = [(1, 'a'), (2, 'b')]
        first, second = compat.unzip2(pairs)

        self.assertIsInstance(first, tuple)
        self.assertIsInstance(second, tuple)

    def test_unzip2_with_generator(self):
        """Test unzip2 with generator input."""

        def pair_generator():
            yield (1, 'a')
            yield (2, 'b')
            yield (3, 'c')

        first, second = compat.unzip2(pair_generator())
        self.assertEqual(first, (1, 2, 3))
        self.assertEqual(second, ('a', 'b', 'c'))


class TestFunctionWrapping(unittest.TestCase):
    """Test function wrapping and metadata handling."""

    def setUp(self):
        """Set up test environment."""
        pass

    def tearDown(self):
        """Clean up after tests."""
        pass

    def test_fun_name_basic(self):
        """Test fun_name function with regular functions."""

        def test_function():
            """Test function docstring."""
            pass

        name = compat.fun_name(test_function)
        self.assertEqual(name, 'test_function')

    def test_fun_name_lambda(self):
        """Test fun_name with lambda functions."""
        lambda_func = lambda x: x * 2
        name = compat.fun_name(lambda_func)
        self.assertEqual(name, '<lambda>')

    def test_fun_name_partial(self):
        """Test fun_name with partial functions."""

        def original_function(x, y):
            return x + y

        partial_func = partial(original_function, 10)
        name = compat.fun_name(partial_func)
        self.assertEqual(name, 'original_function')

    def test_fun_name_nested_partial(self):
        """Test fun_name with nested partial functions."""

        def base_function(x, y, z):
            return x + y + z

        partial1 = partial(base_function, 1)
        partial2 = partial(partial1, 2)

        name = compat.fun_name(partial2)
        self.assertEqual(name, 'base_function')

    def test_fun_name_no_name_attribute(self):
        """Test fun_name with objects without __name__."""

        class CallableClass:
            def __call__(self):
                pass

        callable_obj = CallableClass()
        name = compat.fun_name(callable_obj)
        self.assertEqual(name, '<unnamed function>')

    def test_wraps_basic(self):
        """Test basic wraps functionality."""

        def original_function():
            """Original function docstring."""
            return 42

        @compat.wraps(original_function)
        def wrapper():
            return original_function()

        self.assertEqual(wrapper.__name__, 'original_function')
        self.assertEqual(wrapper.__doc__, 'Original function docstring.')
        self.assertEqual(wrapper.__wrapped__, original_function)

    def test_wraps_with_namestr(self):
        """Test wraps with custom name string."""

        def original_function():
            pass

        @compat.wraps(original_function, namestr="wrapped_{fun}")
        def wrapper():
            pass

        self.assertEqual(wrapper.__name__, 'wrapped_original_function')

    def test_wraps_with_docstr(self):
        """Test wraps with custom docstring."""

        def original_function():
            """Original docstring."""
            pass

        @compat.wraps(original_function, docstr="Wrapper for {fun}: {doc}")
        def wrapper():
            pass

        expected_doc = "Wrapper for original_function: Original docstring."
        self.assertEqual(wrapper.__doc__, expected_doc)

    def test_wraps_with_kwargs(self):
        """Test wraps with additional keyword arguments."""

        def original_function():
            pass

        @compat.wraps(original_function,
                      docstr="Function {fun} version {version}",
                      version="1.0")
        def wrapper():
            pass

        expected_doc = "Function original_function version 1.0"
        self.assertEqual(wrapper.__doc__, expected_doc)

    def test_wraps_preserves_annotations(self):
        """Test wraps preserves function annotations."""

        def original_function(x: int, y: str) -> float:
            return float(x)

        @compat.wraps(original_function)
        def wrapper(x: int, y: str) -> float:
            return original_function(x, y)

        self.assertEqual(wrapper.__annotations__, original_function.__annotations__)

    def test_wraps_preserves_dict(self):
        """Test wraps preserves function __dict__."""

        def original_function():
            pass

        original_function.custom_attr = "test_value"
        original_function.another_attr = 42

        @compat.wraps(original_function)
        def wrapper():
            pass

        self.assertEqual(wrapper.custom_attr, "test_value")
        self.assertEqual(wrapper.another_attr, 42)

    def test_wraps_handles_exceptions(self):
        """Test wraps handles exceptions gracefully."""
        # Create a mock object that raises exceptions
        mock_func = Mock()
        mock_func.__name__ = Mock(side_effect=Exception("Test exception"))

        @compat.wraps(mock_func)
        def wrapper():
            pass

        # Should not raise exception, just continue
        self.assertTrue(callable(wrapper))

    def test_wraps_with_missing_attributes(self):
        """Test wraps handles missing attributes gracefully."""

        class MinimalCallable:
            pass

        minimal_func = MinimalCallable()

        @compat.wraps(minimal_func)
        def wrapper():
            pass

        # Should handle missing attributes without crashing
        self.assertTrue(callable(wrapper))


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""

    def setUp(self):
        """Set up test environment."""
        pass

    def tearDown(self):
        """Clean up after tests."""
        pass

    def test_safe_map_with_none_inputs(self):
        """Test safe_map behavior with None inputs."""
        # Function that handles None
        result = compat.safe_map(lambda x: x if x is not None else 'default',
                                 [1, None, 3])
        self.assertEqual(result, [1, 'default', 3])

    def test_safe_map_with_zero_length(self):
        """Test safe_map with zero-length sequences."""
        result = compat.safe_map(str, [])
        self.assertEqual(result, [])

    def test_safe_zip_with_none_values(self):
        """Test safe_zip with None values."""
        result = compat.safe_zip([1, None, 3], [4, 5, None])
        expected = [(1, 4), (None, 5), (3, None)]
        self.assertEqual(result, expected)

    def test_unzip2_with_none_values(self):
        """Test unzip2 with None values."""
        pairs = [(1, None), (None, 'a'), (2, 'b')]
        first, second = compat.unzip2(pairs)

        self.assertEqual(first, (1, None, 2))
        self.assertEqual(second, (None, 'a', 'b'))

    def test_large_sequences(self):
        """Test utility functions with large sequences."""
        large_seq1 = list(range(10000))
        large_seq2 = list(range(10000, 20000))

        # Test safe_map
        result = compat.safe_map(lambda x, y: x + y, large_seq1[:100], large_seq2[:100])
        self.assertEqual(len(result), 100)
        self.assertEqual(result[0], 10000)  # 0 + 10000

        # Test safe_zip
        result = compat.safe_zip(large_seq1[:100], large_seq2[:100])
        self.assertEqual(len(result), 100)
        self.assertEqual(result[0], (0, 10000))

        # Test unzip2
        pairs = list(zip(large_seq1[:100], large_seq2[:100]))
        first, second = compat.unzip2(pairs)
        self.assertEqual(len(first), 100)
        self.assertEqual(len(second), 100)

    # def test_to_concrete_aval_edge_cases(self):
    #     """Test to_concrete_aval with edge cases."""
    #     # Test with None
    #     result = compat.to_concrete_aval(None)
    #     self.assertIsNone(result)
    #
    #     # Test with regular Python objects
    #     result = compat.to_concrete_aval(42)
    #     self.assertEqual(result, 42)
    #
    #     result = compat.to_concrete_aval("string")
    #     self.assertEqual(result, "string")
    #
    #     # Test with list
    #     test_list = [1, 2, 3]
    #     result = compat.to_concrete_aval(test_list)
    #     self.assertEqual(result, test_list)

    def test_function_name_edge_cases(self):
        """Test fun_name with edge cases."""
        # Built-in function
        name = compat.fun_name(len)
        self.assertEqual(name, 'len')

        # Method
        name = compat.fun_name(str.upper)
        self.assertEqual(name, 'upper')

        # Nested function
        def outer():
            def inner():
                pass

            return inner

        inner_func = outer()
        name = compat.fun_name(inner_func)
        self.assertEqual(name, 'inner')

    def test_concurrent_usage(self):
        """Test thread safety of utility functions."""
        import threading

        results = []
        errors = []

        def worker():
            try:
                # Test safe_map in concurrent context
                for i in range(100):
                    result = compat.safe_map(lambda x: x * 2, [1, 2, 3])
                    results.append(result)

                # Test safe_zip
                for i in range(100):
                    result = compat.safe_zip([1, 2], [3, 4])

                # Test unzip2
                for i in range(100):
                    first, second = compat.unzip2([(1, 'a'), (2, 'b')])

            except Exception as e:
                errors.append(e)

        threads = []
        for _ in range(5):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Should have no errors
        self.assertEqual(len(errors), 0)
        # Should have expected number of results
        self.assertEqual(len(results), 500)  # 5 threads * 100 iterations


class TestTypeHints(unittest.TestCase):
    """Test type hints and generic type variables."""

    def test_type_variables_defined(self):
        """Test that type variables are properly defined."""
        # Check TypeVars are available in the module
        self.assertTrue(hasattr(compat, 'T'))
        self.assertTrue(hasattr(compat, 'T1'))
        self.assertTrue(hasattr(compat, 'T2'))
        self.assertTrue(hasattr(compat, 'T3'))

    def test_unzip2_type_preservation(self):
        """Test unzip2 preserves type information."""
        # Test with specific types
        int_str_pairs = [(1, 'a'), (2, 'b'), (3, 'c')]
        ints, strs = compat.unzip2(int_str_pairs)

        # Verify types are preserved
        self.assertTrue(all(isinstance(x, int) for x in ints))
        self.assertTrue(all(isinstance(x, str) for x in strs))

    def test_safe_functions_with_different_types(self):
        """Test safe functions work with different types."""
        # Test safe_map with different input types
        mixed_inputs = [1, 2.5, '3']
        result = compat.safe_map(str, mixed_inputs)
        expected = ['1', '2.5', '3']
        self.assertEqual(result, expected)

        # Test safe_zip with different types
        result = compat.safe_zip([1, 2], [3.14, 2.71], ['a', 'b'])
        expected = [(1, 3.14, 'a'), (2, 2.71, 'b')]
        self.assertEqual(result, expected)


class TestIntegration(unittest.TestCase):
    """Integration tests with JAX functionality."""

    def test_jax_integration(self):
        """Test integration with JAX arrays and operations."""
        # Create JAX arrays
        arr1 = jnp.array([1, 2, 3])
        arr2 = jnp.array([4, 5, 6])

        # Use safe_map with JAX operations
        result = compat.safe_map(lambda x, y: x + y, arr1.tolist(), arr2.tolist())
        expected = [5, 7, 9]
        self.assertEqual(result, expected)

        # Use safe_zip with JAX arrays
        result = compat.safe_zip(arr1.tolist(), arr2.tolist())
        expected = [(1, 4), (2, 5), (3, 6)]
        self.assertEqual(result, expected)

    def test_with_jax_transformations(self):
        """Test compatibility with JAX transformations."""

        def test_function(x):
            # Use utility functions inside JAX-transformable code
            pairs = [(x, x + 1), (x + 2, x + 3)]
            first, second = compat.unzip2(pairs)
            return jnp.array(first), jnp.array(second)

        # Test function works
        result1, result2 = test_function(10.0)
        np.testing.assert_array_equal(result1, [10.0, 12.0])
        np.testing.assert_array_equal(result2, [11.0, 13.0])

        # Test with JAX transformations
        jitted_func = jax.jit(test_function)
        result1, result2 = jitted_func(10.0)
        np.testing.assert_array_equal(result1, [10.0, 12.0])
        np.testing.assert_array_equal(result2, [11.0, 13.0])


class TestModuleStructure(unittest.TestCase):
    """Test module structure and __all__ exports."""

    def test_all_exports(self):
        """Test that __all__ contains expected exports."""
        expected_exports = [
            'ClosedJaxpr', 'Primitive', 'extend_axis_env_nd', 'jaxpr_as_fun',
            'get_aval', 'Tracer', 'to_concrete_aval', 'safe_map', 'safe_zip',
            'unzip2', 'wraps', 'Device', 'wrap_init', 'Var', 'JaxprEqn',
            'Jaxpr', 'Literal'
        ]

        for export in expected_exports:
            self.assertIn(export, compat.__all__,
                          f"{export} should be in __all__")
            self.assertTrue(hasattr(compat, export),
                            f"{export} should be available in module")

    def test_no_unexpected_exports(self):
        """Test that no private functions are exported."""
        for name in compat.__all__:
            self.assertFalse(name.startswith('_'),
                             f"Private name {name} should not be in __all__")

    def test_module_docstring(self):
        """Test module has proper docstring."""
        self.assertIsNotNone(compat.__doc__)
        self.assertIn('Compatibility layer', compat.__doc__)


if __name__ == '__main__':
    unittest.main(verbosity=2)
