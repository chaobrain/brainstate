#!/usr/bin/env python3
"""
Unit tests for the DeprecatedModule class itself.
"""

import unittest
import warnings
import sys
import os
from types import ModuleType

# Add brainstate to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from brainstate._deprecation import DeprecatedModule, create_deprecated_module_proxy


class MockReplacementModule:
    """Mock module for testing."""
    __all__ = ['test_function', 'test_variable', 'test_class']

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
        self.assertEqual(self.deprecated.__all__, ['test_function', 'test_variable', 'test_class'])

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
            self.assertIn('_deprecation_v2_test.py', w[0].filename)


if __name__ == '__main__':
    unittest.main(verbosity=2)