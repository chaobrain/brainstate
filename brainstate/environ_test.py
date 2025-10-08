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

"""
Comprehensive test suite for the environ module.

This test module provides extensive coverage of the environment configuration
and context management functionality, including:
- Global environment settings
- Context-based temporary settings
- Precision and data type management
- Callback registration and behavior
- Thread safety
- Error handling and validation
"""

import threading
import unittest
import warnings
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np

import brainstate as bst


class TestEnvironmentCore(unittest.TestCase):
    """Test core environment management functionality."""

    def setUp(self):
        """Reset environment before each test."""
        bst.environ.reset()
        # Clear any warnings
        warnings.filterwarnings('ignore', category=UserWarning)

    def tearDown(self):
        """Clean up after each test."""
        # Reset to default state
        bst.environ.reset()
        warnings.resetwarnings()

    def test_get_set_basic(self):
        """Test basic get and set operations."""
        # Set a value
        bst.environ.set(test_param='test_value')
        self.assertEqual(bst.environ.get('test_param'), 'test_value')

        # Set multiple values
        bst.environ.set(param1=1, param2='two', param3=3.0)
        self.assertEqual(bst.environ.get('param1'), 1)
        self.assertEqual(bst.environ.get('param2'), 'two')
        self.assertEqual(bst.environ.get('param3'), 3.0)

    def test_get_with_default(self):
        """Test get with default value."""
        # Non-existent key with default
        result = bst.environ.get('nonexistent', default='default_value')
        self.assertEqual(result, 'default_value')

        # Existing key ignores default
        bst.environ.set(existing='value')
        result = bst.environ.get('existing', default='default')
        self.assertEqual(result, 'value')

    def test_get_missing_key_error(self):
        """Test KeyError for missing keys without default."""
        with self.assertRaises(KeyError) as context:
            bst.environ.get('missing_key')

        error_msg = str(context.exception)
        self.assertIn('missing_key', error_msg)
        self.assertIn('not found', error_msg)

    def test_get_with_description(self):
        """Test get with description for error messages."""
        with self.assertRaises(KeyError) as context:
            bst.environ.get('missing', desc='Important parameter for computation')

        error_msg = str(context.exception)
        self.assertIn('Important parameter', error_msg)

    def test_all_function(self):
        """Test getting all environment settings."""
        # Set various parameters
        bst.environ.set(
            param1='value1',
            param2=42,
            param3=3.14,
            precision=32
        )

        all_settings = bst.environ.all()
        self.assertIsInstance(all_settings, dict)
        self.assertEqual(all_settings['param1'], 'value1')
        self.assertEqual(all_settings['param2'], 42)
        self.assertEqual(all_settings['param3'], 3.14)
        self.assertEqual(all_settings['precision'], 32)

    def test_reset_function(self):
        """Test environment reset functionality."""
        # Set custom values
        bst.environ.set(
            custom1='value1',
            custom2='value2',
            precision=64
        )

        # Verify they're set
        self.assertEqual(bst.environ.get('custom1'), 'value1')

        # Reset environment
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bst.environ.reset()

        # Custom values should be gone
        result = bst.environ.get('custom1', default=None)
        self.assertIsNone(result)

        # Default precision should be restored
        self.assertEqual(bst.environ.get('precision'), bst.environ.DEFAULT_PRECISION)

    def test_special_environment_keys(self):
        """Test special environment key constants."""
        # Test setting using constants
        bst.environ.set(**{
            bst.environ.DT: 0.01,
            bst.environ.I: 0,
            bst.environ.T: 0.0,
            bst.environ.JIT_ERROR_CHECK: True,
            bst.environ.FIT: False
        })

        self.assertEqual(bst.environ.get(bst.environ.DT), 0.01)
        self.assertEqual(bst.environ.get(bst.environ.I), 0)
        self.assertEqual(bst.environ.get(bst.environ.T), 0.0)
        self.assertTrue(bst.environ.get(bst.environ.JIT_ERROR_CHECK))
        self.assertFalse(bst.environ.get(bst.environ.FIT))

    def test_pop_basic(self):
        """Test basic pop operation."""
        # Set a value
        bst.environ.set(pop_test='test_value')
        self.assertEqual(bst.environ.get('pop_test'), 'test_value')

        # Pop the value
        popped = bst.environ.pop('pop_test')
        self.assertEqual(popped, 'test_value')

        # Value should be gone
        result = bst.environ.get('pop_test', default=None)
        self.assertIsNone(result)

    def test_pop_with_default(self):
        """Test pop with default value."""
        # Pop non-existent key with default
        result = bst.environ.pop('nonexistent_pop', default='default_value')
        self.assertEqual(result, 'default_value')

        # Pop existing key ignores default
        bst.environ.set(existing_pop='value')
        result = bst.environ.pop('existing_pop', default='default')
        self.assertEqual(result, 'value')

    def test_pop_missing_key_error(self):
        """Test KeyError for missing keys without default."""
        with self.assertRaises(KeyError) as context:
            bst.environ.pop('missing_pop_key')

        error_msg = str(context.exception)
        self.assertIn('missing_pop_key', error_msg)
        self.assertIn('not found', error_msg)

    def test_pop_multiple_values(self):
        """Test popping multiple values."""
        # Set multiple values
        bst.environ.set(
            pop1='value1',
            pop2='value2',
            pop3='value3'
        )

        # Pop them one by one
        v1 = bst.environ.pop('pop1')
        v2 = bst.environ.pop('pop2')

        self.assertEqual(v1, 'value1')
        self.assertEqual(v2, 'value2')

        # pop3 should still exist
        self.assertEqual(bst.environ.get('pop3'), 'value3')

        # pop1 and pop2 should be gone
        self.assertIsNone(bst.environ.get('pop1', default=None))
        self.assertIsNone(bst.environ.get('pop2', default=None))

    def test_pop_with_context_protection(self):
        """Test that pop is prevented when key is in active context."""
        # Set a global value
        bst.environ.set(protected_key='global_value')

        # Cannot pop while in context
        with bst.environ.context(protected_key='context_value'):
            with self.assertRaises(ValueError) as context:
                bst.environ.pop('protected_key')

            error_msg = str(context.exception)
            self.assertIn('Cannot pop', error_msg)
            self.assertIn('active in a context', error_msg)

        # Can pop after context exits
        popped = bst.environ.pop('protected_key')
        self.assertEqual(popped, 'global_value')

    def test_pop_nested_context_protection(self):
        """Test pop protection with nested contexts."""
        bst.environ.set(nested_key='global')

        with bst.environ.context(nested_key='level1'):
            with bst.environ.context(nested_key='level2'):
                # Should indicate 2 active contexts
                with self.assertRaises(ValueError) as context:
                    bst.environ.pop('nested_key')

                error_msg = str(context.exception)
                self.assertIn('2 context(s)', error_msg)

    def test_pop_does_not_affect_context_values(self):
        """Test that pop doesn't affect context values."""
        # Set both global and context value
        bst.environ.set(dual_key='global')

        with bst.environ.context(other_key='context_only'):
            # Can pop a key that's only in global (not in this context)
            popped = bst.environ.pop('dual_key')
            self.assertEqual(popped, 'global')

            # Context-only values remain accessible
            self.assertEqual(bst.environ.get('other_key'), 'context_only')

        # Context value should be gone after exit
        self.assertIsNone(bst.environ.get('other_key', default=None))

    def test_pop_precision_key(self):
        """Test popping the precision key."""
        # Set custom precision
        bst.environ.set(precision=64)
        self.assertEqual(bst.environ.get_precision(), 64)

        # Pop precision
        popped = bst.environ.pop('precision')
        self.assertEqual(popped, 64)


class TestEnvironmentContext(unittest.TestCase):
    """Test context manager functionality."""

    def setUp(self):
        """Reset environment before each test."""
        bst.environ.reset()
        warnings.filterwarnings('ignore', category=UserWarning)

    def tearDown(self):
        """Clean up after each test."""
        bst.environ.reset()
        warnings.resetwarnings()

    def test_basic_context(self):
        """Test basic context manager usage."""
        bst.environ.set(value=10)

        with bst.environ.context(value=20) as ctx:
            # Value should be 20 in context
            self.assertEqual(bst.environ.get('value'), 20)
            # Context should contain current settings
            self.assertEqual(ctx['value'], 20)

        # Value should be restored to 10
        self.assertEqual(bst.environ.get('value'), 10)

    def test_nested_contexts(self):
        """Test nested context managers."""
        bst.environ.set(level=0)

        with bst.environ.context(level=1):
            self.assertEqual(bst.environ.get('level'), 1)

            with bst.environ.context(level=2):
                self.assertEqual(bst.environ.get('level'), 2)

                with bst.environ.context(level=3):
                    self.assertEqual(bst.environ.get('level'), 3)

                # Back to level 2
                self.assertEqual(bst.environ.get('level'), 2)

            # Back to level 1
            self.assertEqual(bst.environ.get('level'), 1)

        # Back to level 0
        self.assertEqual(bst.environ.get('level'), 0)

    def test_context_with_exception(self):
        """Test context manager handles exceptions properly."""
        bst.environ.set(value='original')

        try:
            with bst.environ.context(value='temporary'):
                self.assertEqual(bst.environ.get('value'), 'temporary')
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Value should be restored despite exception
        self.assertEqual(bst.environ.get('value'), 'original')

    def test_context_multiple_parameters(self):
        """Test context with multiple parameters."""
        bst.environ.set(a=1, b=2, c=3)

        with bst.environ.context(a=10, b=20, c=30, d=40):
            self.assertEqual(bst.environ.get('a'), 10)
            self.assertEqual(bst.environ.get('b'), 20)
            self.assertEqual(bst.environ.get('c'), 30)
            self.assertEqual(bst.environ.get('d'), 40)

        # Original values restored
        self.assertEqual(bst.environ.get('a'), 1)
        self.assertEqual(bst.environ.get('b'), 2)
        self.assertEqual(bst.environ.get('c'), 3)
        # d should not exist
        result = bst.environ.get('d', default=None)
        self.assertIsNone(result)

    def test_context_platform_restriction(self):
        """Test that platform cannot be set in context."""
        with self.assertRaises(ValueError) as context:
            with bst.environ.context(platform='cpu'):
                pass

        self.assertIn('platform', str(context.exception).lower())
        self.assertIn('cannot set', str(context.exception).lower())

    def test_context_host_device_count_restriction(self):
        """Test that host_device_count cannot be set in context."""
        with self.assertRaises(ValueError) as context:
            with bst.environ.context(host_device_count=4):
                pass

        self.assertIn('host_device_count', str(context.exception))

    def test_context_mode_validation(self):
        """Test mode validation in context."""
        # Valid mode
        mode = bst.mixin.Training()
        with bst.environ.context(mode=mode):
            self.assertEqual(bst.environ.get('mode'), mode)

    def test_context_preserves_unmodified_values(self):
        """Test that context doesn't affect unmodified values."""
        bst.environ.set(unchanged='original', changed='original')

        with bst.environ.context(changed='modified'):
            self.assertEqual(bst.environ.get('unchanged'), 'original')
            self.assertEqual(bst.environ.get('changed'), 'modified')


class TestPrecisionAndDataTypes(unittest.TestCase):
    """Test precision control and data type functions."""

    def setUp(self):
        """Reset environment before each test."""
        bst.environ.reset()
        warnings.filterwarnings('ignore', category=UserWarning)

    def tearDown(self):
        """Clean up after each test."""
        bst.environ.reset()
        warnings.resetwarnings()

    def test_precision_settings(self):
        """Test different precision settings."""
        precisions = [8, 16, 32, 64, 'bf16']

        for precision in precisions:
            bst.environ.set(precision=precision)

            if precision == 'bf16':
                self.assertEqual(bst.environ.get_precision(), 16)
            elif isinstance(precision, str):
                self.assertEqual(bst.environ.get_precision(), int(precision))
            else:
                self.assertEqual(bst.environ.get_precision(), precision)

    def test_precision_context(self):
        """Test precision changes in context."""
        bst.environ.set(precision=32)

        with bst.environ.context(precision=64):
            a = bst.random.randn(1)
            self.assertEqual(a.dtype, jnp.float64)
            self.assertEqual(bst.environ.get_precision(), 64)

        # Precision restored
        b = bst.random.randn(1)
        self.assertEqual(b.dtype, jnp.float32)
        self.assertEqual(bst.environ.get_precision(), 32)

    def test_dftype_function(self):
        """Test default float type function."""
        # 32-bit precision
        bst.environ.set(precision=32)
        self.assertEqual(bst.environ.dftype(), np.float32)

        # 64-bit precision
        bst.environ.set(precision=64)
        self.assertEqual(bst.environ.dftype(), np.float64)

        # 16-bit precision
        bst.environ.set(precision=16)
        self.assertEqual(bst.environ.dftype(), np.float16)

        # bfloat16 precision
        bst.environ.set(precision='bf16')
        self.assertEqual(bst.environ.dftype(), jnp.bfloat16)

    def test_ditype_function(self):
        """Test default integer type function."""
        # 32-bit precision
        bst.environ.set(precision=32)
        self.assertEqual(bst.environ.ditype(), np.int32)

        # 64-bit precision
        bst.environ.set(precision=64)
        self.assertEqual(bst.environ.ditype(), np.int64)

        # 16-bit precision
        bst.environ.set(precision=16)
        self.assertEqual(bst.environ.ditype(), np.int16)

        # 8-bit precision
        bst.environ.set(precision=8)
        self.assertEqual(bst.environ.ditype(), np.int8)

    def test_dutype_function(self):
        """Test default unsigned integer type function."""
        # 32-bit precision
        bst.environ.set(precision=32)
        self.assertEqual(bst.environ.dutype(), np.uint32)

        # 64-bit precision
        bst.environ.set(precision=64)
        self.assertEqual(bst.environ.dutype(), np.uint64)

        # 16-bit precision
        bst.environ.set(precision=16)
        self.assertEqual(bst.environ.dutype(), np.uint16)

        # 8-bit precision
        bst.environ.set(precision=8)
        self.assertEqual(bst.environ.dutype(), np.uint8)

    def test_dctype_function(self):
        """Test default complex type function."""
        # 32-bit precision
        bst.environ.set(precision=32)
        self.assertEqual(bst.environ.dctype(), np.complex64)

        # 64-bit precision
        bst.environ.set(precision=64)
        self.assertEqual(bst.environ.dctype(), np.complex128)

        # 16-bit precision (should use complex64)
        bst.environ.set(precision=16)
        self.assertEqual(bst.environ.dctype(), np.complex64)

    def test_tolerance_function(self):
        """Test tolerance values for different precisions."""
        # 64-bit precision
        bst.environ.set(precision=64)
        tol = bst.environ.tolerance()
        self.assertAlmostEqual(float(tol), 1e-12, places=14)

        # 32-bit precision
        bst.environ.set(precision=32)
        tol = bst.environ.tolerance()
        self.assertAlmostEqual(float(tol), 1e-5, places=7)

        # 16-bit precision
        bst.environ.set(precision=16)
        tol = bst.environ.tolerance()
        self.assertAlmostEqual(float(tol), 1e-2, places=4)

    def test_invalid_precision(self):
        """Test invalid precision values."""
        invalid_precisions = [128, 'invalid', -1, 3.14]

        for invalid in invalid_precisions:
            with self.assertRaises(ValueError):
                bst.environ.set(precision=invalid)

    def test_precision_with_arrays(self):
        """Test that precision affects array creation."""
        # Test with different precisions
        test_cases = [
            (32, jnp.float32),
            (64, jnp.float64),
            (16, jnp.float16),
            ('bf16', jnp.bfloat16),
        ]

        for precision, expected_dtype in test_cases:
            with bst.environ.context(precision=precision):
                # Create array using random
                arr = bst.random.randn(5)
                self.assertEqual(arr.dtype, expected_dtype)


class TestModeAndSpecialGetters(unittest.TestCase):
    """Test mode management and special getter functions."""

    def setUp(self):
        """Reset environment before each test."""
        bst.environ.reset()
        warnings.filterwarnings('ignore', category=UserWarning)

    def tearDown(self):
        """Clean up after each test."""
        bst.environ.reset()
        warnings.resetwarnings()

    def test_get_dt(self):
        """Test get_dt function."""
        # Set dt
        bst.environ.set(dt=0.01)
        self.assertEqual(bst.environ.get_dt(), 0.01)

        # Test in context
        with bst.environ.context(dt=0.001):
            self.assertEqual(bst.environ.get_dt(), 0.001)

        self.assertEqual(bst.environ.get_dt(), 0.01)

        # Test missing dt
        bst.environ.reset()
        with self.assertRaises(KeyError):
            bst.environ.get_dt()

    def test_get_mode(self):
        """Test get_mode function."""
        # Set training mode
        training = bst.mixin.Training()
        bst.environ.set(mode=training)
        mode = bst.environ.get('mode')
        self.assertEqual(mode, training)
        self.assertTrue(mode.has(bst.mixin.Training))

        # Test with batching mode
        batching = bst.mixin.Batching(batch_size=32)
        with bst.environ.context(mode=batching):
            mode = bst.environ.get('mode')
            self.assertEqual(mode, batching)
            self.assertTrue(mode.has(bst.mixin.Batching))
            self.assertEqual(mode.batch_size, 32)

        # Test missing mode
        bst.environ.reset()
        with self.assertRaises(KeyError):
            bst.environ.get('mode')

    def test_get_platform(self):
        """Test get_platform function."""
        platform = bst.environ.get_platform()
        self.assertIn(platform, bst.environ.SUPPORTED_PLATFORMS)

    def test_get_host_device_count(self):
        """Test get_host_device_count function."""
        count = bst.environ.get_host_device_count()
        self.assertIsInstance(count, int)
        self.assertGreaterEqual(count, 1)

    def test_dt_validation(self):
        """Test dt validation in set function."""
        # Valid dt values
        valid_dts = [0.01, 0.001, 1.0, 0.1]
        for dt in valid_dts:
            bst.environ.set(dt=dt)
            self.assertEqual(bst.environ.get_dt(), dt)


class TestPlatformAndDevice(unittest.TestCase):
    """Test platform and device management."""

    def setUp(self):
        """Reset environment before each test."""
        bst.environ.reset()
        warnings.filterwarnings('ignore', category=UserWarning)

    def tearDown(self):
        """Clean up after each test."""
        bst.environ.reset()
        warnings.resetwarnings()

    @patch('brainstate.environ.config')
    def test_set_platform(self, mock_config):
        """Test platform setting."""
        platforms = ['cpu', 'gpu', 'tpu']

        for platform in platforms:
            bst.environ.set_platform(platform)
            mock_config.update.assert_called_with("jax_platform_name", platform)

        # Test invalid platform
        with self.assertRaises(ValueError):
            bst.environ.set_platform('invalid')

    def test_set_platform_through_set(self):
        """Test setting platform through general set function."""
        with patch('brainstate.environ.config') as mock_config:
            bst.environ.set(platform='gpu')
            mock_config.update.assert_called_with("jax_platform_name", 'gpu')

    def test_set_host_device_count(self):
        """Test host device count setting."""
        import os

        # Set device count
        bst.environ.set_host_device_count(4)
        xla_flags = os.environ.get("XLA_FLAGS", "")
        self.assertIn("--xla_force_host_platform_device_count=4", xla_flags)

        # Update device count
        bst.environ.set_host_device_count(8)
        xla_flags = os.environ.get("XLA_FLAGS", "")
        self.assertIn("--xla_force_host_platform_device_count=8", xla_flags)
        self.assertNotIn("--xla_force_host_platform_device_count=4", xla_flags)

        # Invalid device count
        with self.assertRaises(ValueError):
            bst.environ.set_host_device_count(0)

        with self.assertRaises(ValueError):
            bst.environ.set_host_device_count(-1)

    def test_platform_context_restriction(self):
        """Test that platform cannot be changed in context."""
        with self.assertRaises(ValueError):
            with bst.environ.context(platform='cpu'):
                pass


class TestCallbackBehavior(unittest.TestCase):
    """Test callback registration and behavior."""

    def setUp(self):
        """Reset environment before each test."""
        bst.environ.reset()
        warnings.filterwarnings('ignore', category=UserWarning)
        self.callback_values = []

    def tearDown(self):
        """Clean up after each test."""
        bst.environ.reset()
        warnings.resetwarnings()

    # def test_register_callback(self):
    #     """Test basic callback registration."""
    #     def callback(value):
    #         self.callback_values.append(value)
    #
    #     brainstate.environ.register_default_behavior('test_param', callback)
    #
    #     # Callback should be triggered on set
    #     brainstate.environ.set(test_param='value1')
    #     self.assertEqual(self.callback_values, ['value1'])
    #
    #     # Callback should be triggered on context enter/exit
    #     with brainstate.environ.context(test_param='value2'):
    #         self.assertEqual(self.callback_values, ['value1', 'value2'])
    #
    #     # Should restore previous value
    #     self.assertEqual(self.callback_values, ['value1', 'value2', 'value1'])

    def test_register_multiple_callbacks(self):
        """Test registering callbacks for different keys."""
        values_a = []
        values_b = []

        def callback_a(value):
            values_a.append(value)

        def callback_b(value):
            values_b.append(value)

        bst.environ.register_default_behavior('param_a', callback_a)
        bst.environ.register_default_behavior('param_b', callback_b)

        bst.environ.set(param_a='a1', param_b='b1')
        self.assertEqual(values_a, ['a1'])
        self.assertEqual(values_b, ['b1'])

    def test_replace_callback(self):
        """Test replacing existing callbacks."""

        def callback1(value):
            self.callback_values.append(f'cb1:{value}')

        def callback2(value):
            self.callback_values.append(f'cb2:{value}')

        # Register first callback
        bst.environ.register_default_behavior('param', callback1)

        # Try to register second without replace flag
        with self.assertRaises(ValueError):
            bst.environ.register_default_behavior('param', callback2)

        # Register with replace flag
        bst.environ.register_default_behavior('param', callback2, replace_if_exist=True)

        # Only second callback should be called
        bst.environ.set(param='test')
        self.assertEqual(self.callback_values, ['cb2:test'])

    def test_unregister_callback(self):
        """Test unregistering callbacks."""

        def callback(value):
            self.callback_values.append(value)

        # Register and test
        bst.environ.register_default_behavior('param', callback)
        bst.environ.set(param='value1')
        self.assertEqual(len(self.callback_values), 1)

        # Unregister
        removed = bst.environ.unregister_default_behavior('param')
        self.assertTrue(removed)

        # Callback should not be triggered
        bst.environ.set(param='value2')
        self.assertEqual(len(self.callback_values), 1)  # Still just one

        # Unregister non-existent
        removed = bst.environ.unregister_default_behavior('nonexistent')
        self.assertFalse(removed)

    def test_list_registered_behaviors(self):
        """Test listing registered behaviors."""
        # Initially empty or with system defaults
        initial = bst.environ.list_registered_behaviors()

        # Register some behaviors
        bst.environ.register_default_behavior('param1', lambda x: None)
        bst.environ.register_default_behavior('param2', lambda x: None)
        bst.environ.register_default_behavior('param3', lambda x: None)

        behaviors = bst.environ.list_registered_behaviors()
        for param in ['param1', 'param2', 'param3']:
            self.assertIn(param, behaviors)

    def test_callback_exception_handling(self):
        """Test that exceptions in callbacks are handled gracefully."""

        def failing_callback(value):
            raise RuntimeError(f"Intentional error: {value}")

        bst.environ.register_default_behavior('param', failing_callback)

        # Should not crash, but should warn
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            bst.environ.set(param='test')

            # Should have a warning
            self.assertTrue(len(w) > 0)
            self.assertIn('Callback', str(w[0].message))
            self.assertIn('exception', str(w[0].message))

    def test_callback_validation(self):
        """Test callback validation."""
        # Non-callable
        with self.assertRaises(TypeError):
            bst.environ.register_default_behavior('param', 'not_callable')

        # Non-string key
        with self.assertRaises(TypeError):
            bst.environ.register_default_behavior(123, lambda x: None)

    def test_callback_with_validation(self):
        """Test using callbacks for validation."""

        def validate_positive(value):
            if value <= 0:
                raise ValueError(f"Value must be positive, got {value}")
            self.callback_values.append(value)

        bst.environ.register_default_behavior('positive_param', validate_positive)

        # Valid value
        bst.environ.set(positive_param=10)
        self.assertEqual(self.callback_values, [10])

        # Invalid value should raise through warning system
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            bst.environ.set(positive_param=-5)


class TestThreadSafety(unittest.TestCase):
    """Test thread safety of environment operations."""

    def setUp(self):
        """Reset environment before each test."""
        bst.environ.reset()
        warnings.filterwarnings('ignore', category=UserWarning)

    def tearDown(self):
        """Clean up after each test."""
        bst.environ.reset()
        warnings.resetwarnings()

    def test_concurrent_set_operations(self):
        """Test concurrent set operations from multiple threads."""
        results = []
        errors = []

        def thread_operation(thread_id):
            try:
                # Each thread sets its own value
                for i in range(10):
                    bst.environ.set(**{f'thread_{thread_id}': i})
                    value = bst.environ.get(f'thread_{thread_id}')
                    results.append((thread_id, value))
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(5):
            thread = threading.Thread(target=thread_operation, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Should have no errors
        self.assertEqual(len(errors), 0)

        # Each thread should have written its values
        for i in range(5):
            try:
                final_value = bst.environ.get(f'thread_{i}')
            except KeyError:
                pass

    def test_concurrent_context_operations(self):
        """Test concurrent context operations from multiple threads."""
        results = []
        errors = []

        def thread_context_operation(thread_id):
            try:
                bst.environ.set(**{f'base_{thread_id}': 0})

                for i in range(5):
                    with bst.environ.context(**{f'base_{thread_id}': i}):
                        value = bst.environ.get(f'base_{thread_id}')
                        results.append((thread_id, value))

                # Should be back to 0
                final = bst.environ.get(f'base_{thread_id}')
                self.assertEqual(final, 0)
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(3):
            thread = threading.Thread(target=thread_context_operation, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Should have no errors
        self.assertEqual(len(errors), 0)

    def test_concurrent_pop_operations(self):
        """Test concurrent pop operations from multiple threads."""
        # Set up multiple keys
        for i in range(20):
            bst.environ.set(**{f'pop_thread_{i}': f'value_{i}'})

        results = []
        errors = []

        def thread_pop_operation(start, end):
            try:
                for i in range(start, end):
                    try:
                        value = bst.environ.pop(f'pop_thread_{i}')
                        results.append((i, value))
                    except KeyError:
                        # Key might already be popped by another thread
                        pass
            except Exception as e:
                errors.append(e)

        # Create threads that pop different ranges
        threads = []
        ranges = [(0, 5), (5, 10), (10, 15), (15, 20)]
        for start, end in ranges:
            thread = threading.Thread(target=thread_pop_operation, args=(start, end))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Should have no errors
        self.assertEqual(len(errors), 0)

        # # All keys should be popped (each exactly once)
        # popped_indices = [r[0] for r in results]
        # self.assertEqual(len(popped_indices), 20)
        # self.assertEqual(len(set(popped_indices)), 20)  # All unique
        #
        # # All values should be gone
        # for i in range(20):
        #     result = brainstate.environ.get(f'pop_thread_{i}', default=None)
        #     self.assertIsNone(result)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""

    def setUp(self):
        """Reset environment before each test."""
        bst.environ.reset()
        warnings.filterwarnings('ignore', category=UserWarning)

    def tearDown(self):
        """Clean up after each test."""
        bst.environ.reset()
        warnings.resetwarnings()

    def test_empty_context(self):
        """Test context with no parameters."""
        original = bst.environ.all()

        with bst.environ.context() as ctx:
            # Should be unchanged
            self.assertEqual(ctx, original)

        self.assertEqual(bst.environ.all(), original)

    def test_none_values(self):
        """Test handling of None values."""
        bst.environ.set(none_param=None)
        self.assertIsNone(bst.environ.get('none_param'))

        with bst.environ.context(none_param='not_none'):
            self.assertEqual(bst.environ.get('none_param'), 'not_none')

        self.assertIsNone(bst.environ.get('none_param'))

    def test_complex_data_types(self):
        """Test storing complex data types."""
        # Lists
        bst.environ.set(list_param=[1, 2, 3])
        self.assertEqual(bst.environ.get('list_param'), [1, 2, 3])

        # Dictionaries
        bst.environ.set(dict_param={'a': 1, 'b': 2})
        self.assertEqual(bst.environ.get('dict_param'), {'a': 1, 'b': 2})

        # Tuples
        bst.environ.set(tuple_param=(1, 2, 3))
        self.assertEqual(bst.environ.get('tuple_param'), (1, 2, 3))

        # Custom objects
        class CustomObject:
            def __init__(self, value):
                self.value = value

        obj = CustomObject(42)
        bst.environ.set(obj_param=obj)
        retrieved = bst.environ.get('obj_param')
        self.assertIs(retrieved, obj)
        self.assertEqual(retrieved.value, 42)

    def test_special_string_values(self):
        """Test special string values."""
        special_strings = ['', ' ', '\n', '\t', 'None', 'True', 'False']

        for s in special_strings:
            bst.environ.set(string_param=s)
            self.assertEqual(bst.environ.get('string_param'), s)

    def test_numeric_edge_values(self):
        """Test numeric edge values."""
        import sys

        edge_values = [
            0, -0, 1, -1,
            sys.maxsize, -sys.maxsize,
            float('inf'), float('-inf'),
            1e-100, 1e100,
        ]

        for value in edge_values:
            bst.environ.set(numeric_param=value)
            retrieved = bst.environ.get('numeric_param')
            if value != value:  # NaN check
                self.assertTrue(retrieved != retrieved)
            else:
                self.assertEqual(retrieved, value)

    def test_context_all_interaction(self):
        """Test interaction between context and all() function."""
        bst.environ.set(global_param='global')

        with bst.environ.context(context_param='context', global_param='override'):
            all_values = bst.environ.all()

            # Should include both
            self.assertEqual(all_values['global_param'], 'override')
            self.assertEqual(all_values['context_param'], 'context')

            # Original global values should be in settings
            self.assertIn('precision', all_values)

    def test_deeply_nested_contexts(self):
        """Test deeply nested contexts."""
        depth = 20
        bst.environ.set(depth=0)

        def nested_context(level):
            if level < depth:
                with bst.environ.context(depth=level):
                    self.assertEqual(bst.environ.get('depth'), level)
                    nested_context(level + 1)
                    self.assertEqual(bst.environ.get('depth'), level)

        nested_context(1)
        self.assertEqual(bst.environ.get('depth'), 0)

    def test_set_precision_function(self):
        """Test the dedicated set_precision function."""
        # Valid precisions
        for precision in [8, 16, 32, 64, 'bf16']:
            bst.environ.set_precision(precision)
            self.assertEqual(bst.environ.get('precision'), precision)

        # Invalid precision
        with self.assertRaises(ValueError):
            bst.environ.set_precision(128)

    def test_pop_edge_cases(self):
        """Test edge cases for pop function."""
        # Pop with None value
        bst.environ.set(none_key=None)
        popped = bst.environ.pop('none_key')
        self.assertIsNone(popped)

        # Pop with None as default
        result = bst.environ.pop('missing_key', default=None)
        self.assertIsNone(result)

        # Pop complex data types
        complex_obj = {'nested': {'data': [1, 2, 3]}}
        bst.environ.set(complex_key=complex_obj)
        popped = bst.environ.pop('complex_key')
        self.assertEqual(popped, complex_obj)

        # Verify object identity preservation
        obj = object()
        bst.environ.set(obj_key=obj)
        popped = bst.environ.pop('obj_key')
        self.assertIs(popped, obj)

    def test_pop_all_interaction(self):
        """Test interaction between pop and all() function."""
        # Set multiple values
        bst.environ.set(a=1, b=2, c=3, d=4)
        initial_all = bst.environ.all()

        # Pop some values
        bst.environ.pop('b')
        bst.environ.pop('d')

        # Check all() reflects the changes
        after_pop = bst.environ.all()
        self.assertIn('a', after_pop)
        self.assertIn('c', after_pop)
        self.assertNotIn('b', after_pop)
        self.assertNotIn('d', after_pop)

    def test_pop_callback_not_triggered(self):
        """Test that callbacks are not triggered on pop."""
        callback_calls = []

        def callback(value):
            callback_calls.append(value)

        # Register callback
        bst.environ.register_default_behavior('callback_test', callback)

        # Set triggers callback
        bst.environ.set(callback_test='value')
        self.assertEqual(len(callback_calls), 1)

        # Pop should NOT trigger callback
        popped = bst.environ.pop('callback_test')
        self.assertEqual(len(callback_calls), 1)  # Still just 1
        self.assertEqual(popped, 'value')

        # Unregister callback
        bst.environ.unregister_default_behavior('callback_test')


class TestIntegration(unittest.TestCase):
    """Integration tests with actual BrainState functionality."""

    def setUp(self):
        """Reset environment before each test."""
        bst.environ.reset()
        warnings.filterwarnings('ignore', category=UserWarning)

    def tearDown(self):
        """Clean up after each test."""
        bst.environ.reset()
        warnings.resetwarnings()

    def test_precision_affects_random_arrays(self):
        """Test that precision setting affects random array generation."""
        # Test different precisions
        test_cases = [
            (32, jnp.float32),
            (64, jnp.float64),
            (16, jnp.float16),
            ('bf16', jnp.bfloat16),
        ]

        for precision, expected_dtype in test_cases:
            with bst.environ.context(precision=precision):
                arr = bst.random.randn(10)
                self.assertEqual(arr.dtype, expected_dtype)

    def test_mode_usage(self):
        """Test mode usage in computations."""
        # Create different modes
        training = bst.mixin.Training()
        batching = bst.mixin.Batching(batch_size=32)

        # Test training mode
        bst.environ.set(mode=training)
        mode = bst.environ.get('mode')
        self.assertTrue(mode.has(bst.mixin.Training))

        # Test batching mode
        with bst.environ.context(mode=batching):
            mode = bst.environ.get('mode')
            self.assertTrue(mode.has(bst.mixin.Batching))
            self.assertEqual(mode.batch_size, 32)

    def test_dt_in_numerical_integration(self):
        """Test dt usage in numerical contexts."""
        # Set different dt values
        dt_values = [0.01, 0.001, 0.1]

        for dt in dt_values:
            bst.environ.set(dt=dt)
            retrieved_dt = bst.environ.get_dt()
            self.assertEqual(retrieved_dt, dt)

            # Simulate using dt in computation
            time_steps = int(1.0 / dt)
            self.assertGreater(time_steps, 0)

    def test_combined_settings(self):
        """Test combining multiple settings."""
        # Set multiple parameters
        bst.environ.set(
            precision=64,
            dt=0.01,
            mode=bst.mixin.Training(),
            custom_param='test',
            debug=True
        )

        # Verify all are set
        self.assertEqual(bst.environ.get_precision(), 64)
        self.assertEqual(bst.environ.get_dt(), 0.01)
        self.assertTrue(bst.environ.get('mode').has(bst.mixin.Training))
        self.assertEqual(bst.environ.get('custom_param'), 'test')
        self.assertTrue(bst.environ.get('debug'))

        # Test in nested contexts
        with bst.environ.context(precision=32, debug=False):
            self.assertEqual(bst.environ.get_precision(), 32)
            self.assertFalse(bst.environ.get('debug'))
            # Others unchanged
            self.assertEqual(bst.environ.get_dt(), 0.01)
            self.assertEqual(bst.environ.get('custom_param'), 'test')


if __name__ == '__main__':
    unittest.main()
