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

"""Unit tests for HookManager."""

import threading
import weakref
from unittest import TestCase

import brainstate
from brainstate import (
    HookManager,
    HookConfig,
    HookExecutionError,
)


class MockState:
    """Mock State class that supports weak references for testing."""

    def __init__(self, name="test_state"):
        self.name = name


class TestHookManager(TestCase):
    """Test suite for HookManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self.manager = HookManager()
        self.call_log = []

    def test_registration_and_unregistration(self):
        """Test basic hook registration and unregistration."""

        def hook_fn(ctx):
            self.call_log.append('hook_called')

        # Register hook
        handle = self.manager.register_hook('read', hook_fn)
        self.assertTrue(self.manager.has_hooks('read'))
        self.assertEqual(len(self.manager.get_hooks('read')), 1)

        # Unregister hook
        success = self.manager.unregister_hook(handle)
        self.assertTrue(success)
        self.assertFalse(self.manager.has_hooks('read'))
        self.assertEqual(len(self.manager.get_hooks('read')), 0)

    def test_priority_ordering(self):
        """Test that hooks execute in priority order (descending)."""

        def make_hook(name):
            def hook_fn(ctx):
                self.call_log.append(name)

            return hook_fn

        # Register hooks with different priorities
        self.manager.register_hook('read', make_hook('low'), priority=1)
        self.manager.register_hook('read', make_hook('high'), priority=100)
        self.manager.register_hook('read', make_hook('medium'), priority=50)

        # Execute hooks
        mock_state_ref = weakref.ref(MockState())
        self.manager.execute_read_hooks(value=42, state_ref=mock_state_ref)

        # Should execute in order: high, medium, low
        self.assertEqual(self.call_log, ['high', 'medium', 'low'])

    def test_hook_execution_order_within_same_priority(self):
        """Test stable ordering for hooks with same priority."""

        def make_hook(name):
            def hook_fn(ctx):
                self.call_log.append(name)

            return hook_fn

        # Register multiple hooks with same priority
        self.manager.register_hook('read', make_hook('first'), priority=10)
        self.manager.register_hook('read', make_hook('second'), priority=10)
        self.manager.register_hook('read', make_hook('third'), priority=10)

        # Execute hooks
        mock_state_ref = weakref.ref(MockState())
        self.manager.execute_read_hooks(value=42, state_ref=mock_state_ref)

        # Should maintain registration order
        self.assertEqual(self.call_log, ['first', 'second', 'third'])

    def test_caching_behavior(self):
        """Test that hook cache is properly invalidated and rebuilt."""

        def hook_fn(ctx):
            self.call_log.append('hook')

        # Register hook
        handle = self.manager.register_hook('read', hook_fn)
        mock_state_ref = weakref.ref(MockState())

        # Execute - should build cache
        self.manager.execute_read_hooks(value=42, state_ref=mock_state_ref)
        self.assertEqual(len(self.call_log), 1)
        self.call_log.clear()

        # Disable hook - should invalidate cache
        handle.disable()
        self.manager.execute_read_hooks(value=42, state_ref=mock_state_ref)
        self.assertEqual(len(self.call_log), 0)  # Hook not executed

        # Re-enable hook - should invalidate cache
        handle.enable()
        self.manager.execute_read_hooks(value=42, state_ref=mock_state_ref)
        self.assertEqual(len(self.call_log), 1)  # Hook executed

    def test_error_handling_raise_mode(self):
        """Test error handling in 'raise' mode."""
        config = HookConfig(on_error='raise')
        manager = HookManager(config)

        def failing_hook(ctx):
            raise ValueError("Test error")

        manager.register_hook('read', failing_hook)
        mock_state_ref = weakref.ref(MockState())

        with self.assertRaises(HookExecutionError):
            manager.execute_read_hooks(value=42, state_ref=mock_state_ref)

    def test_error_handling_log_mode(self):
        """Test error handling in 'log' mode."""
        config = HookConfig(on_error='log')
        manager = HookManager(config)

        def failing_hook(ctx):
            raise ValueError("Test error")

        def successful_hook(ctx):
            self.call_log.append('success')

        manager.register_hook('read', failing_hook, priority=10)
        manager.register_hook('read', successful_hook, priority=5)
        mock_state_ref = weakref.ref(MockState())

        # Should not raise, but continue executing remaining hooks
        manager.execute_read_hooks(value=42, state_ref=mock_state_ref)
        self.assertEqual(self.call_log, ['success'])

    def test_error_handling_ignore_mode(self):
        """Test error handling in 'ignore' mode."""
        config = HookConfig(on_error='ignore')
        manager = HookManager(config)

        def failing_hook(ctx):
            raise ValueError("Test error")

        manager.register_hook('read', failing_hook)
        mock_state_ref = weakref.ref(MockState())

        # Should silently ignore error
        manager.execute_read_hooks(value=42, state_ref=mock_state_ref)

    def test_disable_on_error(self):
        """Test auto-disabling hooks after max errors."""
        config = HookConfig(on_error='log', disable_on_error=True, max_errors_per_hook=3)
        manager = HookManager(config)

        error_count = [0]

        def failing_hook(ctx):
            error_count[0] += 1
            raise ValueError(f"Error {error_count[0]}")

        handle = manager.register_hook('read', failing_hook)
        mock_state_ref = weakref.ref(MockState())

        # Execute hook multiple times
        for _ in range(5):
            manager.execute_read_hooks(value=42, state_ref=mock_state_ref)

        # Hook should be disabled after 3 errors
        self.assertFalse(handle.is_enabled())
        self.assertEqual(error_count[0], 3)  # Only 3 errors before disable

    def test_sequential_chaining_write_before(self):
        """Test sequential chaining in write_before hooks."""

        def multiply_by_2(ctx):
            input_val = ctx.transformed_value if ctx.transformed_value is not None else ctx.value
            ctx.transformed_value = input_val * 2

        def add_10(ctx):
            input_val = ctx.transformed_value if ctx.transformed_value is not None else ctx.value
            ctx.transformed_value = input_val + 10

        self.manager.register_hook('write_before', multiply_by_2, priority=10)
        self.manager.register_hook('write_before', add_10, priority=5)
        mock_state_ref = weakref.ref(MockState())

        # Execute hooks: (5 * 2) + 10 = 20
        result = self.manager.execute_write_before_hooks(
            new_value=5, old_value=0, state_ref=mock_state_ref
        )
        self.assertEqual(result, 20)

    def test_hook_cancellation(self):
        """Test cancellation in write_before hooks."""

        def validate_positive(ctx):
            if ctx.value < 0:
                ctx.cancel = True
                ctx.cancel_reason = "Value must be positive"

        self.manager.register_hook('write_before', validate_positive)
        mock_state_ref = weakref.ref(MockState())

        # Should not raise for positive value
        result = self.manager.execute_write_before_hooks(
            new_value=5, old_value=0, state_ref=mock_state_ref
        )
        self.assertEqual(result, 5)

        # Should raise for negative value
        with self.assertRaises(brainstate.HookCancellationError):
            self.manager.execute_write_before_hooks(
                new_value=-5, old_value=0, state_ref=mock_state_ref
            )

    def test_thread_safety_concurrent_registration(self):
        """Test thread-safe hook registration."""
        num_threads = 10
        hooks_per_thread = 5

        def register_hooks(thread_id):
            for i in range(hooks_per_thread):
                self.manager.register_hook(
                    'read',
                    lambda ctx: None,
                    priority=thread_id * 100 + i,
                    name=f"thread_{thread_id}_hook_{i}"
                )

        threads = [threading.Thread(target=register_hooks, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All hooks should be registered
        self.assertEqual(len(self.manager.get_hooks('read')), num_threads * hooks_per_thread)

    def test_thread_safety_concurrent_execution(self):
        """Test thread-safe hook execution."""
        execution_count = {'count': 0}
        lock = threading.Lock()

        def counting_hook(ctx):
            with lock:
                execution_count['count'] += 1

        self.manager.register_hook('read', counting_hook)
        mock_state_ref = weakref.ref(MockState())

        num_threads = 10
        executions_per_thread = 20

        def execute_hooks():
            for _ in range(executions_per_thread):
                self.manager.execute_read_hooks(value=42, state_ref=mock_state_ref)

        threads = [threading.Thread(target=execute_hooks) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All executions should be counted
        self.assertEqual(execution_count['count'], num_threads * executions_per_thread)

    def test_clear_hooks_by_type(self):
        """Test clearing hooks filtered by type."""
        self.manager.register_hook('read', lambda ctx: None)
        self.manager.register_hook('write_before', lambda ctx: None)
        self.manager.register_hook('write_after', lambda ctx: None)

        # Clear only read hooks
        self.manager.clear_hooks('read')
        self.assertFalse(self.manager.has_hooks('read'))
        self.assertTrue(self.manager.has_hooks('write_before'))
        self.assertTrue(self.manager.has_hooks('write_after'))

    def test_clear_all_hooks(self):
        """Test clearing all hooks."""
        self.manager.register_hook('read', lambda ctx: None)
        self.manager.register_hook('write_before', lambda ctx: None)
        self.manager.register_hook('write_after', lambda ctx: None)
        self.manager.register_hook('restore', lambda ctx: None)

        # Clear all hooks
        self.manager.clear_hooks()
        self.assertFalse(self.manager.has_hooks())

    def test_handle_operations(self):
        """Test HookHandle enable/disable/remove operations."""

        def hook_fn(ctx):
            self.call_log.append('hook')

        handle = self.manager.register_hook('read', hook_fn)
        mock_state_ref = weakref.ref(MockState())

        # Initially enabled
        self.assertTrue(handle.is_enabled())
        self.manager.execute_read_hooks(value=42, state_ref=mock_state_ref)
        self.assertEqual(len(self.call_log), 1)
        self.call_log.clear()

        # Disable
        handle.disable()
        self.assertFalse(handle.is_enabled())
        self.manager.execute_read_hooks(value=42, state_ref=mock_state_ref)
        self.assertEqual(len(self.call_log), 0)

        # Re-enable
        handle.enable()
        self.assertTrue(handle.is_enabled())
        self.manager.execute_read_hooks(value=42, state_ref=mock_state_ref)
        self.assertEqual(len(self.call_log), 1)
        self.call_log.clear()

        # Remove
        success = handle.remove()
        self.assertTrue(success)
        self.assertTrue(handle.is_removed())
        self.manager.execute_read_hooks(value=42, state_ref=mock_state_ref)
        self.assertEqual(len(self.call_log), 0)


if __name__ == '__main__':
    import unittest

    unittest.main()
