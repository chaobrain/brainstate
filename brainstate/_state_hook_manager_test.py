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


class TestHookManagerCoverage(TestCase):
    """Cover HookManager validation, transformation, dispatch, and error handling."""

    def setUp(self):
        """Provide a manager and a live state ref (kept alive via self._state)."""
        self.manager = HookManager()
        self._state = MockState()
        self.ref = weakref.ref(self._state)

    def test_register_invalid_type_raises(self):
        """Registering an unknown hook type raises HookRegistrationError."""
        from brainstate._state_hook_core import HookRegistrationError
        with self.assertRaises(HookRegistrationError):
            self.manager.register_hook('bogus', lambda ctx: None)

    def test_get_hook_list_invalid_type_raises(self):
        """The internal _get_hook_list rejects unknown types with ValueError."""
        with self.assertRaises(ValueError):
            self.manager._get_hook_list('bogus')

    def test_unregister_unknown_handle_returns_false(self):
        """Unregistering a hook twice returns False the second time."""
        handle = self.manager.register_hook('read', lambda ctx: None)
        self.assertTrue(self.manager.unregister_hook(handle))
        self.assertFalse(self.manager.unregister_hook(handle))

    def test_has_hooks_all_types_true(self):
        """has_hooks() with no type returns True when any hook is registered."""
        self.assertFalse(self.manager.has_hooks())
        self.manager.register_hook('write_after', lambda ctx: None)
        self.assertTrue(self.manager.has_hooks())

    def test_execute_paths_with_no_hooks_are_noops(self):
        """Every execute_* path returns cleanly when no hooks are registered."""
        self.assertEqual(self.manager.execute_write_before_hooks(7, 0, self.ref), 7)
        self.assertIsNone(self.manager.execute_write_after_hooks(7, 0, self.ref))
        self.assertIsNone(self.manager.execute_restore_hooks(7, 0, self.ref))
        self.assertIsNone(self.manager.execute_init_hooks(7, self.ref, {}))

    def test_write_before_transforms_value(self):
        """A write_before hook can transform the written value (sequential chaining)."""
        def double(ctx):
            ctx.transformed_value = ctx.value * 2

        self.manager.register_hook('write_before', double)
        self.assertEqual(self.manager.execute_write_before_hooks(5, 0, self.ref), 10)

    def test_write_before_without_transform_keeps_value(self):
        """A write_before hook that sets nothing leaves the value unchanged."""
        self.manager.register_hook('write_before', lambda ctx: None)
        self.assertEqual(self.manager.execute_write_before_hooks(5, 0, self.ref), 5)

    def test_write_before_cancel_raises(self):
        """A write_before hook can cancel the write, raising HookCancellationError."""
        from brainstate._state_hook_core import HookCancellationError

        def veto(ctx):
            ctx.cancel = True
            ctx.cancel_reason = "nope"

        self.manager.register_hook('write_before', veto)
        with self.assertRaises(HookCancellationError):
            self.manager.execute_write_before_hooks(5, 0, self.ref)

    def test_init_hooks_execute(self):
        """Registered init hooks run with an InitHookContext carrying the value."""
        seen = []
        self.manager.register_hook('init', lambda ctx: seen.append(ctx.value))
        self.manager.execute_init_hooks(99, self.ref, {'k': 1})
        self.assertEqual(seen, [99])

    def test_error_raise_mode_propagates(self):
        """on_error='raise' surfaces hook failures as HookExecutionError."""
        mgr = HookManager(HookConfig(on_error='raise'))

        def boom(ctx):
            raise ValueError("inner")

        mgr.register_hook('read', boom)
        with self.assertRaises(HookExecutionError):
            mgr.execute_read_hooks(1, self.ref)

    def test_error_custom_logger_invoked(self):
        """on_error='log' with a custom error_logger routes the error to it."""
        logged = []
        cfg = HookConfig(on_error='log', error_logger=lambda *a: logged.append(a))
        mgr = HookManager(cfg)
        mgr.register_hook('write_after', lambda ctx: (_ for _ in ()).throw(RuntimeError("x")))
        mgr.execute_write_after_hooks(1, 0, self.ref)
        self.assertEqual(len(logged), 1)

    def test_error_default_log_warns(self):
        """on_error='log' without a logger emits a HookWarning."""
        from brainstate._state_hook_core import HookWarning
        mgr = HookManager(HookConfig(on_error='log'))
        mgr.register_hook('restore', lambda ctx: (_ for _ in ()).throw(RuntimeError("x")))
        with self.assertWarns(HookWarning):
            mgr.execute_restore_hooks(1, 0, self.ref)

    def test_disable_on_error_auto_disables(self):
        """disable_on_error disables a hook once the error threshold is hit."""
        cfg = HookConfig(on_error='log', disable_on_error=True, max_errors_per_hook=1)
        mgr = HookManager(cfg)
        mgr.register_hook('read', lambda ctx: (_ for _ in ()).throw(RuntimeError("x")))
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mgr.execute_read_hooks(1, self.ref)
        self.assertFalse(mgr.has_hooks('read'))

    # --- A13: has_hooks(invalid) is not state-dependent ---
    def test_a13_has_hooks_invalid_type_raises_when_empty(self):
        """An invalid hook type raises ValueError even with no hooks registered."""
        mgr = HookManager()
        with self.assertRaises(ValueError):
            mgr.has_hooks('bogus')

    def test_a13_has_hooks_invalid_type_raises_when_nonempty(self):
        """An invalid hook type raises ValueError when hooks are registered too."""
        mgr = HookManager()
        mgr.register_hook('read', lambda ctx: None)
        with self.assertRaises(ValueError):
            mgr.has_hooks('bogus')

    def test_a13_has_hooks_valid_types_ok(self):
        """Valid hook types and None never raise."""
        mgr = HookManager()
        self.assertFalse(mgr.has_hooks())
        for t in ('read', 'write_before', 'write_after', 'restore', 'init'):
            self.assertFalse(mgr.has_hooks(t))


if __name__ == '__main__':
    import unittest

    unittest.main()
