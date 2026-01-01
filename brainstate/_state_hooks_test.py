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

"""Integration tests for State hooks."""

from unittest import TestCase

import jax.numpy as jnp

import brainstate
from brainstate import HookCancellationError


class TestStateHooks(TestCase):
    """Integration tests for State hook functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.call_log = []
        # Reset global hooks before each test
        brainstate.clear_state_hooks()

    def tearDown(self):
        """Clean up after tests."""
        brainstate.clear_state_hooks()

    def test_read_hook_execution(self):
        """Test that read hooks execute on value access."""
        state = brainstate.State(jnp.array([1, 2, 3]))

        def read_hook(ctx):
            self.call_log.append(f"read: {ctx.value.tolist()}")

        state.register_hook('read', read_hook)

        # Access value
        _ = state.value
        self.assertEqual(len(self.call_log), 1)
        self.assertEqual(self.call_log[0], "read: [1, 2, 3]")

        # Access again
        _ = state.value
        self.assertEqual(len(self.call_log), 2)

    def test_write_before_hook_transformation(self):
        """Test write_before hooks can transform values."""
        state = brainstate.State(jnp.array([0, 0, 0]))

        def clip_values(ctx):
            # Clip to [-1, 1]
            input_val = ctx.transformed_value if ctx.transformed_value is not None else ctx.value
            ctx.transformed_value = jnp.clip(input_val, -1.0, 1.0)

        state.register_hook('write_before', clip_values)

        # Set value that exceeds clip range
        state.value = jnp.array([5, -5, 0.5])

        # Value should be clipped
        self.assertTrue(jnp.allclose(state.value, jnp.array([1, -1, 0.5])))

    def test_write_before_hook_cancellation(self):
        """Test write_before hooks can cancel operations."""
        state = brainstate.State(jnp.array([1, 2, 3]))

        def validate_positive(ctx):
            if jnp.any(ctx.value < 0):
                ctx.cancel = True
                ctx.cancel_reason = "All values must be non-negative"

        state.register_hook('write_before', validate_positive, priority=100)

        # Valid write should succeed
        state.value = jnp.array([1, 2, 3])
        self.assertTrue(jnp.array_equal(state.value, jnp.array([1, 2, 3])))

        # Invalid write should be cancelled
        with self.assertRaises(HookCancellationError):
            state.value = jnp.array([1, -1, 3])

    def test_write_after_hook_notification(self):
        """Test write_after hooks receive notifications."""
        state = brainstate.State(jnp.array([1, 2, 3]))

        def log_change(ctx):
            self.call_log.append({
                'old': ctx.old_value.tolist(),
                'new': ctx.value.tolist()
            })

        state.register_hook('write_after', log_change)

        state.value = jnp.array([4, 5, 6])
        self.assertEqual(len(self.call_log), 1)
        self.assertEqual(self.call_log[0]['old'], [1, 2, 3])
        self.assertEqual(self.call_log[0]['new'], [4, 5, 6])

    def test_restore_hook_with_old_and_new_values(self):
        """Test restore hooks receive both old and new values."""
        state = brainstate.State(jnp.array([1, 2, 3]))
        state.value = jnp.array([4, 5, 6])  # Change the value

        def log_restore(ctx):
            self.call_log.append({
                'old': ctx.old_value.tolist(),
                'new': ctx.value.tolist()
            })

        state.register_hook('restore', log_restore)

        state.restore_value(jnp.array([7, 8, 9]))
        self.assertEqual(len(self.call_log), 1)
        self.assertEqual(self.call_log[0]['old'], [4, 5, 6])
        self.assertEqual(self.call_log[0]['new'], [7, 8, 9])

    def test_sequential_chaining_behavior(self):
        """Test sequential chaining of write_before hooks."""
        state = brainstate.State(jnp.array([2.0, 3.0]))

        def multiply_by_2(ctx):
            input_val = ctx.transformed_value if ctx.transformed_value is not None else ctx.value
            ctx.transformed_value = input_val * 2

        def add_10(ctx):
            input_val = ctx.transformed_value if ctx.transformed_value is not None else ctx.value
            ctx.transformed_value = input_val + 10

        state.register_hook('write_before', multiply_by_2, priority=10)
        state.register_hook('write_before', add_10, priority=5)

        state.value = jnp.array([1.0, 2.0])
        # Should be: (1 * 2) + 10 = 12, (2 * 2) + 10 = 14
        self.assertTrue(jnp.allclose(state.value, jnp.array([12.0, 14.0])))

    def test_global_and_instance_hook_interaction(self):
        """Test global hooks execute before instance hooks."""
        global_log = []
        instance_log = []

        def global_hook(ctx):
            global_log.append('global')

        def instance_hook(ctx):
            instance_log.append('instance')

        # Register global hook
        brainstate.register_state_hook('read', global_hook)

        # Create state with instance hook
        state = brainstate.State(jnp.array([1, 2, 3]))
        state.register_hook('read', instance_hook)

        # Access value
        _ = state.value

        # Both should be called, global first
        self.assertEqual(global_log, ['global'])
        self.assertEqual(instance_log, ['instance'])

    def test_context_manager_api(self):
        """Test temporary_hook context manager."""
        state = brainstate.State(jnp.array([1, 2, 3]))

        def temp_hook(ctx):
            self.call_log.append('temp')

        # Hook active only within context
        with state.temporary_hook('read', temp_hook):
            _ = state.value
            self.assertEqual(len(self.call_log), 1)

        # Hook removed after context
        _ = state.value
        self.assertEqual(len(self.call_log), 1)  # No additional call

    def test_hooks_always_enabled(self):
        """Test that hooks are always enabled by default."""
        # Hooks are always initialized
        state = brainstate.State(jnp.array([1, 2, 3]))
        self.assertIsNotNone(state._hooks_manager)

        # Hook manager should be accessible
        self.assertIsNotNone(state.hooks)

    def test_hook_manager_always_initialized(self):
        """Test hook manager is always initialized."""
        state = brainstate.State(jnp.array([1, 2, 3]))
        # Hook manager should exist from the start
        self.assertIsNotNone(state._hooks_manager)

        # Should be able to register hooks immediately
        handle = state.register_hook('read', lambda ctx: None)
        self.assertIsNotNone(handle)

    def test_hook_context_metadata(self):
        """Test that hook contexts contain proper metadata."""
        state = brainstate.State(jnp.array([1, 2, 3]), name="test_state")

        context_data = {}

        def capture_context(ctx):
            context_data['operation'] = ctx.operation
            context_data['state_name'] = ctx.state_name
            context_data['value'] = ctx.value
            context_data['timestamp'] = ctx.timestamp

        state.register_hook('read', capture_context)
        _ = state.value

        self.assertEqual(context_data['operation'], 'read')
        self.assertEqual(context_data['state_name'], 'test_state')
        self.assertTrue(jnp.array_equal(context_data['value'], jnp.array([1, 2, 3])))
        self.assertIsInstance(context_data['timestamp'], float)
        self.assertGreater(context_data['timestamp'], 0)

    def test_multiple_hooks_same_type(self):
        """Test registering multiple hooks of the same type."""
        state = brainstate.State(jnp.array([1, 2, 3]))

        state.register_hook('read', lambda ctx: self.call_log.append('hook1'))
        state.register_hook('read', lambda ctx: self.call_log.append('hook2'))
        state.register_hook('read', lambda ctx: self.call_log.append('hook3'))

        _ = state.value
        self.assertEqual(self.call_log, ['hook1', 'hook2', 'hook3'])

    def test_hook_manager_property(self):
        """Test the hooks property accessor."""
        state = brainstate.State(jnp.array([1, 2, 3]))

        # Access hooks property - should create manager
        manager = state.hooks
        self.assertIsNotNone(manager)
        self.assertIsNotNone(state._hooks_manager)

        # Subsequent access should return same manager
        manager2 = state.hooks
        self.assertIs(manager, manager2)

    def test_list_and_clear_hooks(self):
        """Test listing and clearing hooks."""
        state = brainstate.State(jnp.array([1, 2, 3]))

        state.register_hook('read', lambda ctx: None, name='hook1')
        state.register_hook('write_before', lambda ctx: None, name='hook2')
        state.register_hook('write_after', lambda ctx: None, name='hook3')

        # List all hooks
        all_hooks = state.list_hooks()
        self.assertEqual(len(all_hooks), 3)

        # List hooks by type
        read_hooks = state.list_hooks('read')
        self.assertEqual(len(read_hooks), 1)

        # Clear specific type
        state.clear_hooks('read')
        self.assertEqual(len(state.list_hooks('read')), 0)
        self.assertEqual(len(state.list_hooks('write_before')), 1)

        # Clear all
        state.clear_hooks()
        self.assertEqual(len(state.list_hooks()), 0)

    def test_write_transformation_with_state_operations(self):
        """Test that transformations work with normal state operations."""
        state = brainstate.State(jnp.array([1.0, 2.0, 3.0]))

        # Add normalization hook
        def normalize(ctx):
            input_val = ctx.transformed_value if ctx.transformed_value is not None else ctx.value
            norm = jnp.linalg.norm(input_val)
            if norm > 0:
                ctx.transformed_value = input_val / norm

        state.register_hook('write_before', normalize)

        # Set a value
        state.value = jnp.array([3.0, 4.0, 0.0])

        # Should be normalized
        expected = jnp.array([3.0, 4.0, 0.0]) / 5.0
        self.assertTrue(jnp.allclose(state.value, expected))

    def test_global_hooks_multiple_states(self):
        """Test global hooks apply to all states."""
        global_call_count = {'count': 0}

        def global_counter(ctx):
            global_call_count['count'] += 1

        brainstate.register_state_hook('read', global_counter)

        # Create multiple states
        state1 = brainstate.State(jnp.array([1, 2]))
        state2 = brainstate.State(jnp.array([3, 4]))

        _ = state1.value
        _ = state2.value

        # Global hook should be called for both
        self.assertEqual(global_call_count['count'], 2)


if __name__ == '__main__':
    import unittest

    unittest.main()
