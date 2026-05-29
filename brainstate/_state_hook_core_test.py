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

"""Tests for the core Hook and HookHandle classes and hook exceptions."""

import unittest
import weakref

import brainstate
from brainstate._state_hook_context import ReadHookContext
from brainstate._state_hook_core import (
    Hook,
    HookHandle,
    HookError,
    HookExecutionError,
    HookRegistrationError,
    HookCancellationError,
    HookWarning,
)


def _make_read_context(state):
    """Build a minimal ReadHookContext for direct Hook.execute calls."""
    return ReadHookContext(operation='read', state_ref=weakref.ref(state), value=state.value)


class TestHookExceptions(unittest.TestCase):
    """Validate the hook exception hierarchy."""

    def test_subclassing(self):
        """All hook errors derive from HookError; warning from UserWarning."""
        for exc in (HookExecutionError, HookRegistrationError, HookCancellationError):
            self.assertTrue(issubclass(exc, HookError))
        self.assertTrue(issubclass(HookError, Exception))
        self.assertTrue(issubclass(HookWarning, UserWarning))

    def test_top_level_exports(self):
        """Hook exceptions are re-exported from the brainstate namespace."""
        self.assertIs(brainstate.HookError, HookError)
        self.assertIs(brainstate.HookExecutionError, HookExecutionError)


class TestHook(unittest.TestCase):
    """Validate Hook construction, execution, priority ordering."""

    def test_non_callable_raises_registration_error(self):
        """A non-callable callback raises HookRegistrationError."""
        with self.assertRaises(HookRegistrationError):
            Hook(callback=123)

    def test_default_name_and_fields(self):
        """Defaults: auto name, priority 0, enabled True, unique id."""
        h = Hook(callback=lambda ctx: None)
        self.assertTrue(h.name.startswith("hook_"))
        self.assertEqual(h.priority, 0)
        self.assertTrue(h.enabled)

    def test_execute_returns_callback_value(self):
        """execute() returns the callback's return value."""
        state = brainstate.State(1.0)
        h = Hook(callback=lambda ctx: "ran")
        self.assertEqual(h.execute(_make_read_context(state)), "ran")

    def test_disabled_hook_returns_none(self):
        """A disabled hook does not invoke the callback."""
        state = brainstate.State(1.0)
        h = Hook(callback=lambda ctx: "ran", enabled=False)
        self.assertIsNone(h.execute(_make_read_context(state)))

    def test_execute_wraps_callback_error(self):
        """A raising callback is wrapped in HookExecutionError."""
        state = brainstate.State(1.0)

        def boom(ctx):
            raise ValueError("inner")

        h = Hook(callback=boom)
        with self.assertRaises(HookExecutionError):
            h.execute(_make_read_context(state))

    def test_priority_ordering(self):
        """Higher priority sorts earlier (Hook.__lt__ is descending)."""
        low = Hook(callback=lambda ctx: None, priority=1)
        high = Hook(callback=lambda ctx: None, priority=10)
        self.assertTrue(high < low)
        self.assertEqual(sorted([low, high])[0], high)

    def test_repr_reflects_status(self):
        """repr shows enabled/disabled status."""
        h = Hook(callback=lambda ctx: None, name="x")
        self.assertIn("enabled", repr(h))
        h.enabled = False
        self.assertIn("disabled", repr(h))


class TestHookHandle(unittest.TestCase):
    """Validate HookHandle enable/disable/remove lifecycle via a real State."""

    def test_enable_disable_remove(self):
        """A registered hook can be disabled, re-enabled, and removed."""
        state = brainstate.State(0.0)
        handle = state.register_hook('read', lambda ctx: None, name="h")
        self.assertIsInstance(handle, HookHandle)
        self.assertTrue(handle.is_enabled())

        handle.disable()
        self.assertFalse(handle.is_enabled())
        handle.enable()
        self.assertTrue(handle.is_enabled())

        self.assertTrue(handle.remove())
        self.assertTrue(handle.is_removed())
        self.assertFalse(handle.is_enabled())

    def test_double_remove_returns_false(self):
        """Removing an already-removed hook returns False."""
        state = brainstate.State(0.0)
        handle = state.register_hook('read', lambda ctx: None)
        self.assertTrue(handle.remove())
        self.assertFalse(handle.remove())

    def test_enable_after_remove_raises(self):
        """Enabling a removed hook raises HookError."""
        state = brainstate.State(0.0)
        handle = state.register_hook('read', lambda ctx: None)
        handle.remove()
        with self.assertRaises(HookError):
            handle.enable()

    def test_name_and_priority_properties(self):
        """Handle exposes the hook's name and priority."""
        state = brainstate.State(0.0)
        handle = state.register_hook('read', lambda ctx: None, name="named", priority=5)
        self.assertEqual(handle.name, "named")
        self.assertEqual(handle.priority, 5)

    def test_disable_after_remove_raises(self):
        """Disabling a removed hook raises HookError."""
        state = brainstate.State(0.0)
        handle = state.register_hook('read', lambda ctx: None)
        handle.remove()
        with self.assertRaises(HookError):
            handle.disable()

    def test_repr_reflects_lifecycle(self):
        """HookHandle repr shows enabled, disabled, then removed status."""
        state = brainstate.State(0.0)
        handle = state.register_hook('read', lambda ctx: None, name="r")
        self.assertIn("enabled", repr(handle))
        handle.disable()
        self.assertIn("disabled", repr(handle))
        handle.remove()
        self.assertIn("removed", repr(handle))


class TestHookHandleManagerGarbageCollected(unittest.TestCase):
    """Validate HookHandle behaviour once its owning HookManager is gone."""

    def _orphan_handle(self):
        """Return a handle whose manager weakref has been cleared via GC."""
        import gc

        state = brainstate.State(0.0)
        handle = state.register_hook('read', lambda ctx: None)
        del state
        gc.collect()
        return handle

    def test_enable_raises_when_manager_collected(self):
        """enable() raises HookError once the manager is garbage collected."""
        handle = self._orphan_handle()
        if handle._manager_ref() is not None:
            self.skipTest("HookManager survived GC in this environment")
        with self.assertRaises(HookError):
            handle.enable()

    def test_disable_raises_when_manager_collected(self):
        """disable() raises HookError once the manager is garbage collected."""
        handle = self._orphan_handle()
        if handle._manager_ref() is not None:
            self.skipTest("HookManager survived GC in this environment")
        with self.assertRaises(HookError):
            handle.disable()

    def test_remove_returns_false_when_manager_collected(self):
        """remove() returns False once the manager is garbage collected."""
        handle = self._orphan_handle()
        if handle._manager_ref() is not None:
            self.skipTest("HookManager survived GC in this environment")
        self.assertFalse(handle.remove())


if __name__ == "__main__":
    unittest.main()
