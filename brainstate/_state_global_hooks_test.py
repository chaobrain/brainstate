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

"""Tests for the global hook registry and its module-level helpers."""

import unittest

import brainstate
from brainstate._state_global_hooks import (
    GlobalHookRegistry,
    register_state_hook,
    unregister_state_hook,
    clear_state_hooks,
    has_state_hooks,
    list_state_hooks,
)


class TestGlobalHookRegistry(unittest.TestCase):
    """Validate the singleton registry and module-level convenience functions."""

    def setUp(self):
        """Reset the singleton so each test starts clean."""
        GlobalHookRegistry.reset()

    def tearDown(self):
        """Reset again to avoid leaking global hooks into other tests."""
        GlobalHookRegistry.reset()

    def test_singleton_identity(self):
        """instance() always returns the same object."""
        self.assertIs(GlobalHookRegistry.instance(), GlobalHookRegistry.instance())

    def test_register_and_has_and_list(self):
        """Registering a global hook is observable via has/list."""
        self.assertFalse(has_state_hooks('read'))
        handle = register_state_hook('read', lambda ctx: None, name="g")
        self.assertTrue(has_state_hooks('read'))
        self.assertEqual(len(list_state_hooks('read')), 1)
        self.assertEqual(list_state_hooks('read')[0].name, "g")
        self.assertTrue(unregister_state_hook(handle))
        self.assertFalse(has_state_hooks('read'))

    def test_clear_by_type(self):
        """clear_state_hooks(type) removes only that type."""
        register_state_hook('read', lambda ctx: None)
        register_state_hook('write_after', lambda ctx: None)
        clear_state_hooks('read')
        self.assertFalse(has_state_hooks('read'))
        self.assertTrue(has_state_hooks('write_after'))

    def test_clear_all(self):
        """clear_state_hooks() with no type removes everything."""
        register_state_hook('read', lambda ctx: None)
        register_state_hook('write_after', lambda ctx: None)
        clear_state_hooks()
        self.assertFalse(has_state_hooks())

    def test_global_hook_fires_for_all_states(self):
        """A global read hook fires for every State instance."""
        seen = []
        register_state_hook('read', lambda ctx: seen.append(ctx.operation))
        s1 = brainstate.State(1.0)
        s2 = brainstate.State(2.0)
        _ = s1.value
        _ = s2.value
        self.assertGreaterEqual(len(seen), 2)

    def test_top_level_exports(self):
        """The registry and functions are re-exported from brainstate."""
        self.assertIs(brainstate.GlobalHookRegistry, GlobalHookRegistry)
        self.assertIs(brainstate.register_state_hook, register_state_hook)


if __name__ == "__main__":
    unittest.main()
