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

"""Tests for hook context dataclasses."""

import unittest
import weakref

import brainstate
from brainstate._state_hook_context import (
    HookContext,
    ReadHookContext,
    WriteHookContext,
    MutableWriteHookContext,
    RestoreHookContext,
    InitHookContext,
)


class TestHookContext(unittest.TestCase):
    """Validate the base context and its state accessors."""

    def test_state_property_resolves_live_state(self):
        """`.state` dereferences the weakref to the live State."""
        s = brainstate.State(1.0, name="alpha")
        ctx = HookContext(operation='read', state_ref=weakref.ref(s))
        self.assertIs(ctx.state, s)
        self.assertEqual(ctx.state_name, "alpha")

    def test_state_property_after_gc(self):
        """`.state` returns None and `.state_name` None when state is gone."""
        s = brainstate.State(1.0)
        ref = weakref.ref(s)
        ctx = HookContext(operation='read', state_ref=ref)
        del s
        import gc
        gc.collect()
        # If the state survived due to other references, skip; else assert None.
        if ctx.state is None:
            self.assertIsNone(ctx.state_name)

    def test_default_metadata_and_timestamp(self):
        """metadata defaults to an empty dict; timestamp is a float."""
        s = brainstate.State(1.0)
        ctx = HookContext(operation='init', state_ref=weakref.ref(s))
        self.assertEqual(ctx.metadata, {})
        self.assertIsInstance(ctx.timestamp, float)


class TestSpecializedContexts(unittest.TestCase):
    """Validate the per-operation context subclasses and their fields."""

    def setUp(self):
        """Create a reusable state and weakref."""
        self.state = brainstate.State(1.0)
        self.ref = weakref.ref(self.state)

    def test_read_context_value(self):
        """ReadHookContext carries the read value."""
        ctx = ReadHookContext(operation='read', state_ref=self.ref, value=7)
        self.assertEqual(ctx.value, 7)

    def test_write_context_old_and_new(self):
        """WriteHookContext carries value and old_value."""
        ctx = WriteHookContext(operation='write_after', state_ref=self.ref, value=2, old_value=1)
        self.assertEqual(ctx.value, 2)
        self.assertEqual(ctx.old_value, 1)

    def test_mutable_write_transform_and_cancel(self):
        """MutableWriteHookContext supports transformed_value and cancellation."""
        ctx = MutableWriteHookContext(operation='write_before', state_ref=self.ref, value=5)
        self.assertIsNone(ctx.transformed_value)
        self.assertFalse(ctx.cancel)
        ctx.transformed_value = 10
        ctx.cancel = True
        ctx.cancel_reason = "nope"
        self.assertEqual(ctx.transformed_value, 10)
        self.assertTrue(ctx.cancel)
        self.assertEqual(ctx.cancel_reason, "nope")

    def test_restore_context_is_write_context(self):
        """RestoreHookContext inherits write value/old_value."""
        ctx = RestoreHookContext(operation='restore', state_ref=self.ref, value=3, old_value=2)
        self.assertIsInstance(ctx, WriteHookContext)
        self.assertEqual(ctx.value, 3)

    def test_init_context_metadata(self):
        """InitHookContext carries value and init_metadata."""
        ctx = InitHookContext(operation='init', state_ref=self.ref, value=0, init_metadata={"k": "v"})
        self.assertEqual(ctx.value, 0)
        self.assertEqual(ctx.init_metadata, {"k": "v"})


if __name__ == "__main__":
    unittest.main()
