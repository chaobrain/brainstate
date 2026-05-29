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

"""Tests for the brainstate error hierarchy."""

import unittest

import brainstate
from brainstate._error import BrainStateError, BatchAxisError, TraceContextError


class TestErrorHierarchy(unittest.TestCase):
    """Validate the exception class hierarchy and exports."""

    def test_batch_axis_is_brainstate_error(self):
        """BatchAxisError subclasses BrainStateError."""
        self.assertTrue(issubclass(BatchAxisError, BrainStateError))

    def test_trace_context_is_brainstate_error(self):
        """TraceContextError subclasses BrainStateError."""
        self.assertTrue(issubclass(TraceContextError, BrainStateError))

    def test_base_is_exception(self):
        """BrainStateError subclasses the builtin Exception."""
        self.assertTrue(issubclass(BrainStateError, Exception))

    def test_exported_at_top_level(self):
        """The three errors are re-exported from the brainstate namespace."""
        self.assertIs(brainstate.BrainStateError, BrainStateError)
        self.assertIs(brainstate.BatchAxisError, BatchAxisError)
        self.assertIs(brainstate.TraceContextError, TraceContextError)

    def test_message_round_trips(self):
        """Raising with a message preserves it."""
        with self.assertRaises(BrainStateError) as ctx:
            raise BrainStateError("boom")
        self.assertIn("boom", str(ctx.exception))

    def test_batch_axis_error_module_label(self):
        """BatchAxisError advertises the transform module for tracebacks."""
        self.assertEqual(BatchAxisError.__module__, "brainstate.transform")

    def test_catch_via_base(self):
        """Subclass instances are catchable as the base type."""
        with self.assertRaises(BrainStateError):
            raise TraceContextError("trace")


if __name__ == "__main__":
    unittest.main()
