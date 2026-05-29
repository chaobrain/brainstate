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


import unittest

import jax
import jax.numpy as jnp

from brainstate.util._tracers import StateJaxTracer, current_jax_trace


class TestCurrentJaxTrace(unittest.TestCase):
    """Validate the module-level ``current_jax_trace`` helper."""

    def test_returns_non_none_at_top_level(self):
        """Return a non-None tracing state when called outside any transform."""
        self.assertIsNotNone(current_jax_trace())

    def test_top_level_calls_are_equal(self):
        """Return equal tracing states for repeated top-level calls."""
        self.assertEqual(current_jax_trace(), current_jax_trace())


class TestStateJaxTracer(unittest.TestCase):
    """Validate trace capture, validity, equality, and pretty-repr."""

    def test_jax_trace_property_matches_current(self):
        """Expose the captured trace through the ``jax_trace`` property."""
        tracer = StateJaxTracer()
        self.assertIsNotNone(tracer.jax_trace)
        self.assertEqual(tracer.jax_trace, current_jax_trace())

    def test_top_level_tracers_are_equal_and_valid(self):
        """Capture identical traces and report validity at the top level."""
        a = StateJaxTracer()
        b = StateJaxTracer()
        self.assertEqual(a, b)
        self.assertTrue(a.is_valid())
        self.assertTrue(b.is_valid())
        self.assertIsNotNone(a.jax_trace)

    def test_eq_rejects_non_tracer(self):
        """Return False from ``__eq__`` for non-tracer objects."""
        self.assertNotEqual(StateJaxTracer(), object())
        self.assertFalse(StateJaxTracer().__eq__(42))

    def test_eq_accepts_matching_tracer(self):
        """Return True from ``__eq__`` for another tracer with the same trace."""
        self.assertTrue(StateJaxTracer().__eq__(StateJaxTracer()))

    def test_repr_contains_class_name(self):
        """Name the class and the captured attribute in the pretty repr."""
        text = repr(StateJaxTracer())
        self.assertIn("StateJaxTracer", text)
        self.assertIn("jax_trace", text)

    def test_pretty_repr_yields_type_and_attr(self):
        """Yield a type header and a ``jax_trace`` attribute from ``__pretty_repr__``."""
        tracer = StateJaxTracer()
        parts = list(tracer.__pretty_repr__())
        self.assertEqual(len(parts), 2)
        self.assertEqual(parts[0].type, "StateJaxTracer")
        self.assertEqual(parts[1].key, "jax_trace")

    def test_tracer_captured_in_jit_is_invalid_outside(self):
        """Invalidate a tracer captured inside a jit trace once tracing finishes."""
        captured = []

        @jax.jit
        def f(x):
            captured.append(StateJaxTracer())
            return x

        f(jnp.ones((3,)))
        self.assertFalse(captured[0].is_valid())

    def test_tracer_inside_jit_differs_from_top_level(self):
        """Capture a distinct trace inside jit compared to the enclosing scope."""
        outer = StateJaxTracer()
        captured = []

        @jax.jit
        def f(x):
            captured.append(StateJaxTracer())
            return x

        f(jnp.ones((3,)))
        self.assertNotEqual(outer, captured[0])

    def test_tracer_inside_jit_is_valid_during_trace(self):
        """Report validity for a tracer while its own jit trace is still active."""
        results = []

        @jax.jit
        def f(x):
            tracer = StateJaxTracer()
            results.append(tracer.is_valid())
            return x

        f(jnp.ones((3,)))
        self.assertTrue(results[0])


if __name__ == "__main__":
    unittest.main()
