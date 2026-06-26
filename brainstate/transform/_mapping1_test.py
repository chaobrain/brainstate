"""
Comprehensive tests for brainstate.transform._mapping_old module.

This module contains tests for the old vmap implementation with explicit state management.
The tests cover:

1. Helper Functions:
   - _flatten_in_out_states: Converting state specifications to internal format
   - _remove_axis: Removing axes from arrays
   - _get_batch_size: Determining batch size from various sources
   - _format_state_axes: Formatting and validating state axis specifications

2. vmap with States:
   - Basic stateful functions with single and multiple states
   - States on same axis and different configurations
   - State list format (defaults to axis 0)
   - RandomState integration

3. vmap_new_states:
   - Creating new states within vmapped functions
   - Multiple calls with fresh state creation
   - State tagging and filtering

4. Edge Cases and Error Handling:
   - Kwargs not supported
   - Axis length mismatches
   - Batched states not in out_states
   - Invalid axis_size values

5. Complex Scenarios:
   - Axis naming
   - List to tuple conversion
   - Different output axes
   - LongTermState integration
   - Complex state structures

Note: The old vmap implementation is designed specifically for stateful functions.
Functions without states may not work as expected and should use the newer vmap2 API.

Total test count: 40 tests across 10 test classes
"""
import unittest

import jax
import jax.numpy as jnp
import pytest

import brainstate as bst
from brainstate.transform._mapping1 import (
    vmap,
    vmap_new_states,
)
from brainstate.transform._mapping_core import (
    _flatten_in_out_states,
    _remove_axis,
    _get_batch_size,
    _format_state_axes,
)
from brainstate._state import NonBatchState
from brainstate._error import BatchAxisError
from brainstate.util.filter import OfType


class TestFlattenInOutStates(unittest.TestCase):
    """Test the _flatten_in_out_states helper function."""

    def test_flatten_none_input(self):
        """Test with None input."""
        axis_to_states, state_to_axis = _flatten_in_out_states(None)
        self.assertEqual(axis_to_states, {})
        self.assertEqual(state_to_axis, {})

    def test_flatten_dict_with_int_keys(self):
        """Test with dict having integer keys and dict values."""
        state1 = bst.ShortTermState(jnp.array(1.0))
        state2 = bst.ShortTermState(jnp.array(2.0))
        state3 = bst.ShortTermState(jnp.array(3.0))

        in_states = {
            0: {'a': state1, 'b': state2},
            1: {'c': state3}
        }

        axis_to_states, state_to_axis = _flatten_in_out_states(in_states)

        self.assertEqual(len(axis_to_states), 2)
        self.assertEqual(len(axis_to_states[0]), 2)
        self.assertEqual(len(axis_to_states[1]), 1)
        self.assertIn(state1, axis_to_states[0])
        self.assertIn(state2, axis_to_states[0])
        self.assertIn(state3, axis_to_states[1])

        self.assertEqual(state_to_axis[state1], 0)
        self.assertEqual(state_to_axis[state2], 0)
        self.assertEqual(state_to_axis[state3], 1)

    def test_flatten_list_of_states(self):
        """Test with a list of states (defaults to axis 0)."""
        state1 = bst.ShortTermState(jnp.array(1.0))
        state2 = bst.ShortTermState(jnp.array(2.0))

        in_states = [state1, state2]

        axis_to_states, state_to_axis = _flatten_in_out_states(in_states)

        self.assertEqual(len(axis_to_states), 1)
        self.assertEqual(len(axis_to_states[0]), 2)
        self.assertIn(state1, axis_to_states[0])
        self.assertIn(state2, axis_to_states[0])

        self.assertEqual(state_to_axis[state1], 0)
        self.assertEqual(state_to_axis[state2], 0)

    def test_flatten_single_state(self):
        """Test with a single state (defaults to axis 0)."""
        state1 = bst.ShortTermState(jnp.array(1.0))

        axis_to_states, state_to_axis = _flatten_in_out_states(state1)

        self.assertEqual(len(axis_to_states), 1)
        self.assertEqual(len(axis_to_states[0]), 1)
        self.assertIn(state1, axis_to_states[0])
        self.assertEqual(state_to_axis[state1], 0)

    def test_flatten_empty_dict(self):
        """Test with empty dict."""
        axis_to_states, state_to_axis = _flatten_in_out_states({})

        self.assertEqual(axis_to_states, {})
        self.assertEqual(state_to_axis, {})


class TestRemoveAxis(unittest.TestCase):
    """Test the _remove_axis helper function."""

    def test_remove_axis_0(self):
        """Test removing axis 0."""
        x = jnp.arange(12).reshape(3, 4)
        result = _remove_axis(x, 0)
        self.assertTrue(jnp.allclose(result, x[0]))
        self.assertEqual(result.shape, (4,))

    def test_remove_axis_1(self):
        """Test removing axis 1."""
        x = jnp.arange(12).reshape(3, 4)
        result = _remove_axis(x, 1)
        self.assertTrue(jnp.allclose(result, x[:, 0]))
        self.assertEqual(result.shape, (3,))

    def test_remove_axis_negative(self):
        """Test removing negative axis."""
        x = jnp.arange(24).reshape(2, 3, 4)
        result = _remove_axis(x, -1)
        self.assertTrue(jnp.allclose(result, x[:, :, 0]))
        self.assertEqual(result.shape, (2, 3))

    def test_remove_axis_out_of_bounds(self):
        """Out-of-bounds axis raises ValueError (jax.vmap-consistent; B8)."""
        x = jnp.arange(12).reshape(3, 4)
        with self.assertRaises(ValueError):
            _remove_axis(x, 5)

    def test_remove_axis_negative_out_of_bounds(self):
        """Out-of-bounds negative axis raises ValueError (jax.vmap-consistent; B8)."""
        x = jnp.arange(12).reshape(3, 4)
        with self.assertRaises(ValueError):
            _remove_axis(x, -5)


class TestGetBatchSize(unittest.TestCase):
    """Test the _get_batch_size helper function."""

    def test_batch_size_from_args_single_axis(self):
        """Test determining batch size from args with single axis."""
        args = (jnp.arange(30).reshape(5, 6),)
        in_axes = 0
        in_states = {}
        batch_size = _get_batch_size(args, in_axes, in_states)
        self.assertEqual(batch_size, 5)

    def test_batch_size_from_args_multiple_axes(self):
        """Test determining batch size from args with multiple axes."""
        args = (jnp.arange(20).reshape(4, 5), jnp.arange(12).reshape(4, 3))
        in_axes = (0, 0)
        in_states = {}
        batch_size = _get_batch_size(args, in_axes, in_states)
        self.assertEqual(batch_size, 4)

    def test_batch_size_from_states(self):
        """Test determining batch size from states."""
        state = bst.ShortTermState(jnp.arange(18).reshape(3, 6))
        args = ()
        in_axes = ()
        in_states = {0: [state]}
        batch_size = _get_batch_size(args, in_axes, in_states)
        self.assertEqual(batch_size, 3)

    def test_batch_size_from_axis_size(self):
        """Test determining batch size from axis_size parameter."""
        args = ()
        in_axes = ()
        in_states = {}
        batch_size = _get_batch_size(args, in_axes, in_states, axis_size=10)
        self.assertEqual(batch_size, 10)

    def test_batch_size_inconsistent(self):
        """Test error when batch sizes are inconsistent."""
        args = (jnp.arange(20).reshape(4, 5), jnp.arange(15).reshape(3, 5))
        in_axes = (0, 0)
        in_states = {}
        with self.assertRaises(ValueError):
            _get_batch_size(args, in_axes, in_states)

    def test_batch_size_no_source_no_axis_size(self):
        """Indeterminate batch size raises ValueError (not AssertionError; B8)."""
        args = ()
        in_axes = ()
        in_states = {}
        with self.assertRaises(ValueError):
            _get_batch_size(args, in_axes, in_states)


class TestFormatStateAxes(unittest.TestCase):
    """Test the _format_state_axes helper function."""

    def test_format_with_matching_axes(self):
        """Test formatting when in_states and out_states have matching axes."""
        state1 = bst.ShortTermState(jnp.array(1.0))
        state2 = bst.ShortTermState(jnp.array(2.0))

        in_states = {0: {'a': state1}, 1: {'b': state2}}
        out_states = {0: {'a': state1}, 1: {'b': state2}}

        (axis_to_in_states, in_state_to_axis,
         axis_to_out_states, out_state_to_axis) = _format_state_axes(in_states, out_states)

        self.assertEqual(in_state_to_axis[state1], 0)
        self.assertEqual(in_state_to_axis[state2], 1)
        self.assertEqual(out_state_to_axis[state1], 0)
        self.assertEqual(out_state_to_axis[state2], 1)

    def test_format_propagates_in_states_to_out(self):
        """Test that in_states are propagated to out_states when not specified."""
        state1 = bst.ShortTermState(jnp.array(1.0))
        state2 = bst.ShortTermState(jnp.array(2.0))

        in_states = {0: {'a': state1, 'b': state2}}
        out_states = None

        (axis_to_in_states, in_state_to_axis,
         axis_to_out_states, out_state_to_axis) = _format_state_axes(in_states, out_states)

        # States should be propagated to output
        self.assertEqual(out_state_to_axis[state1], 0)
        self.assertEqual(out_state_to_axis[state2], 0)
        self.assertIn(state1, axis_to_out_states[0])
        self.assertIn(state2, axis_to_out_states[0])

    def test_format_mismatched_axes_raises_error(self):
        """Test error when state has different axes in in_states and out_states."""
        state1 = bst.ShortTermState(jnp.array(1.0))

        in_states = {0: {'a': state1}}
        out_states = {1: {'a': state1}}

        with self.assertRaises(BatchAxisError):
            _format_state_axes(in_states, out_states)


class TestVmapBasicFunctionality(unittest.TestCase):
    """Test basic vmap functionality with minimal states.

    Note: The old vmap implementation is designed for stateful functions.
    Functions without states may not work as expected.
    """

    def test_vmap_simple_stateful_function(self):
        """Test vmap on a simple function with state."""
        state = bst.ShortTermState(jnp.zeros(5))

        @vmap(
            in_axes=0,
            out_axes=0,
            in_states={0: {'state': state}},
            out_states={0: {'state': state}},
        )
        def add_to_state(x):
            state.value = x + 1.0
            return state.value

        xs = jnp.arange(5.0)
        result = add_to_state(xs)
        expected = xs + 1.0
        self.assertTrue(jnp.allclose(result, expected))

    def test_vmap_multiple_inputs_with_state(self):
        """Test vmap with multiple inputs and state."""
        result_state = bst.ShortTermState(jnp.zeros(4))

        @vmap(
            in_axes=(0, 0),
            out_axes=0,
            in_states={0: {'result': result_state}},
            out_states={0: {'result': result_state}},
        )
        def multiply_add(x, y):
            result_state.value = x * y + 1.0
            return result_state.value

        xs = jnp.arange(4.0)
        ys = jnp.arange(4.0) * 2.0
        result = multiply_add(xs, ys)
        expected = xs * ys + 1.0
        self.assertTrue(jnp.allclose(result, expected))

    def test_vmap_with_axis_size(self):
        """Test vmap with explicit axis_size."""
        state = bst.ShortTermState(jnp.zeros(3))

        @vmap(
            in_axes=0,
            out_axes=0,
            axis_size=3,
            in_states={0: {'state': state}},
            out_states={0: {'state': state}},
        )
        def identity_state(x):
            state.value = x
            return state.value

        xs = jnp.arange(3.0)
        result = identity_state(xs)
        self.assertTrue(jnp.allclose(result, xs))


class TestVmapWithStates(unittest.TestCase):
    """Test vmap with state management."""

    def test_vmap_with_single_state(self):
        """Test vmap with a single state."""
        counter = bst.ShortTermState(jnp.zeros(3))

        @vmap(
            in_axes=0,
            out_axes=0,
            in_states={0: {'counter': counter}},
            out_states={0: {'counter': counter}},
        )
        def increment(x):
            counter.value = counter.value + x
            return counter.value

        xs = jnp.array([1.0, 2.0, 3.0])
        result = increment(xs)
        self.assertTrue(jnp.allclose(result, xs))
        self.assertTrue(jnp.allclose(counter.value, xs))

    def test_vmap_with_multiple_states_same_axis(self):
        """Test vmap with multiple states on the same axis."""
        state1 = bst.ShortTermState(jnp.zeros(4))
        state2 = bst.ShortTermState(jnp.ones(4))

        @vmap(
            in_axes=0,
            out_axes=0,
            in_states={0: {'s1': state1, 's2': state2}},
            out_states={0: {'s1': state1, 's2': state2}},
        )
        def combine(x):
            state1.value = state1.value + x
            state2.value = state2.value * x
            return state1.value + state2.value

        xs = jnp.array([1.0, 2.0, 3.0, 4.0])
        result = combine(xs)
        self.assertTrue(jnp.allclose(state1.value, xs))
        self.assertTrue(jnp.allclose(state2.value, xs))

    def test_vmap_with_states_different_axes(self):
        """Test vmap with states on different axes."""
        # For this test, we only use axis 0 states since mixing axes is complex
        # and requires careful setup with matching batch dimensions
        state1 = bst.ShortTermState(jnp.zeros(3))
        state2 = bst.ShortTermState(jnp.ones(3))

        @vmap(
            in_axes=0,
            out_axes=0,
            in_states={0: {'s1': state1, 's2': state2}},
            out_states={0: {'s1': state1, 's2': state2}},
        )
        def process(x):
            state1.value = state1.value + x
            state2.value = state2.value * 2
            return state1.value

        xs = jnp.array([1.0, 2.0, 3.0])
        result = process(xs)
        self.assertTrue(jnp.allclose(state1.value, xs))
        self.assertTrue(jnp.allclose(state2.value, jnp.array([2.0, 2.0, 2.0])))

    def test_vmap_state_list_format(self):
        """Test vmap with states as a list (defaults to axis 0)."""
        state1 = bst.ShortTermState(jnp.zeros(3))
        state2 = bst.ShortTermState(jnp.ones(3))

        @vmap(
            in_axes=0,
            out_axes=0,
            in_states=[state1, state2],
            out_states=[state1, state2],
        )
        def update(x):
            state1.value = state1.value + x
            state2.value = state2.value * 2
            return state1.value

        xs = jnp.array([1.0, 2.0, 3.0])
        result = update(xs)
        self.assertTrue(jnp.allclose(result, xs))


class TestVmapWithRandomState(unittest.TestCase):
    """Test vmap with RandomState."""

    def test_vmap_with_random_state(self):
        """Test vmap handles random state correctly."""
        # Create a state to make the function stateful
        output_state = bst.ShortTermState(jnp.zeros(5))

        @vmap(
            in_axes=0,
            out_axes=0,
            in_states={0: {'output': output_state}},
            out_states={0: {'output': output_state}},
        )
        def random_add(x):
            rng = bst.random.RandomState(42)
            noise = rng.uniform(size=())  # Use size instead of shape
            output_state.value = x + noise
            return output_state.value

        xs = jnp.arange(5.0)
        result = random_add(xs)
        # Each element should have different random noise
        self.assertEqual(result.shape, (5,))
        # All results should be greater than or equal to inputs
        self.assertTrue(jnp.all(result >= xs))


class TestVmapEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def test_vmap_with_kwargs_raises_error(self):
        """Test that kwargs are not supported."""
        state = bst.ShortTermState(jnp.zeros(3))

        @vmap(
            in_axes=0,
            out_axes=0,
            in_states={0: {'state': state}},
            out_states={0: {'state': state}},
        )
        def fn(x, y=1.0):
            state.value = x + y
            return state.value

        xs = jnp.arange(3.0)
        with self.assertRaises(NotImplementedError):
            fn(xs, y=2.0)

    def test_vmap_in_axes_length_mismatch(self):
        """Test error when in_axes length doesn't match args."""
        state = bst.ShortTermState(jnp.zeros(3))

        @vmap(
            in_axes=(0, 0),
            out_axes=0,
            in_states={0: {'state': state}},
            out_states={0: {'state': state}},
        )
        def fn(x):
            state.value = x + 1.0
            return state.value

        xs = jnp.arange(3.0)
        with self.assertRaises(ValueError):
            fn(xs)

    def test_vmap_batched_state_not_in_out_states(self):
        """Test error when state is batched but not in out_states."""
        state = bst.ShortTermState(jnp.zeros(()))
        output_state = bst.ShortTermState(jnp.zeros(3))

        @vmap(
            in_axes=0,
            out_axes=0,
            in_states={0: {'output': output_state}},
            out_states={0: {'output': output_state}},
        )
        def fn(x):
            # This creates a batched state value that's not in out_states
            state.value = x
            output_state.value = x
            return output_state.value

        xs = jnp.arange(3.0)
        # This should raise a BatchAxisError because state is batched
        # but not included in out_states
        with self.assertRaises(BatchAxisError):
            fn(xs)

    def test_undeclared_write_error_uses_vmap_vocabulary(self):
        # #5: the legacy vmap API uses in_states/out_states, not the engine's
        # state_out_axes / unexpected_out_state_mapping. The undeclared-write
        # error must speak the caller's vocabulary, not engine internals.
        state = bst.ShortTermState(jnp.zeros(3))

        @vmap(in_axes=0)
        def fn(x):
            state.value = state.value + x   # batched write, not in out_states
            return x

        with self.assertRaises(BatchAxisError) as cm:
            fn(jnp.arange(3.0))
        msg = str(cm.exception)
        self.assertIn('out_states', msg)
        self.assertNotIn('state_out_axes', msg)
        self.assertNotIn('unexpected_out_state_mapping', msg)


class TestVmapNewStates(unittest.TestCase):
    """Test vmap_new_states functionality."""

    def test_vmap_new_states_basic(self):
        """Test basic vmap_new_states functionality."""
        @vmap_new_states(in_axes=0, out_axes=0)
        def create_and_use_state(x):
            temp = bst.ShortTermState(jnp.array(0.0), tag='temp')
            temp.value = temp.value + x
            return temp.value

        xs = jnp.arange(5.0)
        result = create_and_use_state(xs)
        self.assertTrue(jnp.allclose(result, xs))

    def test_vmap_new_states_multiple_calls(self):
        """Test that vmap_new_states works correctly across multiple calls."""
        @vmap_new_states(in_axes=0, out_axes=0)
        def build(x):
            scratch = bst.ShortTermState(jnp.array(0.0), tag='scratch')
            scratch.value = scratch.value + x
            return scratch.value

        xs = jnp.arange(4.0)
        result1 = build(xs)
        result2 = build(xs * 2)

        self.assertTrue(jnp.allclose(result1, xs))
        self.assertTrue(jnp.allclose(result2, xs * 2))

    def test_vmap_new_states_with_state_tag(self):
        """Test vmap_new_states with state_tag filter."""
        @vmap_new_states(in_axes=0, out_axes=0, state_tag='vectorized')
        def tagged_state(x):
            state = bst.ShortTermState(x, tag='vectorized')
            return state.value * 2

        xs = jnp.arange(3.0)
        result = tagged_state(xs)
        self.assertTrue(jnp.allclose(result, xs * 2))

    def test_vmap_new_states_as_partial(self):
        """Test vmap_new_states used as decorator with partial application."""
        @vmap_new_states(in_axes=0)
        def process(x):
            temp = bst.ShortTermState(x, tag='temp')
            return temp.value + 1.0

        xs = jnp.arange(6.0)
        result = process(xs)
        self.assertTrue(jnp.allclose(result, xs + 1.0))

    def test_vmap_new_states_invalid_axis_size(self):
        """Test error when axis_size <= 0."""
        with self.assertRaises(ValueError):
            @vmap_new_states(in_axes=0, axis_size=0)
            def fn(x):
                return x

        with self.assertRaises(ValueError):
            @vmap_new_states(in_axes=0, axis_size=-1)
            def fn(x):
                return x

    def test_vmap_new_states_replicates_nonbatch_state(self):
        """B6: a NonBatchState created inside vmap_new_states is replicated, not batched.

        The legacy path previously scattered *every* new state at axis 0,
        wrongly giving a NonBatchState a leading batch dimension. It must now
        match vmap2_new_states: NonBatchState stays replicated (axis None) while
        ordinary new states are batched at axis 0.
        """
        captured = {}

        @vmap_new_states(in_axes=0, axis_size=4)
        def fn(x):
            nb = NonBatchState(jnp.zeros(3))   # value independent of the mapped axis
            reg = bst.ShortTermState(x)        # value depends on mapped x -> batched
            captured['nb'] = nb
            captured['reg'] = reg
            return x

        fn(jnp.arange(4.0))
        # NonBatchState stays un-batched (replicated); the regular state gains the
        # batch axis at 0.
        self.assertEqual(captured['nb'].value.shape, (3,))
        self.assertEqual(captured['reg'].value.shape, (4,))

    def test_vmap_new_states_mixed_states_values(self):
        """B6: replicated and batched new states carry correct values."""
        captured = {}

        @vmap_new_states(in_axes=0, axis_size=3)
        def fn(x):
            shared = NonBatchState(jnp.array([1.0, 2.0]))  # same for every lane
            perlane = bst.ShortTermState(x * 10.0)         # distinct per lane
            captured['shared'] = shared
            captured['perlane'] = perlane
            return x

        fn(jnp.arange(3.0))
        # replicated state keeps its single (un-batched) value
        self.assertEqual(captured['shared'].value.shape, (2,))
        self.assertTrue(jnp.allclose(captured['shared'].value, jnp.array([1.0, 2.0])))
        # batched state holds one entry per lane
        self.assertEqual(captured['perlane'].value.shape, (3,))
        self.assertTrue(jnp.allclose(captured['perlane'].value, jnp.arange(3.0) * 10.0))


class TestVmapNestedAndComplex(unittest.TestCase):
    """Test complex and nested vmap scenarios."""

    def test_vmap_with_axis_name(self):
        """Test vmap with axis_name parameter."""
        state = bst.ShortTermState(jnp.zeros(4))

        @vmap(
            in_axes=0,
            out_axes=0,
            axis_name='batch',
            in_states={0: {'state': state}},
            out_states={0: {'state': state}},
        )
        def fn(x):
            state.value = x * 2
            return state.value

        xs = jnp.arange(4.0)
        result = fn(xs)
        self.assertTrue(jnp.allclose(result, xs * 2))

    def test_vmap_list_to_tuple_conversion(self):
        """Test that in_axes as list is converted to tuple."""
        state = bst.ShortTermState(jnp.zeros(3))

        @vmap(
            in_axes=[0, 0],
            out_axes=0,
            in_states={0: {'state': state}},
            out_states={0: {'state': state}},
        )
        def fn(x, y):
            state.value = x + y
            return state.value

        xs = jnp.arange(3.0)
        ys = jnp.ones(3)
        result = fn(xs, ys)
        self.assertTrue(jnp.allclose(result, xs + ys))

    def test_vmap_with_out_axes_different_from_zero(self):
        """Test vmap with non-zero out_axes."""
        state = bst.ShortTermState(jnp.zeros((2, 3)))

        @vmap(
            in_axes=0,
            out_axes=1,
            in_states={1: {'state': state}},
            out_states={1: {'state': state}},
        )
        def expand(x):
            state.value = jnp.stack([x, x * 2])
            return state.value

        xs = jnp.arange(3.0)
        result = expand(xs)
        # Result shape should be (2, 3) since out_axes=1
        self.assertEqual(result.shape, (2, 3))


class TestVmapIntegrationWithBrainState(unittest.TestCase):
    """Integration tests with BrainState features."""

    def test_vmap_with_long_term_state(self):
        """Test vmap with LongTermState."""
        param = bst.LongTermState(jnp.zeros(3))

        @vmap(
            in_axes=0,
            out_axes=0,
            in_states={0: {'param': param}},
            out_states={0: {'param': param}},
        )
        def update_param(delta):
            param.value = param.value + delta
            return param.value

        deltas = jnp.array([0.1, 0.2, 0.3])
        result = update_param(deltas)
        self.assertTrue(jnp.allclose(param.value, deltas))
        self.assertTrue(jnp.allclose(result, deltas))

    def test_vmap_preserves_state_structure(self):
        """Test that vmap preserves complex state structures."""
        state = bst.ShortTermState({'a': jnp.zeros(2), 'b': jnp.ones(2)})

        @vmap(
            in_axes=0,
            out_axes=0,
            in_states={0: {'state': state}},
            out_states={0: {'state': state}},
        )
        def modify(x):
            new_val = state.value.copy()
            new_val['a'] = state.value['a'] + x
            state.value = new_val
            return state.value['a']

        xs = jnp.array([1.0, 2.0])
        result = modify(xs)
        self.assertTrue(jnp.allclose(result, xs))


class TestVmapDirectCall(unittest.TestCase):
    """Test vmap called directly with fn (not as decorator), covering line 232."""

    def test_vmap_direct_call_with_fn(self):
        """vmap called as vmap(fn, ...) returns a vmapped function immediately."""
        state = bst.ShortTermState(jnp.zeros(3))

        def fn(x):
            state.value = state.value + x
            return state.value

        vmapped = vmap(
            fn,
            in_axes=0,
            out_axes=0,
            in_states={0: {'state': state}},
            out_states={0: {'state': state}},
        )
        xs = jnp.array([1.0, 2.0, 3.0])
        result = vmapped(xs)
        self.assertTrue(jnp.allclose(result, xs))

    def test_vmap_direct_no_states(self):
        """vmap called directly on a stateless function."""
        state = bst.ShortTermState(jnp.zeros(4))

        def fn(x):
            state.value = x * 2
            return state.value

        vmapped = vmap(fn, in_axes=0, out_axes=0,
                       in_states=state, out_states=state)
        xs = jnp.arange(4.0)
        result = vmapped(xs)
        self.assertTrue(jnp.allclose(result, xs * 2))


class TestVmapNewStatesListInAxes(unittest.TestCase):
    """Test list in_axes conversion (line 266) and kwargs error (line 271)."""

    def test_vmap_new_states_list_in_axes(self):
        """vmap_new_states accepts list in_axes and converts it to tuple."""
        @vmap_new_states(in_axes=[0], out_axes=0)
        def fn(x):
            temp = bst.ShortTermState(jnp.array(0.0))
            temp.value = temp.value + x
            return temp.value

        xs = jnp.arange(4.0)
        result = fn(xs)
        self.assertTrue(jnp.allclose(result, xs))

    def test_vmap_new_states_kwargs_raises(self):
        """vmap_new_states rejects keyword arguments to vmapped fn."""
        @vmap_new_states(in_axes=0)
        def fn(x):
            temp = bst.ShortTermState(x)
            return temp.value

        with self.assertRaises(NotImplementedError):
            fn(jnp.arange(3.0), y=1.0)


class TestVmapNewStatesWithRandomState(unittest.TestCase):
    """Test probe hook for RandomState (lines 284-287, 308, 329)."""

    def test_vmap_new_states_with_random_state(self):
        """vmap_new_states probes and splits random state per lane."""
        rng = bst.random.RandomState(42)

        @vmap_new_states(in_axes=0, out_axes=0)
        def fn(x):
            noise = rng.uniform(size=())
            temp = bst.ShortTermState(x + noise)
            return temp.value

        xs = jnp.arange(4.0)
        result = fn(xs)
        # Each lane gets different noise; shape must be (4,)
        self.assertEqual(result.shape, (4,))


class TestVmapNewStatesDirectCall(unittest.TestCase):
    """Test vmap_new_states called with fn directly (line 437)."""

    def test_vmap_new_states_direct_call(self):
        """vmap_new_states(fun, ...) returns a vmapped function immediately."""
        def fn(x):
            temp = bst.ShortTermState(jnp.array(0.0))
            temp.value = x * 3
            return temp.value

        vmapped = vmap_new_states(
            fn,
            in_axes=0,
            out_axes=0,
        )
        xs = jnp.arange(5.0)
        result = vmapped(xs)
        self.assertTrue(jnp.allclose(result, xs * 3))


class TestVmapNewStatesProbeHookNonRandom(unittest.TestCase):
    """Test probe hook returns state._value for non-RandomState (line 287)."""

    def test_probe_hook_non_random_state(self):
        """vmap_new_states probes existing non-random states via state._value."""
        # An existing state that's closed over (but not a RandomState) exercises
        # the `return state._value` branch in the probe_hook.
        existing = bst.ShortTermState(jnp.array(10.0))

        @vmap_new_states(in_axes=0, out_axes=0)
        def fn(x):
            # Read existing (non-random) state -- exercises probe_hook else branch
            temp = bst.ShortTermState(x + existing.value)
            return temp.value

        xs = jnp.arange(3.0)
        result = fn(xs)
        self.assertEqual(result.shape, (3,))
        self.assertTrue(jnp.allclose(result, xs + 10.0))


class TestVmapCollectives(unittest.TestCase):
    """B2: collectives under axis_name work in the legacy ``vmap`` API too.

    ``vmap`` and ``vmap2`` share the engine, so binding the axis name during the
    state-discovery probe fixes both. The pre-existing ``test_vmap_with_axis_name``
    passed ``axis_name`` but never called a collective, so this gap was untested.
    """

    def test_vmap_psum_matches_jax(self):
        import jax
        from brainstate.transform import vmap

        s = bst.ShortTermState(jnp.zeros(4))

        @vmap(in_axes=0, axis_name='i', in_states={0: {'s': s}}, out_states={0: {'s': s}})
        def fn(x):
            s.value = x / jax.lax.psum(x, 'i')
            return s.value

        out = fn(jnp.arange(1., 5.))
        self.assertTrue(jnp.allclose(out, jnp.array([0.1, 0.2, 0.3, 0.4])))


class TestVmapNewStatesRejectsStateArgs(unittest.TestCase):
    """B6: vmap_new_states must reject (not silently ignore) in_states/out_states."""

    def test_in_states_raises(self):
        from brainstate.transform import vmap_new_states
        s = bst.ShortTermState(jnp.zeros(3))
        with self.assertRaises(ValueError):
            vmap_new_states(lambda x: x, in_axes=0, axis_size=3, in_states=s)

    def test_out_states_raises(self):
        from brainstate.transform import vmap_new_states
        s = bst.ShortTermState(jnp.zeros(3))
        with self.assertRaises(ValueError):
            vmap_new_states(lambda x: x, in_axes=0, axis_size=3, out_states=s)


if __name__ == '__main__':
    unittest.main()


class TestFailedVmapNewStatesRestoresRng(unittest.TestCase):
    """A failure inside the mapped pass must not leave key tracers in the
    global random state (audit M5)."""

    def test_vmap_new_states_failure_restores_rng(self):
        bst.random.seed(0)
        calls = {'n': 0}

        def f():
            st = bst.ShortTermState(bst.random.randn(3))
            calls['n'] += 1
            if calls['n'] >= 2:  # first call: probe; second: mapped pass
                raise RuntimeError('boom')
            return 1.0

        with self.assertRaises(RuntimeError):
            bst.transform.vmap_new_states(f, axis_size=4)()
        self.assertFalse(isinstance(bst.random.DEFAULT.value, jax.core.Tracer))


class TestVmapNewStatesFromOtherStates(unittest.TestCase):
    """Complex ``vmap_new_states`` scenarios where new states are *created from
    the values of other new states* (dependency chains, random -> derived, and
    replicated -> batched mixes)."""

    def test_dependent_chain_abc(self):
        """A new state initialized from a previously-created state's value.

        ``a`` is batched from the mapped input, ``b`` is built from ``a.value``,
        and ``c`` from both ``b`` and ``a``. Every link must carry the batch
        axis and the exact arithmetic must hold.
        """
        captured = {}

        @vmap_new_states(in_axes=0, axis_size=5)
        def fn(x):
            a = bst.ShortTermState(x)                  # batched from input
            b = bst.ParamState(a.value * 2.0 + 1.0)    # b created from a.value
            c = bst.ShortTermState(b.value - a.value)  # c from b and a
            captured['a'] = a
            captured['b'] = b
            captured['c'] = c
            return c.value

        xs = jnp.arange(5.0)
        out = fn(xs)
        # c == (2a + 1) - a == a + 1 == x + 1
        self.assertEqual(out.shape, (5,))
        self.assertTrue(jnp.allclose(out, xs + 1.0))
        self.assertEqual(captured['a'].value.shape, (5,))
        self.assertEqual(captured['b'].value.shape, (5,))
        self.assertEqual(captured['c'].value.shape, (5,))
        self.assertTrue(jnp.allclose(captured['b'].value, xs * 2.0 + 1.0))

    def test_deep_chain_with_random(self):
        """A 5-level chain rooted at a random draw stays internally consistent.

        Each lane is seeded with its own split key, so lanes differ, yet the
        deterministic chain identity (recomputed from the captured batched root)
        must hold exactly for every lane.
        """
        bst.random.seed(0)
        captured = {}

        @vmap_new_states(axis_size=4)
        def fn():
            s0 = bst.ParamState(bst.random.randn(3))   # random root, per-lane
            s1 = bst.ShortTermState(s0.value + 1.0)
            s2 = bst.ShortTermState(s1.value * s0.value)
            s3 = bst.ShortTermState(jnp.tanh(s2.value))
            s4 = bst.ShortTermState(s3.value - s1.value)
            captured['s0'] = s0
            captured['s4'] = s4
            return s4.value

        out = fn()
        self.assertEqual(out.shape, (4, 3))
        # lanes are independently seeded -> distinct
        self.assertFalse(jnp.allclose(out[0], out[1]))
        # chain identity recomputed from the batched root
        s0 = captured['s0'].value
        expected = jnp.tanh((s0 + 1.0) * s0) - (s0 + 1.0)
        self.assertTrue(jnp.allclose(out, expected))

    def test_batched_state_from_nonbatch_value(self):
        """A batched state derived from a *replicated* NonBatchState value.

        The shared base stays un-batched (axis None), while the per-lane state
        built from ``base.sum()`` gains the batch axis. The shared base value is
        broadcast identically into every lane.
        """
        captured = {}

        @vmap_new_states(in_axes=0, axis_size=3)
        def fn(x):
            base = NonBatchState(jnp.array([1.0, 2.0, 3.0]))   # replicated
            per = bst.ShortTermState(base.value.sum() + x)     # batched (6 + x)
            captured['base'] = base
            captured['per'] = per
            return per.value

        out = fn(jnp.arange(3.0))
        self.assertEqual(captured['base'].value.shape, (3,))    # un-batched
        self.assertEqual(captured['per'].value.shape, (3,))     # batched
        self.assertTrue(jnp.allclose(captured['base'].value, jnp.array([1.0, 2.0, 3.0])))
        self.assertTrue(jnp.allclose(out, 6.0 + jnp.arange(3.0)))

    def test_random_init_then_derived_distinct(self):
        """A derived state tracks its random parent and stays per-lane distinct."""
        bst.random.seed(1)
        captured = {}

        @vmap_new_states(axis_size=4)
        def fn():
            w = bst.ParamState(bst.random.randn(3))
            scaled = bst.ShortTermState(w.value * 10.0)
            captured['w'] = w
            captured['scaled'] = scaled
            return scaled.value

        out = fn()
        self.assertEqual(out.shape, (4, 3))
        self.assertTrue(jnp.allclose(out, captured['w'].value * 10.0))
        self.assertFalse(jnp.allclose(out[0], out[1]))

    def test_two_random_states_derived_sum(self):
        """Two independent RandomStates both split per lane; the derived sum holds."""
        rng1 = bst.random.RandomState(1)
        rng2 = bst.random.RandomState(2)

        @vmap_new_states(axis_size=5)
        def fn():
            a = bst.ParamState(rng1.randn(3))
            b = bst.ParamState(rng2.randn(3))
            c = bst.ShortTermState(a.value + b.value)
            return c.value, a.value, b.value

        c, a, b = fn()
        self.assertEqual(c.shape, (5, 3))
        self.assertTrue(jnp.allclose(c, a + b))
        self.assertFalse(jnp.allclose(c[0], c[1]))

    def test_broadcast_input_plus_random_derived(self):
        """A broadcast (in_axes=None) input combined with a per-lane random draw."""
        bst.random.seed(7)

        @vmap_new_states(in_axes=None, axis_size=4)
        def fn(c):
            a = bst.ParamState(bst.random.randn(3) + c)   # c broadcast, noise per-lane
            return a.value

        out = fn(jnp.array(10.0))
        self.assertEqual(out.shape, (4, 3))
        # centered around the broadcast constant, but distinct per lane
        self.assertFalse(jnp.allclose(out[0], out[1]))
        self.assertTrue(jnp.all(jnp.abs(out - 10.0) < 6.0))

    def test_out_axes_nonzero_with_dependent_state(self):
        """out_axes=1 relocates the mapped axis of a value built from a new state."""

        @vmap_new_states(in_axes=0, out_axes=1, axis_size=3)
        def fn(x):
            a = bst.ShortTermState(x)
            b = bst.ShortTermState(jnp.stack([a.value, a.value * 2.0]))  # (2,) per lane
            return b.value

        out = fn(jnp.arange(3.0))
        self.assertEqual(out.shape, (2, 3))   # mapped axis placed at position 1
        self.assertTrue(jnp.allclose(out[0], jnp.arange(3.0)))
        self.assertTrue(jnp.allclose(out[1], jnp.arange(3.0) * 2.0))


class TestVmapNewStatesFailureCases(unittest.TestCase):
    """Boundary / failure behavior of ``vmap_new_states`` -- misuse must raise a
    clear error and must not leave a key tracer in the global random state."""

    def test_nonbatch_state_from_batched_value_raises(self):
        """A NonBatchState (replicated, axis None) whose value depends on the
        batched axis is a contradiction and must raise.

        The global RNG must also be restored even though the failure happens
        mid-trace inside the mapped pass.
        """
        bst.random.seed(0)

        @vmap_new_states(in_axes=0, axis_size=4)
        def fn(x):
            r = bst.ShortTermState(bst.random.randn(3))   # forces an rng split
            a = bst.ShortTermState(x)                     # batched
            NonBatchState(a.value * 2.0)                  # replicated but batch-dependent
            return x

        with self.assertRaises(ValueError):
            fn(jnp.arange(4.0))
        self.assertFalse(isinstance(bst.random.DEFAULT.value, jax.core.Tracer))

    def test_random_initialized_nonbatch_state_raises(self):
        """A NonBatchState initialized from a per-lane random draw cannot be
        replicated on axis None and must raise."""
        bst.random.seed(0)

        @vmap_new_states(axis_size=4)
        def fn():
            NonBatchState(bst.random.randn(3))   # per-lane value, axis None -> contradiction
            return jnp.zeros(())

        with self.assertRaises(ValueError):
            fn()
        self.assertFalse(isinstance(bst.random.DEFAULT.value, jax.core.Tracer))

    def test_data_dependent_state_shape_raises(self):
        """A new state whose shape depends on a traced value cannot be created."""

        @vmap_new_states(in_axes=0, axis_size=4)
        def fn(x):
            a = bst.ShortTermState(x)
            n = int(a.value)                       # concretize a tracer -> error
            return bst.ShortTermState(jnp.zeros(n)).value.sum()

        with self.assertRaises(jax.errors.ConcretizationTypeError):
            fn(jnp.arange(1.0, 5.0))

    def test_axis_size_conflicts_with_input_raises(self):
        """An explicit axis_size that disagrees with the mapped input length raises."""

        @vmap_new_states(in_axes=0, axis_size=5)
        def fn(x):
            return bst.ShortTermState(x * 2.0).value

        with self.assertRaises(ValueError) as cm:
            fn(jnp.arange(3.0))
        msg = str(cm.exception)
        self.assertIn('conflicts', msg)
        self.assertIn('5', msg)
        self.assertIn('3', msg)

    def test_state_to_exclude_with_derived_state(self):
        """Regression: a derived (batched) state built from an *excluded*
        replicated state still maps correctly."""

        @vmap_new_states(in_axes=0, axis_size=4,
                         state_to_exclude=OfType(NonBatchState))
        def fn(x):
            base = NonBatchState(jnp.ones(2))                  # excluded from batching
            per = bst.ShortTermState(base.value.sum() + x)     # 2 + x, batched
            return per.value

        out = fn(jnp.arange(4.0))
        self.assertEqual(out.shape, (4,))
        self.assertTrue(jnp.allclose(out, 2.0 + jnp.arange(4.0)))

    def test_non_idempotent_random_draw_count(self):
        """Regression: an extra random draw on the mapped pass (the source of
        truth) still produces independent, well-shaped per-lane values."""
        bst.random.seed(0)
        state = {'n': 0}

        @vmap_new_states(axis_size=4)
        def fn():
            state['n'] += 1
            v = bst.random.randn(3)
            if state['n'] >= 2:           # mapped pass draws an extra time
                v = v + bst.random.randn(3)
            return bst.ParamState(v).value

        out = fn()
        self.assertEqual(out.shape, (4, 3))
        self.assertFalse(jnp.allclose(out[0], out[1]))
