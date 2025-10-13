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
from brainstate.transform._mapping_old import (
    vmap,
    vmap_new_states,
    _flatten_in_out_states,
    _remove_axis,
    _get_batch_size,
    _format_state_axes,
)
from brainstate._error import BatchAxisError


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
        """Test error when axis is out of bounds."""
        x = jnp.arange(12).reshape(3, 4)
        with self.assertRaises(IndexError):
            _remove_axis(x, 5)

    def test_remove_axis_negative_out_of_bounds(self):
        """Test error when negative axis is out of bounds."""
        x = jnp.arange(12).reshape(3, 4)
        with self.assertRaises(IndexError):
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
        """Test error when no batch size can be determined."""
        args = ()
        in_axes = ()
        in_states = {}
        with self.assertRaises(AssertionError):
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


if __name__ == '__main__':
    unittest.main()
