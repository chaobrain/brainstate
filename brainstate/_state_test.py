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


import unittest

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

import brainstate
from brainstate import HookConfig, HookManager


class TestBasicState(unittest.TestCase):
    """Test the basic State class functionality."""

    def test_state_initialization(self):
        """Test basic state initialization."""
        value = jnp.array([1.0, 2.0, 3.0])
        state = brainstate.State(value, name='test_state')

        self.assertEqual(state.name, 'test_state')
        np.testing.assert_array_equal(state.value, value)
        self.assertIsNotNone(state.source_info)

    def test_state_without_name(self):
        """Test state initialization without a name."""
        state = brainstate.State(jnp.zeros(5))
        self.assertIsNone(state.name)

    def test_state_value_setter(self):
        """Test setting state values."""
        state = brainstate.State(jnp.zeros(3))
        new_value = jnp.array([1.0, 2.0, 3.0])
        state.value = new_value
        np.testing.assert_array_equal(state.value, new_value)

    def test_state_value_from_another_state_fails(self):
        """Test that setting value from another State raises an error."""
        state1 = brainstate.State(jnp.zeros(3))
        state2 = brainstate.State(jnp.ones(3))

        with self.assertRaises(ValueError):
            state1.value = state2

    def test_state_numel(self):
        """Test numel calculation."""
        state = brainstate.State(jnp.zeros((2, 3, 4)))
        self.assertEqual(state.numel(), 24)

        state2 = brainstate.State({'a': jnp.zeros(5), 'b': jnp.zeros((2, 3))})
        self.assertEqual(state2.numel(), 11)

    def test_state_copy(self):
        """Test state copying."""
        state = brainstate.State(jnp.array([1.0, 2.0]), name='original')
        state_copy = state.copy()

        self.assertEqual(state_copy.name, 'original')
        np.testing.assert_array_equal(state_copy.value, state.value)
        self.assertIsNot(state_copy, state)

    def test_state_replace(self):
        """Test state replace method."""
        state = brainstate.State(jnp.array([1.0, 2.0]), name='test')
        new_state = state.replace(value=jnp.array([3.0, 4.0]))

        np.testing.assert_array_equal(new_state.value, jnp.array([3.0, 4.0]))
        self.assertEqual(new_state.name, 'test')

    def test_state_replace_with_name(self):
        """Test state replace with name update."""
        state = brainstate.State(jnp.array([1.0, 2.0]), name='test')
        new_state = state.replace(_name='new_name')

        self.assertEqual(new_state.name, 'new_name')
        np.testing.assert_array_equal(new_state.value, state.value)

    def test_state_restore_value(self):
        """Test restoring state values."""
        state = brainstate.State(jnp.zeros(3))
        original = state.value.copy()

        state.value = jnp.ones(3)
        state.restore_value(original)

        np.testing.assert_array_equal(state.value, original)

    def test_state_hashable(self):
        """Test that states are hashable."""
        state = brainstate.State(jnp.zeros(3))
        hash_val = hash(state)
        self.assertIsInstance(hash_val, int)

        # Can be used in sets and dicts
        state_set = {state}
        self.assertIn(state, state_set)

    def test_state_to_state_ref(self):
        """Test converting state to TreefyState reference."""
        state = brainstate.State(jnp.array([1.0, 2.0]), name='test')
        state_ref = state.to_state_ref()

        self.assertIsInstance(state_ref, brainstate.TreefyState)
        np.testing.assert_array_equal(state_ref.value, state.value)
        self.assertEqual(state_ref.name, 'test')

    def test_state_update_from_ref(self):
        """Test updating state from TreefyState reference."""
        state = brainstate.State(jnp.zeros(3), name='test')
        state_ref = brainstate.TreefyState(brainstate.State, jnp.ones(3), _name='test', _been_writen=True)

        state.update_from_ref(state_ref)
        np.testing.assert_array_equal(state.value, jnp.ones(3))

    def test_state_stack_level(self):
        """Test state stack level management."""
        state = brainstate.State(jnp.zeros(3))
        initial_level = state.stack_level

        state.increase_stack_level()
        self.assertEqual(state.stack_level, initial_level + 1)

        state.decrease_stack_level()
        self.assertEqual(state.stack_level, initial_level)

        # Should not go below 0
        state.stack_level = 0
        state.decrease_stack_level()
        self.assertEqual(state.stack_level, 0)


class TestStateSourceInfo(unittest.TestCase):
    """Test state source information tracking."""

    def test_state_source_info(self):
        """Test that source info is captured."""
        state = brainstate.State(brainstate.random.randn(10))
        self.assertIsNotNone(state.source_info)

    def test_state_value_tree(self):
        """Test state value tree checking."""
        state = brainstate.ShortTermState(jnp.zeros((2, 3)))

        with brainstate.check_state_value_tree():
            state.value = jnp.zeros((2, 3))

            with self.assertRaises(ValueError):
                state.value = (jnp.zeros((2, 3)), jnp.zeros((2, 3)))


class TestStateRepr(unittest.TestCase):
    """Test state string representation."""

    def test_state_repr(self):
        """Test basic state representation."""
        state = brainstate.State(brainstate.random.randn(10))
        repr_str = repr(state)
        self.assertIsInstance(repr_str, str)

    def test_state_dict_repr(self):
        """Test state representation with dict value."""
        state = brainstate.State({'a': brainstate.random.randn(10), 'b': brainstate.random.randn(10)})
        repr_str = repr(state)
        self.assertIsInstance(repr_str, str)

    def test_state_list_repr(self):
        """Test state representation with list value."""
        state = brainstate.State([brainstate.random.randn(10), brainstate.random.randn(10)])
        repr_str = repr(state)
        self.assertIsInstance(repr_str, str)


class TestShortTermState(unittest.TestCase):
    """Test ShortTermState functionality."""

    def test_short_term_state_creation(self):
        """Test creating a short-term state."""
        state = brainstate.ShortTermState(jnp.zeros(5), name='short_term')
        self.assertIsInstance(state, brainstate.ShortTermState)
        self.assertIsInstance(state, brainstate.State)
        self.assertEqual(state.name, 'short_term')

    def test_short_term_state_semantics(self):
        """Test that ShortTermState behaves like State."""
        state = brainstate.ShortTermState(jnp.array([1.0, 2.0, 3.0]))
        state.value = jnp.array([4.0, 5.0, 6.0])
        np.testing.assert_array_equal(state.value, jnp.array([4.0, 5.0, 6.0]))


class TestLongTermState(unittest.TestCase):
    """Test LongTermState functionality."""

    def test_long_term_state_creation(self):
        """Test creating a long-term state."""
        state = brainstate.LongTermState(jnp.zeros(5), name='long_term')
        self.assertIsInstance(state, brainstate.LongTermState)
        self.assertIsInstance(state, brainstate.State)
        self.assertEqual(state.name, 'long_term')

    def test_long_term_state_semantics(self):
        """Test that LongTermState behaves like State."""
        state = brainstate.LongTermState(jnp.array([1.0, 2.0, 3.0]))
        state.value = jnp.array([4.0, 5.0, 6.0])
        np.testing.assert_array_equal(state.value, jnp.array([4.0, 5.0, 6.0]))


class TestParamState(unittest.TestCase):
    """Test ParamState functionality."""

    def test_param_state_creation(self):
        """Test creating a parameter state."""
        state = brainstate.ParamState(jnp.zeros((3, 3)), name='weights')
        self.assertIsInstance(state, brainstate.ParamState)
        self.assertIsInstance(state, brainstate.LongTermState)
        self.assertEqual(state.name, 'weights')

    def test_param_state_typical_use(self):
        """Test typical parameter state usage."""
        weights = brainstate.ParamState(jnp.ones((10, 5)), name='layer_weights')
        bias = brainstate.ParamState(jnp.zeros(5), name='layer_bias')

        self.assertEqual(weights.value.shape, (10, 5))
        self.assertEqual(bias.value.shape, (5,))


class TestBatchState(unittest.TestCase):
    """Test BatchState functionality."""

    def test_batch_state_creation(self):
        """Test creating a batch state."""
        state = brainstate.BatchState(jnp.zeros((32, 10)), name='batch')
        self.assertIsInstance(state, brainstate.BatchState)
        self.assertIsInstance(state, brainstate.LongTermState)

    def test_batch_state_semantics(self):
        """Test batch state typical usage."""
        batch = brainstate.BatchState(jnp.array([[1, 2], [3, 4], [5, 6]]))
        self.assertEqual(batch.value.shape, (3, 2))


class TestHiddenState(unittest.TestCase):
    """Test HiddenState functionality."""

    def test_hidden_state_creation(self):
        """Test creating a hidden state."""
        state = brainstate.HiddenState(jnp.zeros(10), name='hidden')
        self.assertIsInstance(state, brainstate.HiddenState)
        self.assertIsInstance(state, brainstate.ShortTermState)

    def test_hidden_state_with_array(self):
        """Test HiddenState with numpy array."""
        state = brainstate.HiddenState(np.zeros(5))
        self.assertEqual(state.varshape, (5,))
        self.assertEqual(state.num_state, 1)

    def test_hidden_state_with_jax_array(self):
        """Test HiddenState with JAX array."""
        state = brainstate.HiddenState(jnp.zeros((3, 4)))
        self.assertEqual(state.varshape, (3, 4))
        self.assertEqual(state.num_state, 1)

    def test_hidden_state_with_quantity(self):
        """Test HiddenState with brainunit Quantity."""
        state = brainstate.HiddenState(jnp.zeros(5) * u.mV)
        self.assertEqual(state.varshape, (5,))
        self.assertEqual(state.num_state, 1)

    def test_hidden_state_invalid_type(self):
        """Test that invalid types raise TypeError."""
        with self.assertRaises(TypeError):
            brainstate.HiddenState([1, 2, 3])  # Python list not allowed

        with self.assertRaises(TypeError):
            brainstate.HiddenState({'a': 1})  # Dict not allowed

    def test_hidden_state_varshape(self):
        """Test varshape property."""
        state = brainstate.HiddenState(jnp.zeros((10, 20)))
        self.assertEqual(state.varshape, (10, 20))

    def test_hidden_state_num_state(self):
        """Test num_state property."""
        state = brainstate.HiddenState(jnp.zeros(10))
        self.assertEqual(state.num_state, 1)


class TestHiddenGroupState(unittest.TestCase):
    """Test HiddenGroupState functionality."""

    def test_hidden_group_state_creation(self):
        """Test creating a hidden group state."""
        value = np.random.randn(10, 10, 5)
        state = brainstate.HiddenGroupState(value)

        self.assertIsInstance(state, brainstate.HiddenGroupState)
        self.assertEqual(state.num_state, 5)
        self.assertEqual(state.varshape, (10, 10))

    def test_hidden_group_state_with_quantity(self):
        """Test HiddenGroupState with Quantity."""
        value = np.random.randn(10, 10, 5) * u.mV
        state = brainstate.HiddenGroupState(value)

        self.assertEqual(state.num_state, 5)
        self.assertEqual(state.varshape, (10, 10))

    def test_hidden_group_state_invalid_dimensions(self):
        """Test that 1D arrays raise ValueError."""
        with self.assertRaises(ValueError):
            brainstate.HiddenGroupState(np.zeros(5))

    def test_hidden_group_state_get_value_by_index(self):
        """Test getting value by integer index."""
        value = np.random.randn(10, 10, 3)
        state = brainstate.HiddenGroupState(value)

        first_state = state.get_value(0)
        self.assertEqual(first_state.shape, (10, 10))
        np.testing.assert_array_equal(first_state, value[..., 0])

    def test_hidden_group_state_get_value_by_name(self):
        """Test getting value by string name."""
        value = np.random.randn(10, 10, 3)
        state = brainstate.HiddenGroupState(value)

        first_state = state.get_value('0')
        second_state = state.get_value('1')

        np.testing.assert_array_equal(first_state, value[..., 0])
        np.testing.assert_array_equal(second_state, value[..., 1])

    def test_hidden_group_state_get_value_invalid_index(self):
        """Test that invalid indices raise errors."""
        value = np.random.randn(10, 10, 3)
        state = brainstate.HiddenGroupState(value)

        with self.assertRaises(AssertionError):
            state.get_value(5)  # Out of range

        with self.assertRaises(AssertionError):
            state.get_value('invalid')  # Invalid name

    def test_hidden_group_state_set_value_dict(self):
        """Test setting values with dictionary."""
        value = jnp.array(np.random.randn(10, 10, 3))
        state = brainstate.HiddenGroupState(value)

        new_val = jnp.ones((10, 10))
        state.set_value({0: new_val})

        np.testing.assert_array_equal(state.get_value(0), new_val)

    def test_hidden_group_state_set_value_by_name(self):
        """Test setting values by name."""
        value = jnp.array(np.random.randn(10, 10, 3))
        state = brainstate.HiddenGroupState(value)

        new_val = jnp.ones((10, 10))
        state.set_value({'1': new_val})

        np.testing.assert_array_equal(state.get_value(1), new_val)

    def test_hidden_group_state_set_value_list(self):
        """Test setting values with list."""
        value = jnp.array(np.random.randn(10, 10, 3))
        state = brainstate.HiddenGroupState(value)

        new_vals = [jnp.ones((10, 10)), jnp.zeros((10, 10))]
        state.set_value(new_vals)

        np.testing.assert_array_equal(state.get_value(0), new_vals[0])
        np.testing.assert_array_equal(state.get_value(1), new_vals[1])

    def test_hidden_group_state_set_value_wrong_shape(self):
        """Test that wrong shapes raise errors."""
        value = np.random.randn(10, 10, 3)
        state = brainstate.HiddenGroupState(value)

        with self.assertRaises(AssertionError):
            state.set_value({0: np.ones((5, 5))})  # Wrong shape

    def test_hidden_group_state_name2index(self):
        """Test name2index mapping."""
        value = np.random.randn(10, 10, 5)
        state = brainstate.HiddenGroupState(value)

        self.assertEqual(len(state.name2index), 5)
        self.assertEqual(state.name2index['0'], 0)
        self.assertEqual(state.name2index['4'], 4)


class TestHiddenTreeState(unittest.TestCase):
    """Test HiddenTreeState functionality."""

    def test_hidden_tree_state_from_list(self):
        """Test creating HiddenTreeState from list."""
        value = [
            np.random.randn(10, 10) * u.mV,
            np.random.randn(10, 10) * u.mA,
            np.random.randn(10, 10) * u.mS
        ]
        state = brainstate.HiddenTreeState(value)

        self.assertEqual(state.num_state, 3)
        self.assertEqual(state.varshape, (10, 10))

    def test_hidden_tree_state_from_dict(self):
        """Test creating HiddenTreeState from dict."""
        value = {
            'v': np.random.randn(10, 10) * u.mV,
            'i': np.random.randn(10, 10) * u.mA,
            'g': np.random.randn(10, 10) * u.mS
        }
        state = brainstate.HiddenTreeState(value)

        self.assertEqual(state.num_state, 3)
        self.assertEqual(state.varshape, (10, 10))
        self.assertIn('v', state.name2index)
        self.assertIn('i', state.name2index)
        self.assertIn('g', state.name2index)

    def test_hidden_tree_state_get_value_by_name(self):
        """Test getting values by name."""
        value = {
            'v': np.random.randn(10, 10) * u.mV,
            'i': np.random.randn(10, 10) * u.mA,
        }
        state = brainstate.HiddenTreeState(value)

        v_val = state.get_value('v')
        self.assertEqual(v_val.shape, (10, 10))
        self.assertIsInstance(v_val, u.Quantity)
        self.assertEqual(v_val.unit, u.mV)

    def test_hidden_tree_state_get_value_by_index(self):
        """Test getting values by index."""
        value = {
            'v': np.random.randn(10, 10) * u.mV,
            'i': np.random.randn(10, 10) * u.mA,
        }
        state = brainstate.HiddenTreeState(value)

        val0 = state.get_value(0)
        val1 = state.get_value(1)

        self.assertEqual(val0.shape, (10, 10))
        self.assertEqual(val1.shape, (10, 10))

    def test_hidden_tree_state_set_value_dict(self):
        """Test setting values with dict."""
        value = {
            'v': brainstate.random.randn(10, 10) * u.mV,
            'i': brainstate.random.randn(10, 10) * u.mA,
        }
        state = brainstate.HiddenTreeState(value)

        new_v = jnp.ones((10, 10)) * u.mV
        state.set_value({'v': new_v})

        retrieved = state.get_value('v')
        np.testing.assert_array_almost_equal(retrieved.mantissa, new_v.mantissa)

    def test_hidden_tree_state_set_value_list(self):
        """Test setting values with list."""
        value = [
            brainstate.random.randn(10, 10) * u.mV,
            brainstate.random.randn(10, 10) * u.mA,
        ]
        state = brainstate.HiddenTreeState(value)

        new_vals = [
            jnp.ones((10, 10)) * u.mV,
            jnp.zeros((10, 10)) * u.mA,
        ]
        state.set_value(new_vals)

        val0 = state.get_value(0)
        np.testing.assert_array_almost_equal(val0.mantissa, new_vals[0].mantissa)

    def test_hidden_tree_state_unit_preservation(self):
        """Test that units are preserved correctly."""
        value = {
            'v': np.ones((5, 5)) * u.mV,
            'i': np.ones((5, 5)) * u.mA,
        }
        state = brainstate.HiddenTreeState(value)

        v_val = state.get_value('v')
        i_val = state.get_value('i')

        self.assertEqual(v_val.unit, u.mV)
        self.assertEqual(i_val.unit, u.mA)

    def test_hidden_tree_state_dimensionless(self):
        """Test handling of dimensionless values."""
        value = [
            np.random.randn(10, 10),
            np.random.randn(10, 10),
        ]
        state = brainstate.HiddenTreeState(value)

        val = state.get_value(0)
        self.assertIsInstance(val, (np.ndarray, jax.Array))

    def test_hidden_tree_state_different_shapes_error(self):
        """Test that different shapes raise ValueError."""
        value = [
            np.random.randn(10, 10) * u.mV,
            np.random.randn(5, 5) * u.mA,  # Different shape
        ]

        with self.assertRaises(ValueError):
            brainstate.HiddenTreeState(value)

    def test_hidden_tree_state_invalid_type_error(self):
        """Test that invalid types raise TypeError."""
        value = {
            'v': [1, 2, 3],  # Python list not allowed
            'i': np.random.randn(10, 10) * u.mA,
        }

        with self.assertRaises(TypeError):
            brainstate.HiddenTreeState(value)

    def test_hidden_tree_state_name2unit_mapping(self):
        """Test name2unit mapping."""
        value = {
            'v': np.ones((5, 5)) * u.mV,
            'i': np.ones((5, 5)) * u.mA,
            'g': np.ones((5, 5)) * u.mS,
        }
        state = brainstate.HiddenTreeState(value)

        self.assertEqual(state.name2unit['v'], u.mV)
        self.assertEqual(state.name2unit['i'], u.mA)
        self.assertEqual(state.name2unit['g'], u.mS)

    def test_hidden_tree_state_index2unit_mapping(self):
        """Test index2unit mapping."""
        value = [
            np.ones((5, 5)) * u.mV,
            np.ones((5, 5)) * u.mA,
        ]
        state = brainstate.HiddenTreeState(value)

        self.assertEqual(state.index2unit[0], u.mV)
        self.assertEqual(state.index2unit[1], u.mA)


class TestFakeState(unittest.TestCase):
    """Test FakeState functionality."""

    def test_fake_state_creation(self):
        """Test creating a fake state."""
        state = brainstate.FakeState(42, name='fake')
        self.assertEqual(state.value, 42)
        self.assertEqual(state.name, 'fake')

    def test_fake_state_value_setter(self):
        """Test setting fake state value."""
        state = brainstate.FakeState(10)
        state.value = 20
        self.assertEqual(state.value, 20)

    def test_fake_state_name_setter(self):
        """Test setting fake state name."""
        state = brainstate.FakeState(10, name='old')
        state.name = 'new'
        self.assertEqual(state.name, 'new')

    def test_fake_state_repr(self):
        """Test fake state representation."""
        state = brainstate.FakeState([1, 2, 3])
        repr_str = repr(state)
        self.assertIn('FakedState', repr_str)


class TestStateDictManager(unittest.TestCase):
    """Test StateDictManager functionality."""

    def test_dict_manager_creation(self):
        """Test creating a StateDictManager."""
        manager = brainstate.StateDictManager()
        self.assertIsInstance(manager, brainstate.StateDictManager)

    def test_dict_manager_add_states(self):
        """Test adding states to manager."""
        manager = brainstate.StateDictManager()
        state1 = brainstate.State(jnp.zeros(5), name='state1')
        state2 = brainstate.State(jnp.ones(5), name='state2')

        manager['s1'] = state1
        manager['s2'] = state2

        self.assertEqual(len(manager), 2)
        self.assertIn('s1', manager)
        self.assertIn('s2', manager)

    def test_dict_manager_assign_values(self):
        """Test assigning values through manager."""
        manager = brainstate.StateDictManager()
        state = brainstate.State(jnp.zeros(3), name='test')
        manager['test'] = state

        new_values = {'test': jnp.array([1.0, 2.0, 3.0])}
        manager.assign_values(new_values)

        np.testing.assert_array_equal(state.value, new_values['test'])

    def test_dict_manager_collect_values(self):
        """Test collecting values from manager."""
        manager = brainstate.StateDictManager()
        state1 = brainstate.State(jnp.array([1.0, 2.0]), name='s1')
        state2 = brainstate.State(jnp.array([3.0, 4.0]), name='s2')

        manager['s1'] = state1
        manager['s2'] = state2

        values = manager.collect_values()
        self.assertEqual(len(values), 2)
        np.testing.assert_array_equal(values['s1'], state1.value)
        np.testing.assert_array_equal(values['s2'], state2.value)

    def test_dict_manager_split_values(self):
        """Test splitting values by type."""
        manager = brainstate.StateDictManager()
        short_state = brainstate.ShortTermState(jnp.zeros(3))
        long_state = brainstate.LongTermState(jnp.ones(3))

        manager['short'] = short_state
        manager['long'] = long_state

        short_vals, other_vals = manager.split_values(brainstate.ShortTermState)
        self.assertEqual(len(short_vals), 1)

    def test_dict_manager_to_dict_values(self):
        """Test converting to dict of values."""
        manager = brainstate.StateDictManager()
        state1 = brainstate.State(jnp.array([1.0]), name='s1')
        state2 = brainstate.State(jnp.array([2.0]), name='s2')

        manager['s1'] = state1
        manager['s2'] = state2

        dict_vals = manager.to_dict_values()
        self.assertIsInstance(dict_vals, dict)
        self.assertEqual(len(dict_vals), 2)


class TestStateTraceStack(unittest.TestCase):
    """Test StateTraceStack functionality."""

    def test_trace_stack_creation(self):
        """Test creating a StateTraceStack."""
        stack = brainstate.StateTraceStack(name='test_stack')
        self.assertEqual(stack.name, 'test_stack')
        self.assertEqual(len(stack.states), 0)

    def test_trace_stack_read_value(self):
        """Test recording state reads."""
        stack = brainstate.StateTraceStack()
        state = brainstate.State(jnp.zeros(3))

        with stack:
            _ = state.value

        self.assertEqual(len(stack.states), 1)
        self.assertIn(state, stack.states)
        self.assertFalse(stack.been_writen[0])

    def test_trace_stack_write_value(self):
        """Test recording state writes."""
        stack = brainstate.StateTraceStack()
        state = brainstate.State(jnp.zeros(3))

        with stack:
            state.value = jnp.ones(3)

        self.assertEqual(len(stack.states), 1)
        self.assertTrue(stack.been_writen[0])

    def test_trace_stack_get_state_values(self):
        """Test getting state values from stack."""
        stack = brainstate.StateTraceStack()
        state1 = brainstate.State(jnp.array([1.0, 2.0]))
        state2 = brainstate.State(jnp.array([3.0, 4.0]))

        with stack:
            _ = state1.value
            state2.value = jnp.array([5.0, 6.0])

        values = stack.get_state_values()
        self.assertEqual(len(values), 2)

    def test_trace_stack_get_read_states(self):
        """Test getting read-only states."""
        stack = brainstate.StateTraceStack()
        read_state = brainstate.State(jnp.zeros(3))
        write_state = brainstate.State(jnp.ones(3))

        with stack:
            _ = read_state.value
            write_state.value = jnp.array([1.0, 2.0, 3.0])

        read_states = stack.get_read_states()
        self.assertEqual(len(read_states), 1)
        self.assertIn(read_state, read_states)
        self.assertNotIn(write_state, read_states)

    def test_trace_stack_get_write_states(self):
        """Test getting written states."""
        stack = brainstate.StateTraceStack()
        read_state = brainstate.State(jnp.zeros(3))
        write_state = brainstate.State(jnp.ones(3))

        with stack:
            _ = read_state.value
            write_state.value = jnp.array([1.0, 2.0, 3.0])

        write_states = stack.get_write_states()
        self.assertEqual(len(write_states), 1)
        self.assertIn(write_state, write_states)
        self.assertNotIn(read_state, write_states)

    def test_trace_stack_recovery_original_values(self):
        """Test recovering original values."""
        stack = brainstate.StateTraceStack()
        state = brainstate.State(jnp.zeros(3))
        original = state.value.copy()

        with stack:
            state.value = jnp.ones(3)

        stack.recovery_original_values()
        np.testing.assert_array_equal(state.value, original)

    def test_trace_stack_merge(self):
        """Test merging trace stacks."""
        stack1 = brainstate.StateTraceStack()
        stack2 = brainstate.StateTraceStack()

        state1 = brainstate.State(jnp.zeros(3))
        state2 = brainstate.State(jnp.ones(3))

        with stack1:
            _ = state1.value

        with stack2:
            state2.value = jnp.array([1.0, 2.0, 3.0])

        merged = stack1.merge(stack2)
        self.assertEqual(len(merged.states), 2)

    def test_trace_stack_add_operator(self):
        """Test using + operator to merge stacks."""
        stack1 = brainstate.StateTraceStack()
        stack2 = brainstate.StateTraceStack()

        state1 = brainstate.State(jnp.zeros(3))
        state2 = brainstate.State(jnp.ones(3))

        with stack1:
            _ = state1.value

        with stack2:
            _ = state2.value

        merged = stack1 + stack2
        self.assertEqual(len(merged.states), 2)

    def test_trace_stack_state_subset(self):
        """Test getting state subset by type."""
        stack = brainstate.StateTraceStack()
        short_state = brainstate.ShortTermState(jnp.zeros(3))
        long_state = brainstate.LongTermState(jnp.ones(3))

        with stack:
            _ = short_state.value
            _ = long_state.value

        short_subset = stack.state_subset(brainstate.ShortTermState)
        self.assertEqual(len(short_subset), 1)
        self.assertIn(short_state, short_subset)

    def test_trace_stack_assign_state_vals(self):
        """Test assigning state values."""
        stack = brainstate.StateTraceStack()
        state1 = brainstate.State(jnp.zeros(3))
        state2 = brainstate.State(jnp.zeros(3))

        with stack:
            _ = state1.value
            state2.value = jnp.ones(3)

        new_vals = [jnp.array([1.0, 1.0, 1.0]), jnp.array([2.0, 2.0, 2.0])]
        stack.assign_state_vals(new_vals)

        np.testing.assert_array_equal(state2.value, new_vals[1])


class TestTreefyState(unittest.TestCase):
    """Test TreefyState functionality."""

    def test_treefy_state_creation(self):
        """Test creating a TreefyState."""
        ref = brainstate.TreefyState(brainstate.State, jnp.array([1.0, 2.0]), _name='test')
        self.assertEqual(ref.name, 'test')
        np.testing.assert_array_equal(ref.value, jnp.array([1.0, 2.0]))

    def test_treefy_state_replace(self):
        """Test replacing TreefyState value."""
        ref = brainstate.TreefyState(brainstate.State, jnp.zeros(3), _name='test')
        new_ref = ref.replace(jnp.ones(3))

        np.testing.assert_array_equal(new_ref.value, jnp.ones(3))
        self.assertEqual(new_ref.name, 'test')

    def test_treefy_state_to_state(self):
        """Test converting TreefyState to State."""
        ref = brainstate.State(jnp.array([1.0, 2.0]), name='test').to_state_ref()
        state = ref.to_state()

        self.assertIsInstance(state, brainstate.State)
        np.testing.assert_array_equal(state.value, ref.value)
        self.assertEqual(state.name, 'test')

    def test_treefy_state_copy(self):
        """Test copying TreefyState."""
        ref = brainstate.TreefyState(brainstate.State, jnp.array([1.0, 2.0]), _name='test')
        ref_copy = ref.copy()

        self.assertIsNot(ref, ref_copy)
        np.testing.assert_array_equal(ref_copy.value, ref.value)

    def test_treefy_state_get_metadata(self):
        """Test getting metadata from TreefyState."""
        ref = brainstate.TreefyState(brainstate.State, jnp.zeros(3), _name='test', _been_writen=True)
        metadata = ref.get_metadata()

        self.assertIsInstance(metadata, dict)
        self.assertEqual(metadata['_name'], 'test')
        self.assertTrue(metadata['_been_writen'])
        self.assertNotIn('type', metadata)
        self.assertNotIn('value', metadata)

    def test_treefy_state_pytree_operations(self):
        """Test that TreefyState works as a pytree."""
        ref = brainstate.TreefyState(brainstate.State, jnp.array([1.0, 2.0]), _name='test')

        # Test tree_map
        mapped = jax.tree.map(lambda x: x * 2, ref)
        np.testing.assert_array_equal(mapped.value, jnp.array([2.0, 4.0]))

        # Test tree_leaves
        leaves = jax.tree.leaves(ref)
        self.assertEqual(len(leaves), 1)


class TestContextManagers(unittest.TestCase):
    """Test context managers and utility functions."""

    def test_check_state_value_tree(self):
        """Test check_state_value_tree context manager."""
        state = brainstate.State(jnp.zeros((2, 3)))

        # Should not raise error
        with brainstate.check_state_value_tree():
            state.value = jnp.ones((2, 3))

        # Should raise error on tree structure change
        with brainstate.check_state_value_tree():
            with self.assertRaises(ValueError):
                state.value = {'a': jnp.zeros((2, 3))}

    def test_check_state_value_tree_nested(self):
        """Test nested check_state_value_tree contexts."""
        state = brainstate.State(jnp.zeros(3))

        with brainstate.check_state_value_tree(True):
            with brainstate.check_state_value_tree(False):
                # Inner context disables checking
                state.value = {'a': jnp.zeros(3)}  # Should not raise

    def test_maybe_state(self):
        """Test maybe_state utility function."""
        state = brainstate.State(jnp.array([1.0, 2.0, 3.0]))

        # Should extract value from State
        result = brainstate.maybe_state(state)
        np.testing.assert_array_equal(result, jnp.array([1.0, 2.0, 3.0]))

        # Should return non-State values as-is
        value = jnp.array([4.0, 5.0, 6.0])
        result = brainstate.maybe_state(value)
        np.testing.assert_array_equal(result, value)

    def test_catch_new_states(self):
        """Test catch_new_states context manager."""
        with brainstate.catch_new_states('test_tag') as catcher:
            state1 = brainstate.State(jnp.zeros(3))
            state2 = brainstate.State(jnp.ones(3))

        self.assertEqual(len(catcher), 2)
        self.assertIn('test_tag', state1.tag)
        self.assertIn('test_tag', state2.tag)

    def test_catch_new_states_get_states(self):
        """Test getting caught states."""
        with brainstate.catch_new_states() as catcher:
            state = brainstate.State(jnp.zeros(3))

        states = catcher.get_states()
        self.assertEqual(len(states), 1)
        self.assertIn(state, states)

    def test_catch_new_states_get_state_values(self):
        """Test getting caught state values."""
        with brainstate.catch_new_states() as catcher:
            state = brainstate.State(jnp.array([1.0, 2.0]))

        values = catcher.get_state_values()
        self.assertEqual(len(values), 1)
        np.testing.assert_array_equal(values[0], jnp.array([1.0, 2.0]))


class TestStateCatcher(unittest.TestCase):
    """Test NewStateCatcher functionality via catch_new_states context manager."""

    def test_catch_new_states_basic(self):
        """Test basic catch_new_states functionality."""
        with brainstate.catch_new_states('test') as catcher:
            state = brainstate.State(jnp.zeros(3))

        self.assertEqual(len(catcher), 1)
        self.assertIn('test', state.tag)

    def test_catch_multiple_states(self):
        """Test catching multiple states."""
        with brainstate.catch_new_states('test') as catcher:
            state1 = brainstate.State(jnp.zeros(3))
            state2 = brainstate.State(jnp.ones(3))
            state3 = brainstate.State(jnp.array([1.0, 2.0]))

        self.assertEqual(len(catcher), 3)
        self.assertIn(state1, catcher.get_states())
        self.assertIn(state2, catcher.get_states())
        self.assertIn(state3, catcher.get_states())

    def test_catcher_iteration(self):
        """Test iterating over caught states."""
        with brainstate.catch_new_states('test') as catcher:
            states = [brainstate.State(jnp.zeros(i + 1)) for i in range(3)]

        collected = list(catcher)
        self.assertEqual(len(collected), 3)

    def test_catcher_indexing(self):
        """Test indexing into catcher."""
        with brainstate.catch_new_states('test') as catcher:
            state = brainstate.State(jnp.zeros(3))

        self.assertIs(catcher[0], state)

    def test_catcher_contains(self):
        """Test checking if state is in catcher."""
        with brainstate.catch_new_states('test') as catcher:
            state = brainstate.State(jnp.zeros(3))

        self.assertIn(state, catcher)

    def test_catcher_get_states(self):
        """Test getting list of caught states."""
        with brainstate.catch_new_states('test') as catcher:
            state1 = brainstate.State(jnp.zeros(3))
            state2 = brainstate.State(jnp.ones(3))

        states = catcher.get_states()
        self.assertEqual(len(states), 2)
        self.assertIn(state1, states)
        self.assertIn(state2, states)

    def test_catcher_get_state_values(self):
        """Test getting values of caught states."""
        with brainstate.catch_new_states('test') as catcher:
            state = brainstate.State(jnp.array([1.0, 2.0, 3.0]))

        values = catcher.get_state_values()
        self.assertEqual(len(values), 1)
        np.testing.assert_array_equal(values[0], jnp.array([1.0, 2.0, 3.0]))

    def test_nested_catch_contexts(self):
        """Test nested catch_new_states contexts."""
        with brainstate.catch_new_states('outer') as outer_catcher:
            outer_state = brainstate.State(jnp.zeros(3))

            with brainstate.catch_new_states('inner') as inner_catcher:
                inner_state = brainstate.State(jnp.ones(3))

            # Inner catcher should have only inner state
            self.assertEqual(len(inner_catcher), 1)
            self.assertIn(inner_state, inner_catcher)

            # Outer catcher should have both states
            self.assertEqual(len(outer_catcher), 2)
            self.assertIn(outer_state, outer_catcher)
            self.assertIn(inner_state, outer_catcher)


class TestStateHooks(unittest.TestCase):
    """Comprehensive tests for State hook system."""

    def setUp(self):
        """Set up test fixtures."""
        self.call_log = []
        # Clear global hooks before each test
        brainstate.clear_state_hooks()

    def tearDown(self):
        """Clean up after tests."""
        brainstate.clear_state_hooks()

    # =============================================================================
    # Init Hook Tests
    # =============================================================================

    def test_init_hook_execution(self):
        """Test that init hooks execute during initialization."""
        init_data = {}

        def init_hook(ctx):
            init_data['value'] = ctx.value
            init_data['operation'] = ctx.operation
            init_data['metadata'] = ctx.init_metadata

        # Register global init hook before creating state
        brainstate.register_state_hook('init', init_hook)

        state = brainstate.State(jnp.array([1, 2, 3]), name='test_state')

        # Init hook should have been called
        self.assertEqual(init_data['operation'], 'init')
        np.testing.assert_array_equal(init_data['value'], jnp.array([1, 2, 3]))
        self.assertEqual(init_data['metadata']['_name'], 'test_state')

    def test_init_hook_with_metadata(self):
        """Test init hooks receive initialization metadata."""
        captured_metadata = {}

        def capture_metadata(ctx):
            captured_metadata.update(ctx.init_metadata)

        brainstate.register_state_hook('init', capture_metadata)

        state = brainstate.State(
            jnp.array([1, 2]),
            name='my_state',
            tag='network'
        )

        self.assertEqual(captured_metadata['_name'], 'my_state')
        self.assertIn('network', captured_metadata['tag'])

    def test_init_hook_priority_ordering(self):
        """Test init hooks execute in priority order."""

        def hook_high(ctx):
            self.call_log.append('high')

        def hook_medium(ctx):
            self.call_log.append('medium')

        def hook_low(ctx):
            self.call_log.append('low')

        brainstate.register_state_hook('init', hook_high, priority=100)
        brainstate.register_state_hook('init', hook_medium, priority=50)
        brainstate.register_state_hook('init', hook_low, priority=10)

        state = brainstate.State(jnp.zeros(3))

        # Higher priority should execute first
        self.assertEqual(self.call_log, ['high', 'medium', 'low'])

    def test_init_hook_multiple_states(self):
        """Test init hooks are called for each state creation."""
        call_count = {'count': 0}

        def count_init(ctx):
            call_count['count'] += 1

        brainstate.register_state_hook('init', count_init)

        state1 = brainstate.State(jnp.zeros(3))
        state2 = brainstate.State(jnp.ones(5))
        state3 = brainstate.State(jnp.array([1, 2]))

        self.assertEqual(call_count['count'], 3)

    def test_init_hook_cannot_modify_value(self):
        """Test that init hooks cannot modify the initial value."""

        def try_modify(ctx):
            # Init hooks receive read-only context
            # transformed_value should not exist
            self.assertFalse(hasattr(ctx, 'transformed_value'))

        brainstate.register_state_hook('init', try_modify)

        state = brainstate.State(jnp.array([1, 2, 3]))
        np.testing.assert_array_equal(state.value, jnp.array([1, 2, 3]))

    # =============================================================================
    # Read Hook Tests
    # =============================================================================

    def test_read_hook_basic(self):
        """Test basic read hook functionality."""
        state = brainstate.State(jnp.array([1, 2, 3]))

        def read_hook(ctx):
            self.call_log.append(f"read: {ctx.value.tolist()}")

        state.register_hook('read', read_hook)

        _ = state.value
        self.assertEqual(len(self.call_log), 1)
        self.assertEqual(self.call_log[0], "read: [1, 2, 3]")

    def test_read_hook_multiple_accesses(self):
        """Test read hooks called on each access."""
        state = brainstate.State(jnp.array([1, 2]))

        def count_reads(ctx):
            self.call_log.append('read')

        state.register_hook('read', count_reads)

        for _ in range(5):
            _ = state.value

        self.assertEqual(len(self.call_log), 5)

    def test_read_hook_with_state_reference(self):
        """Test read hooks can access state reference."""
        state = brainstate.State(jnp.array([1, 2, 3]), name='my_state')

        def check_state_ref(ctx):
            self.assertIsNotNone(ctx.state)
            self.assertEqual(ctx.state_name, 'my_state')
            self.assertIs(ctx.state, state)

        state.register_hook('read', check_state_ref)
        _ = state.value

    def test_read_hook_priority_ordering(self):
        """Test read hooks execute in priority order."""
        state = brainstate.State(jnp.zeros(3))

        state.register_hook('read', lambda ctx: self.call_log.append('low'), priority=1)
        state.register_hook('read', lambda ctx: self.call_log.append('high'), priority=100)
        state.register_hook('read', lambda ctx: self.call_log.append('medium'), priority=50)

        _ = state.value

        self.assertEqual(self.call_log, ['high', 'medium', 'low'])

    # =============================================================================
    # Write Before Hook Tests
    # =============================================================================

    def test_write_before_transformation(self):
        """Test write_before hooks can transform values."""
        state = brainstate.State(jnp.array([0, 0, 0]))

        def clip_values(ctx):
            input_val = ctx.transformed_value if ctx.transformed_value is not None else ctx.value
            ctx.transformed_value = jnp.clip(input_val, -1.0, 1.0)

        state.register_hook('write_before', clip_values)

        state.value = jnp.array([5, -5, 0.5])
        np.testing.assert_array_almost_equal(state.value, jnp.array([1, -1, 0.5]))

    def test_write_before_cancellation(self):
        """Test write_before hooks can cancel operations."""
        state = brainstate.State(jnp.array([1, 2, 3]))

        def validate_positive(ctx):
            if jnp.any(ctx.value < 0):
                ctx.cancel = True
                ctx.cancel_reason = "Values must be non-negative"

        state.register_hook('write_before', validate_positive)

        # Valid write succeeds
        state.value = jnp.array([1, 2, 3])
        np.testing.assert_array_equal(state.value, jnp.array([1, 2, 3]))

        # Invalid write fails
        with self.assertRaises(brainstate.HookCancellationError):
            state.value = jnp.array([1, -1, 3])

    def test_write_before_sequential_chaining(self):
        """Test write_before hooks chain transformations."""
        state = brainstate.State(jnp.array([0.0, 0.0]))

        def multiply_by_2(ctx):
            input_val = ctx.transformed_value if ctx.transformed_value is not None else ctx.value
            ctx.transformed_value = input_val * 2

        def add_10(ctx):
            input_val = ctx.transformed_value if ctx.transformed_value is not None else ctx.value
            ctx.transformed_value = input_val + 10

        state.register_hook('write_before', multiply_by_2, priority=10)
        state.register_hook('write_before', add_10, priority=5)

        state.value = jnp.array([1.0, 2.0])
        # (1 * 2) + 10 = 12, (2 * 2) + 10 = 14
        np.testing.assert_array_almost_equal(state.value, jnp.array([12.0, 14.0]))

    def test_write_before_complex_transformation(self):
        """Test complex transformation pipeline."""
        state = brainstate.State(jnp.array([1.0, 2.0, 3.0]))

        def normalize(ctx):
            input_val = ctx.transformed_value if ctx.transformed_value is not None else ctx.value
            norm = jnp.linalg.norm(input_val)
            if norm > 0:
                ctx.transformed_value = input_val / norm

        def scale(ctx):
            input_val = ctx.transformed_value if ctx.transformed_value is not None else ctx.value
            ctx.transformed_value = input_val * 10.0

        state.register_hook('write_before', normalize, priority=10)
        state.register_hook('write_before', scale, priority=5)

        state.value = jnp.array([3.0, 4.0, 0.0])
        # Normalize: [3, 4, 0] / 5 = [0.6, 0.8, 0]
        # Scale: [0.6, 0.8, 0] * 10 = [6, 8, 0]
        expected = jnp.array([6.0, 8.0, 0.0])
        np.testing.assert_array_almost_equal(state.value, expected)

    def test_write_before_validation_with_custom_error(self):
        """Test validation with custom error messages."""
        state = brainstate.State(jnp.array([1.0, 2.0]))

        def validate_range(ctx):
            if jnp.any(ctx.value > 100) or jnp.any(ctx.value < -100):
                ctx.cancel = True
                ctx.cancel_reason = "Values must be in range [-100, 100]"

        state.register_hook('write_before', validate_range)

        with self.assertRaises(brainstate.HookCancellationError) as cm:
            state.value = jnp.array([150.0, 200.0])

        self.assertIn("[-100, 100]", str(cm.exception))

    # =============================================================================
    # Write After Hook Tests
    # =============================================================================

    def test_write_after_notification(self):
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

    def test_write_after_multiple_writes(self):
        """Test write_after hooks track multiple writes."""
        state = brainstate.State(jnp.zeros(2))

        state.register_hook('write_after', lambda ctx: self.call_log.append(ctx.value.tolist()))

        state.value = jnp.array([1.0, 2.0])
        state.value = jnp.array([3.0, 4.0])
        state.value = jnp.array([5.0, 6.0])

        self.assertEqual(len(self.call_log), 3)
        self.assertEqual(self.call_log[0], [1.0, 2.0])
        self.assertEqual(self.call_log[1], [3.0, 4.0])
        self.assertEqual(self.call_log[2], [5.0, 6.0])

    def test_write_after_with_transformation(self):
        """Test write_after hooks see transformed values."""
        state = brainstate.State(jnp.array([0.0, 0.0]))

        # Transformation hook
        def double_value(ctx):
            input_val = ctx.transformed_value if ctx.transformed_value is not None else ctx.value
            ctx.transformed_value = input_val * 2

        # Notification hook
        def check_transformed(ctx):
            self.call_log.append(ctx.value.tolist())

        state.register_hook('write_before', double_value)
        state.register_hook('write_after', check_transformed)

        state.value = jnp.array([1.0, 2.0])

        # write_after should see [2.0, 4.0] (after transformation)
        self.assertEqual(self.call_log[0], [2.0, 4.0])

    # =============================================================================
    # Restore Hook Tests
    # =============================================================================

    def test_restore_hook_basic(self):
        """Test restore hooks execute on restore_value."""
        state = brainstate.State(jnp.array([1, 2, 3]))
        state.value = jnp.array([4, 5, 6])

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

    def test_restore_hook_multiple_restores(self):
        """Test restore hooks track multiple restorations."""
        state = brainstate.State(jnp.zeros(2))

        state.register_hook('restore', lambda ctx: self.call_log.append('restore'))

        state.restore_value(jnp.array([1.0, 2.0]))
        state.restore_value(jnp.array([3.0, 4.0]))

        self.assertEqual(self.call_log, ['restore', 'restore'])

    # =============================================================================
    # Hook Priority and Ordering Tests
    # =============================================================================

    def test_hook_priority_default_zero(self):
        """Test hooks with default priority execute in registration order."""
        state = brainstate.State(jnp.zeros(3))

        state.register_hook('read', lambda ctx: self.call_log.append('first'))
        state.register_hook('read', lambda ctx: self.call_log.append('second'))
        state.register_hook('read', lambda ctx: self.call_log.append('third'))

        _ = state.value

        self.assertEqual(self.call_log, ['first', 'second', 'third'])

    def test_hook_priority_descending_order(self):
        """Test hooks execute in descending priority order."""
        state = brainstate.State(jnp.zeros(3))

        state.register_hook('read', lambda ctx: self.call_log.append('p100'), priority=100)
        state.register_hook('read', lambda ctx: self.call_log.append('p50'), priority=50)
        state.register_hook('read', lambda ctx: self.call_log.append('p1'), priority=1)
        state.register_hook('read', lambda ctx: self.call_log.append('p0'), priority=0)
        state.register_hook('read', lambda ctx: self.call_log.append('p-10'), priority=-10)

        _ = state.value

        self.assertEqual(self.call_log, ['p100', 'p50', 'p1', 'p0', 'p-10'])

    def test_hook_priority_mixed_types(self):
        """Test priority ordering works across hook types."""
        state = brainstate.State(jnp.zeros(2))

        state.register_hook('write_before', lambda ctx: self.call_log.append('wb_high'), priority=100)
        state.register_hook('write_before', lambda ctx: self.call_log.append('wb_low'), priority=1)

        state.value = jnp.ones(2)

        self.assertEqual(self.call_log, ['wb_high', 'wb_low'])

    # =============================================================================
    # Global Hook Tests
    # =============================================================================

    def test_global_hook_applies_to_all_states(self):
        """Test global hooks apply to all state instances."""
        call_count = {'count': 0}

        def global_counter(ctx):
            call_count['count'] += 1

        brainstate.register_state_hook('read', global_counter)

        state1 = brainstate.State(jnp.array([1, 2]))
        state2 = brainstate.State(jnp.array([3, 4]))
        state3 = brainstate.State(jnp.array([5, 6]))

        _ = state1.value
        _ = state2.value
        _ = state3.value

        self.assertEqual(call_count['count'], 3)

    def test_global_hook_executes_before_instance(self):
        """Test global hooks execute before instance hooks."""

        def global_hook(ctx):
            self.call_log.append('global')

        def instance_hook(ctx):
            self.call_log.append('instance')

        brainstate.register_state_hook('read', global_hook)

        state = brainstate.State(jnp.array([1, 2]))
        state.register_hook('read', instance_hook)

        _ = state.value

        self.assertEqual(self.call_log, ['global', 'instance'])

    def test_global_hook_transformation_chains_with_instance(self):
        """Test global and instance hooks chain transformations."""
        state = brainstate.State(jnp.zeros(2))

        def global_transform(ctx):
            input_val = ctx.transformed_value if ctx.transformed_value is not None else ctx.value
            ctx.transformed_value = input_val + 1

        def instance_transform(ctx):
            input_val = ctx.transformed_value if ctx.transformed_value is not None else ctx.value
            ctx.transformed_value = input_val * 2

        brainstate.register_state_hook('write_before', global_transform, priority=100)
        state.register_hook('write_before', instance_transform, priority=50)

        state.value = jnp.array([0.0, 1.0])
        # Global: [0, 1] + 1 = [1, 2]
        # Instance: [1, 2] * 2 = [2, 4]
        np.testing.assert_array_almost_equal(state.value, jnp.array([2.0, 4.0]))

    def test_clear_global_state_hooks(self):
        """Test clearing global hooks."""
        call_count = {'count': 0}

        def counter(ctx):
            call_count['count'] += 1

        brainstate.register_state_hook('read', counter)

        state = brainstate.State(jnp.zeros(3))
        _ = state.value
        self.assertEqual(call_count['count'], 1)

        brainstate.clear_state_hooks()

        state2 = brainstate.State(jnp.ones(3))
        _ = state2.value
        # Count should not increase after clearing
        self.assertEqual(call_count['count'], 1)

    def test_clear_global_state_hooks_by_type(self):
        """Test clearing global hooks by specific type."""
        read_count = {'count': 0}
        write_count = {'count': 0}

        brainstate.register_state_hook('read', lambda ctx: read_count.update(count=read_count['count'] + 1))
        brainstate.register_state_hook('write_after',
                                       lambda ctx: write_count.update(count=write_count['count'] + 1))

        state = brainstate.State(jnp.zeros(3))
        _ = state.value
        state.value = jnp.ones(3)

        self.assertEqual(read_count['count'], 1)
        self.assertEqual(write_count['count'], 1)

        # Clear only read hooks
        brainstate.clear_state_hooks('read')

        _ = state.value
        state.value = jnp.zeros(3)

        # Read count should not increase
        self.assertEqual(read_count['count'], 1)
        # Write count should increase
        self.assertEqual(write_count['count'], 2)

    # =============================================================================
    # Hook Handle Tests
    # =============================================================================

    def test_hook_handle_disable_enable(self):
        """Test disabling and enabling hooks via handle."""
        state = brainstate.State(jnp.zeros(3))

        handle = state.register_hook('read', lambda ctx: self.call_log.append('read'))

        _ = state.value
        self.assertEqual(len(self.call_log), 1)

        # Disable hook
        handle.disable()
        _ = state.value
        # Should not call hook
        self.assertEqual(len(self.call_log), 1)

        # Re-enable hook
        handle.enable()
        _ = state.value
        # Should call hook again
        self.assertEqual(len(self.call_log), 2)

    def test_hook_handle_remove(self):
        """Test removing hooks via handle."""
        state = brainstate.State(jnp.zeros(3))

        handle = state.register_hook('read', lambda ctx: self.call_log.append('read'))

        _ = state.value
        self.assertEqual(len(self.call_log), 1)

        # Remove hook
        handle.remove()
        _ = state.value
        # Should not call hook
        self.assertEqual(len(self.call_log), 1)

    def test_hook_handle_is_enabled(self):
        """Test checking hook enabled status."""
        state = brainstate.State(jnp.zeros(3))

        handle = state.register_hook('read', lambda ctx: None)

        self.assertTrue(handle.is_enabled())

        handle.disable()
        self.assertFalse(handle.is_enabled())

        handle.enable()
        self.assertTrue(handle.is_enabled())

    def test_hook_handle_multiple_operations(self):
        """Test multiple operations on hook handle."""
        state = brainstate.State(jnp.zeros(3))

        handle = state.register_hook('read', lambda ctx: self.call_log.append('read'))

        # Disable, enable, disable, enable cycle
        for _ in range(3):
            handle.disable()
            _ = state.value
            handle.enable()
            _ = state.value

        # Should be called 3 times (when enabled)
        self.assertEqual(len(self.call_log), 3)

    # =============================================================================
    # Hook Context Manager Tests
    # =============================================================================

    def test_temporary_hook_context_manager(self):
        """Test temporary_hook context manager."""
        state = brainstate.State(jnp.zeros(3))

        def temp_hook(ctx):
            self.call_log.append('temp')

        # Hook active only within context
        with state.temporary_hook('read', temp_hook):
            _ = state.value
            self.assertEqual(len(self.call_log), 1)

        # Hook removed after context
        _ = state.value
        self.assertEqual(len(self.call_log), 1)

    def test_temporary_hook_nested_contexts(self):
        """Test nested temporary_hook contexts."""
        state = brainstate.State(jnp.zeros(3))

        with state.temporary_hook('read', lambda ctx: self.call_log.append('outer')):
            _ = state.value
            self.assertEqual(self.call_log, ['outer'])

            with state.temporary_hook('read', lambda ctx: self.call_log.append('inner')):
                self.call_log.clear()
                _ = state.value
                # Both hooks should be active
                self.assertIn('outer', self.call_log)
                self.assertIn('inner', self.call_log)

            # Only outer hook active
            self.call_log.clear()
            _ = state.value
            self.assertEqual(self.call_log, ['outer'])

    def test_temporary_hook_exception_cleanup(self):
        """Test temporary hooks are removed even on exception."""
        state = brainstate.State(jnp.zeros(3))

        try:
            with state.temporary_hook('read', lambda ctx: self.call_log.append('temp')):
                _ = state.value
                self.assertEqual(len(self.call_log), 1)
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Hook should be removed despite exception
        _ = state.value
        self.assertEqual(len(self.call_log), 1)

    # =============================================================================
    # Hook List and Clear Tests
    # =============================================================================

    def test_list_hooks_all_types(self):
        """Test listing hooks across all types."""
        state = brainstate.State(jnp.zeros(3))

        state.register_hook('read', lambda ctx: None, name='read1')
        state.register_hook('write_before', lambda ctx: None, name='wb1')
        state.register_hook('write_after', lambda ctx: None, name='wa1')
        state.register_hook('restore', lambda ctx: None, name='restore1')

        all_hooks = state.list_hooks()
        self.assertEqual(len(all_hooks), 4)

    def test_list_hooks_by_type(self):
        """Test listing hooks filtered by type."""
        state = brainstate.State(jnp.zeros(3))

        state.register_hook('read', lambda ctx: None, name='read1')
        state.register_hook('read', lambda ctx: None, name='read2')
        state.register_hook('write_before', lambda ctx: None, name='wb1')

        read_hooks = state.list_hooks('read')
        self.assertEqual(len(read_hooks), 2)

        write_hooks = state.list_hooks('write_before')
        self.assertEqual(len(write_hooks), 1)

    def test_clear_hooks_all_types(self):
        """Test clearing all hooks."""
        state = brainstate.State(jnp.zeros(3))

        state.register_hook('read', lambda ctx: None)
        state.register_hook('write_before', lambda ctx: None)
        state.register_hook('write_after', lambda ctx: None)

        state.clear_hooks()

        self.assertEqual(len(state.list_hooks()), 0)

    def test_clear_hooks_by_type(self):
        """Test clearing hooks by specific type."""
        state = brainstate.State(jnp.zeros(3))

        state.register_hook('read', lambda ctx: None, name='read1')
        state.register_hook('write_before', lambda ctx: None, name='wb1')

        state.clear_hooks('read')

        self.assertEqual(len(state.list_hooks('read')), 0)
        self.assertEqual(len(state.list_hooks('write_before')), 1)

    def test_has_hooks_check(self):
        """Test has_hooks method."""
        state = brainstate.State(jnp.zeros(3))

        self.assertFalse(state.has_hooks())

        handle = state.register_hook('read', lambda ctx: None)

        self.assertTrue(state.has_hooks())
        self.assertTrue(state.has_hooks('read'))
        self.assertFalse(state.has_hooks('write_before'))

        handle.remove()

        self.assertFalse(state.has_hooks())

    # =============================================================================
    # Error Handling Tests
    # =============================================================================

    def test_hook_error_handling_log_mode(self):
        """Test hook errors are logged in log mode (default)."""
        state = brainstate.State(jnp.zeros(3))

        def failing_hook(ctx):
            raise RuntimeError("Hook error")

        state.register_hook('read', failing_hook)

        # Should not raise, just log warning
        with self.assertWarns(brainstate.HookWarning):
            _ = state.value

    def test_hook_error_handling_raise_mode(self):
        """Test hook errors are raised in raise mode."""

        config = HookConfig(on_error='raise')
        manager = HookManager(config)

        # Create state with custom hook manager (need to use internal API for testing)
        state = brainstate.State(jnp.zeros(3))
        state._hooks_manager = manager

        def failing_hook(ctx):
            raise RuntimeError("Hook error")

        state.register_hook('read', failing_hook)

        with self.assertRaises(brainstate.HookExecutionError):
            _ = state.value

    def test_hook_error_handling_ignore_mode(self):
        """Test hook errors are ignored in ignore mode."""

        config = HookConfig(on_error='ignore')
        manager = HookManager(config)

        state = brainstate.State(jnp.zeros(3))
        state._hooks_manager = manager

        def failing_hook(ctx):
            raise RuntimeError("Hook error")

        state.register_hook('read', failing_hook)

        # Should silently ignore error
        _ = state.value  # No exception or warning

    # =============================================================================
    # Complex Integration Tests
    # =============================================================================

    def test_hook_integration_logging_system(self):
        """Test implementing a logging system with hooks."""
        log = []

        def log_init(ctx):
            log.append(f"INIT: {ctx.state_name} = {ctx.value.tolist()}")

        def log_read(ctx):
            log.append(f"READ: {ctx.state_name}")

        def log_write(ctx):
            log.append(f"WRITE: {ctx.state_name} from {ctx.old_value.tolist()} to {ctx.value.tolist()}")

        brainstate.register_state_hook('init', log_init)
        brainstate.register_state_hook('read', log_read)
        brainstate.register_state_hook('write_after', log_write)

        state = brainstate.State(jnp.array([1, 2]), name='counter')
        _ = state.value
        state.value = jnp.array([3, 4])

        self.assertEqual(len(log), 3)
        self.assertIn("INIT: counter", log[0])
        self.assertIn("READ: counter", log[1])
        self.assertIn("WRITE: counter", log[2])

    def test_hook_integration_value_constraints(self):
        """Test enforcing value constraints with hooks."""
        state = brainstate.State(jnp.array([0.5, 0.5]))

        def enforce_probability(ctx):
            # Ensure values are valid probabilities
            input_val = ctx.transformed_value if ctx.transformed_value is not None else ctx.value
            ctx.transformed_value = jnp.clip(input_val, 0.0, 1.0)

        def validate_sum(ctx):
            # Cancel if sum is too far from 1
            if abs(jnp.sum(ctx.value) - 1.0) > 0.1:
                ctx.cancel = True
                ctx.cancel_reason = "Probabilities must sum to ~1"

        state.register_hook('write_before', enforce_probability, priority=100)
        state.register_hook('write_before', validate_sum, priority=50)

        # Valid update
        state.value = jnp.array([0.6, 0.4])
        np.testing.assert_array_almost_equal(state.value, jnp.array([0.6, 0.4]))

        # Invalid sum should fail
        with self.assertRaises(brainstate.HookCancellationError):
            state.value = jnp.array([0.1, 0.1])

    def test_hook_integration_checkpointing(self):
        """Test implementing checkpointing with hooks."""
        checkpoints = {}

        def checkpoint_write(ctx):
            state_id = id(ctx.state)
            if state_id not in checkpoints:
                checkpoints[state_id] = []
            checkpoints[state_id].append(ctx.value.copy())

        state = brainstate.State(jnp.array([1.0, 2.0]))
        state.register_hook('write_after', checkpoint_write)

        state.value = jnp.array([3.0, 4.0])
        state.value = jnp.array([5.0, 6.0])
        state.value = jnp.array([7.0, 8.0])

        state_id = id(state)
        self.assertEqual(len(checkpoints[state_id]), 3)
        np.testing.assert_array_equal(checkpoints[state_id][0], jnp.array([3.0, 4.0]))
        np.testing.assert_array_equal(checkpoints[state_id][1], jnp.array([5.0, 6.0]))
        np.testing.assert_array_equal(checkpoints[state_id][2], jnp.array([7.0, 8.0]))

    def test_hook_integration_debugging_tracker(self):
        """Test debugging tracker with multiple hook types."""
        tracker = {
            'reads': 0,
            'writes': 0,
            'restores': 0,
            'last_write': None
        }

        def track_read(ctx):
            tracker['reads'] += 1

        def track_write(ctx):
            tracker['writes'] += 1
            tracker['last_write'] = ctx.value.copy()

        def track_restore(ctx):
            tracker['restores'] += 1

        state = brainstate.State(jnp.zeros(3))
        state.register_hook('read', track_read)
        state.register_hook('write_after', track_write)
        state.register_hook('restore', track_restore)

        _ = state.value
        _ = state.value
        state.value = jnp.array([1, 2, 3])
        state.restore_value(jnp.array([4, 5, 6]))

        self.assertEqual(tracker['reads'], 2)
        self.assertEqual(tracker['writes'], 1)
        self.assertEqual(tracker['restores'], 1)
        np.testing.assert_array_equal(tracker['last_write'], jnp.array([1, 2, 3]))

    def test_hook_with_different_state_types(self):
        """Test hooks work with different state types."""
        call_count = {'count': 0}

        def count_hook(ctx):
            call_count['count'] += 1

        brainstate.register_state_hook('read', count_hook)

        state1 = brainstate.State(jnp.zeros(3))
        state2 = brainstate.ShortTermState(jnp.ones(3))
        state3 = brainstate.LongTermState(jnp.array([1, 2]))
        state4 = brainstate.ParamState(jnp.zeros((2, 2)))

        _ = state1.value
        _ = state2.value
        _ = state3.value
        _ = state4.value

        self.assertEqual(call_count['count'], 4)

    def test_hook_performance_many_hooks(self):
        """Test performance with many hooks registered."""
        state = brainstate.State(jnp.zeros(10))

        # Register 50 hooks
        for i in range(50):
            state.register_hook('read', lambda ctx: None, priority=i)

        # Should still work efficiently
        _ = state.value

        # Verify all hooks registered
        self.assertEqual(len(state.list_hooks('read')), 50)

    def test_hook_with_complex_pytree_values(self):
        """Test hooks work with complex pytree state values."""
        complex_value = {
            'a': jnp.array([1, 2]),
            'b': {'c': jnp.array([3, 4]), 'd': jnp.array([5, 6])}
        }

        state = brainstate.State(complex_value)

        captured = {}

        def capture_complex(ctx):
            captured['value'] = ctx.value

        state.register_hook('read', capture_complex)

        _ = state.value

        self.assertIn('a', captured['value'])
        self.assertIn('b', captured['value'])


class TestIntegrationScenarios(unittest.TestCase):
    """Test integration scenarios combining multiple features."""

    def test_neural_network_state_management(self):
        """Test typical neural network state management scenario."""
        # Create network states
        weights = brainstate.ParamState(jnp.ones((10, 5)), name='weights')
        bias = brainstate.ParamState(jnp.zeros(5), name='bias')
        hidden = brainstate.HiddenState(jnp.zeros(5), name='hidden')

        # Trace computation
        stack = brainstate.StateTraceStack(name='forward')
        with stack:
            # Simulate forward pass
            x = jnp.ones(10)
            h = jnp.dot(x, weights.value) + bias.value
            hidden.value = h

        # Check that states were tracked
        self.assertIn(weights, stack.states)
        self.assertIn(bias, stack.states)
        self.assertIn(hidden, stack.states)

        # Check read/write status
        write_states = stack.get_write_states()
        self.assertIn(hidden, write_states)

    def test_recurrent_network_with_multiple_hidden_states(self):
        """Test RNN with multiple hidden states using HiddenGroupState."""
        # Create group of hidden states (e.g., LSTM: h, c)
        hidden_group = brainstate.HiddenGroupState(jnp.array(np.random.randn(10, 10, 2)))

        # Access individual hidden states
        h = hidden_group.get_value(0)
        c = hidden_group.get_value(1)

        self.assertEqual(h.shape, (10, 10))
        self.assertEqual(c.shape, (10, 10))

        # Update hidden states
        new_h = jnp.tanh(h)
        new_c = c * 0.9
        hidden_group.set_value({0: new_h, 1: new_c})

        # Verify updates
        np.testing.assert_array_almost_equal(hidden_group.get_value(0), new_h)
        np.testing.assert_array_almost_equal(hidden_group.get_value(1), new_c)

    def test_state_dict_manager_integration(self):
        """Test managing multiple states with StateDictManager."""
        manager = brainstate.StateDictManager()

        # Create and register states
        with brainstate.catch_new_states('network') as catcher:
            weights = brainstate.ParamState(jnp.ones((5, 3)), name='weights')
            bias = brainstate.ParamState(jnp.zeros(3), name='bias')
            hidden = brainstate.HiddenState(jnp.zeros(3), name='hidden')

        # Add to manager
        for state in catcher.get_states():
            if state.name:
                manager[state.name] = state

        # Collect all values
        values = manager.collect_values()
        self.assertEqual(len(values), 3)

        # Split by type
        params, others = manager.split_values(brainstate.ParamState)
        self.assertEqual(len(params), 2)

    def test_eligibility_trace_learning(self):
        """Test eligibility trace-based learning with HiddenTreeState."""
        # Create multiple eligibility traces with different units
        traces = brainstate.HiddenTreeState({
            'v': brainstate.random.randn(10, 10) * u.mV,
            'u': brainstate.random.randn(10, 10) * u.mV,
            'g': brainstate.random.randn(10, 10) * u.mS,
        })

        # Verify structure
        self.assertEqual(traces.num_state, 3)
        self.assertEqual(traces.varshape, (10, 10))

        # Update individual traces
        new_v = jnp.ones((10, 10)) * u.mV
        traces.set_value({'v': new_v})

        retrieved_v = traces.get_value('v')
        np.testing.assert_array_almost_equal(retrieved_v.mantissa, new_v.mantissa)
        self.assertEqual(retrieved_v.unit, u.mV)

    def test_jit_compilation_with_states(self):
        """Test that states work with JAX JIT compilation."""
        state = brainstate.State(jnp.array([1.0, 2.0, 3.0]))

        @jax.jit
        def update_state(x):
            state.value = state.value + x
            return state.value

        result = update_state(jnp.array([1.0, 1.0, 1.0]))
        np.testing.assert_array_equal(result, jnp.array([2.0, 3.0, 4.0]))


class TestStateTransformInteraction(unittest.TestCase):
    """State read/write semantics inside JAX transforms."""

    def test_state_write_inside_jit_persists(self):
        """A state written inside a jitted function retains its new value."""
        import jax.numpy as jnp
        import brainstate

        st = brainstate.State(jnp.zeros((4,)))

        @brainstate.transform.jit
        def step(x):
            st.value = st.value + x
            return st.value

        out = step(jnp.ones((4,)))
        self.assertTrue(bool(jnp.allclose(out, jnp.ones((4,)))))
        self.assertTrue(bool(jnp.allclose(st.value, jnp.ones((4,)))))

    def test_grad_through_param_state(self):
        """grad w.r.t. a ParamState produces finite gradients."""
        import jax.numpy as jnp
        import brainstate

        p = brainstate.ParamState(jnp.ones((3,)))

        def loss():
            return jnp.sum(p.value ** 2)

        grads = brainstate.transform.grad(loss, grad_states=p)()
        self.assertTrue(bool(jnp.all(jnp.isfinite(grads))))

    def test_state_in_vmap_is_batched(self):
        """A function reading state batches cleanly under vmap_new_states."""
        import jax.numpy as jnp
        import brainstate

        def make_and_run(x):
            s = brainstate.State(x)
            return s.value * 2

        out = brainstate.transform.vmap(make_and_run)(jnp.arange(4.0))
        self.assertEqual(out.shape, (4,))


class TestStateSerializationRoundtrip(unittest.TestCase):
    """Pytree / treefy / state-ref roundtrips for the State hierarchy."""

    def test_treefy_state_pytree_roundtrip(self):
        """A State's treefied form survives flatten/unflatten by structure and value.

        A raw :class:`~brainstate.State` is an opaque pytree *leaf* (it is tracked by
        the state-trace machinery, not by ``jax.tree``); its pytree-serializable form is
        the :class:`TreefyState` produced by :meth:`State.to_state_ref`. The roundtrip is
        therefore asserted on that treefied form.
        """
        import jax.numpy as jnp
        import brainstate
        from brainstate import _testing

        st = brainstate.State(jnp.arange(6.0).reshape(2, 3))
        _testing.assert_pytree_roundtrip(st.to_state_ref())

    def test_to_state_ref_update_from_ref(self):
        """to_state_ref then update_from_ref restores the value."""
        import jax.numpy as jnp
        import brainstate

        src = brainstate.State(jnp.ones((2, 2)))
        ref = src.to_state_ref()
        dst = brainstate.State(jnp.zeros((2, 2)))
        dst.update_from_ref(ref)
        self.assertTrue(bool(jnp.allclose(dst.value, jnp.ones((2, 2)))))

    def test_state_dict_manager_roundtrip(self):
        """StateDictManager collects and reassigns state values."""
        import jax.numpy as jnp
        import brainstate

        s1 = brainstate.State(jnp.ones((2,)))
        mgr = brainstate.StateDictManager()
        mgr['s1'] = s1
        self.assertIn('s1', mgr)


class TestStateHelpersAndEdges(unittest.TestCase):
    """maybe_state, catch_new_states, FakeState, DelayState, and edge cases."""

    def test_maybe_state_unwraps_state(self):
        """maybe_state returns the inner value for a State, else the input."""
        import jax.numpy as jnp
        import brainstate

        st = brainstate.State(jnp.array(3.0))
        self.assertTrue(bool(jnp.allclose(brainstate.maybe_state(st), 3.0)))
        self.assertEqual(brainstate.maybe_state(5), 5)

    def test_catch_new_states_collects(self):
        """catch_new_states captures states created in its scope."""
        import jax.numpy as jnp
        import brainstate

        with brainstate.catch_new_states("tagged") as catcher:
            _ = brainstate.State(jnp.ones((2,)))
        self.assertIsNotNone(catcher)

    def test_invalid_value_tree_assignment_raises(self):
        """Assigning a structurally-different value raises under tree checking."""
        import jax.numpy as jnp
        import brainstate

        st = brainstate.State({'a': jnp.ones((2,))})
        with brainstate.check_state_value_tree(True):
            with self.assertRaises(Exception):
                st.value = {'a': jnp.ones((2,)), 'b': jnp.ones((2,))}


class TestStateMethodsCoverage(unittest.TestCase):
    """Cover the public State methods: tags, numel, replace/copy, stack level, value_call."""

    def test_add_tag_creates_and_dedups(self):
        """add_tag adds to a tagless state and does not duplicate an existing tag."""
        s = brainstate.State(jnp.ones((2,)))
        self.assertIsNone(s.tag)
        s.add_tag('x')
        self.assertIn('x', s.tag)
        s.add_tag('x')
        self.assertEqual(len(s.tag), 1)
        s.add_tag('y')
        self.assertEqual(s.tag, {'x', 'y'})

    def test_numel_scalar_array_and_pytree(self):
        """numel counts scalar=1, arrays by size, and sums over a pytree."""
        self.assertEqual(brainstate.State(jnp.array(3.0)).numel(), 1)
        self.assertEqual(brainstate.State(jnp.ones((2, 3))).numel(), 6)
        self.assertEqual(brainstate.State({'a': jnp.ones((2,)), 'b': jnp.ones((3,))}).numel(), 5)

    def test_replace_plain_value(self):
        """replace(value) returns a new instance with the new value, leaving the original."""
        s = brainstate.State(jnp.ones((2,)))
        s2 = s.replace(jnp.zeros((2,)))
        self.assertTrue(bool(jnp.allclose(s2.value, jnp.zeros((2,)))))
        self.assertTrue(bool(jnp.allclose(s.value, jnp.ones((2,)))))

    def test_replace_with_same_type_state_returns_that_state(self):
        """replace(other_state) of the same type returns the other state directly."""
        s = brainstate.State(jnp.ones((2,)))
        other = brainstate.State(jnp.zeros((2,)))
        self.assertIs(s.replace(other), other)

    def test_replace_with_incompatible_state_raises(self):
        """replace with a different State subclass raises ValueError."""
        s = brainstate.State(jnp.ones((2,)))
        other = brainstate.ShortTermState(jnp.zeros((2,)))
        with self.assertRaises(ValueError):
            s.replace(other)

    def test_copy_preserves_value_and_name(self):
        """copy yields a distinct object with the same value and name."""
        s = brainstate.State(jnp.ones((2,)), name='a')
        c = s.copy()
        self.assertIsNot(c, s)
        self.assertEqual(c.name, 'a')
        self.assertTrue(bool(jnp.allclose(c.value, s.value)))

    def test_copy_from_same_type(self):
        """copy_from overwrites the value from a same-type state."""
        a = brainstate.State(jnp.ones((2,)))
        b = brainstate.State(jnp.zeros((2,)))
        a.copy_from(b)
        self.assertTrue(bool(jnp.allclose(a.value, jnp.zeros((2,)))))

    def test_copy_from_incompatible_raises(self):
        """copy_from a different State subclass raises ValueError."""
        a = brainstate.State(jnp.ones((2,)))
        b = brainstate.ShortTermState(jnp.zeros((2,)))
        with self.assertRaises(ValueError):
            a.copy_from(b)

    def test_copy_from_self_is_noop(self):
        """copy_from(self) returns without altering the value."""
        a = brainstate.State(jnp.ones((2,)))
        a.copy_from(a)
        self.assertTrue(bool(jnp.allclose(a.value, jnp.ones((2,)))))

    def test_stack_level_increase_decrease_and_clamp(self):
        """increase/decrease adjust the stack level, clamped at zero, and the setter works."""
        s = brainstate.State(jnp.ones((2,)))
        base = s.stack_level
        s.increase_stack_level()
        self.assertEqual(s.stack_level, base + 1)
        s.decrease_stack_level()
        self.assertEqual(s.stack_level, base)
        s.stack_level = 0
        s.decrease_stack_level()
        self.assertEqual(s.stack_level, 0)
        s.stack_level = 7
        self.assertEqual(s.stack_level, 7)

    def test_name_setter(self):
        """The name setter updates the state's name."""
        s = brainstate.State(jnp.ones((2,)))
        s.name = 'renamed'
        self.assertEqual(s.name, 'renamed')

    def test_value_call_maps_over_value(self):
        """value_call applies a function to the (pytree) value."""
        s = brainstate.State(jnp.array([1.0, 2.0, 3.0]))
        out = s.value_call(lambda x: x + 1)
        self.assertTrue(bool(jnp.allclose(out, jnp.array([2.0, 3.0, 4.0]))))

    def test_restore_value_replaces_value(self):
        """restore_value writes a structurally-identical value."""
        s = brainstate.State(jnp.ones((2,)))
        s.restore_value(jnp.zeros((2,)))
        self.assertTrue(bool(jnp.allclose(s.value, jnp.zeros((2,)))))

    def test_restore_value_structure_mismatch_raises(self):
        """restore_value always checks the tree structure and raises on mismatch."""
        s = brainstate.State({'a': jnp.ones((2,))})
        with self.assertRaises(Exception):
            s.restore_value({'a': jnp.ones((2,)), 'b': jnp.ones((2,))})

    def test_setting_value_to_a_state_raises(self):
        """Assigning a State as a value is rejected with a clear error."""
        s = brainstate.State(jnp.ones((2,)))
        with self.assertRaises(ValueError):
            s.value = brainstate.State(jnp.zeros((2,)))

    def test_init_from_state_metadata(self):
        """Constructing from StateMetadata unwraps the raw value."""
        from brainstate._state import StateMetadata
        sm = StateMetadata(jnp.ones((2,)), {'custom': 7})
        s = brainstate.State(sm)
        self.assertTrue(bool(jnp.allclose(s.value, jnp.ones((2,)))))
        self.assertEqual(s.custom, 7)

    def test_init_from_state_unwraps_inner_value(self):
        """Constructing from another State copies its inner value."""
        inner = brainstate.State(jnp.full((2,), 4.0))
        s = brainstate.State(inner)
        self.assertTrue(bool(jnp.allclose(s.value, jnp.full((2,), 4.0))))

    def test_repr_shows_name_and_tag(self):
        """repr surfaces both the name and tag of a state."""
        s = brainstate.State(jnp.ones((2,)), name='nm')
        s.add_tag('tg')
        r = repr(s)
        self.assertIn('nm', r)
        self.assertIn('tg', r)


class TestFakeStateCoverage(unittest.TestCase):
    """Cover the FakeState value/name accessors and repr."""

    def test_value_get_set(self):
        """FakeState stores and updates its value."""
        fs = brainstate.FakeState(5)
        self.assertEqual(fs.value, 5)
        fs.value = 9
        self.assertEqual(fs.value, 9)

    def test_name_get_set(self):
        """FakeState stores and updates its name."""
        fs = brainstate.FakeState(0, name='f')
        self.assertEqual(fs.name, 'f')
        fs.name = 'g'
        self.assertEqual(fs.name, 'g')

    def test_repr(self):
        """repr identifies the FakeState and its value."""
        self.assertIn('FakedState', repr(brainstate.FakeState(3)))


class TestStateDictManagerCoverage(unittest.TestCase):
    """Cover StateDictManager collection/assignment helpers and element checking."""

    def setUp(self):
        """Build a manager holding two typed states."""
        self.short = brainstate.ShortTermState(jnp.ones((2,)))
        self.param = brainstate.ParamState(jnp.zeros((3,)))
        self.mgr = brainstate.StateDictManager()
        self.mgr['short'] = self.short
        self.mgr['param'] = self.param

    def test_add_unique_value_rejects_non_state(self):
        """add_unique_value validates the element via _check_elem and rejects non-States."""
        with self.assertRaises(AssertionError):
            self.mgr.add_unique_value('bad', 123)

    def test_add_unique_value_accepts_state(self):
        """add_unique_value stores a State under a new key."""
        extra = brainstate.ShortTermState(jnp.full((2,), 9.0))
        self.assertTrue(self.mgr.add_unique_value('extra', extra))
        self.assertIs(self.mgr['extra'], extra)

    def test_collect_values(self):
        """collect_values returns a mapping of state name to value."""
        vals = self.mgr.collect_values()
        self.assertTrue(bool(jnp.allclose(vals['short'], jnp.ones((2,)))))
        self.assertTrue(bool(jnp.allclose(vals['param'], jnp.zeros((3,)))))

    def test_to_dict_values(self):
        """to_dict_values returns a plain dict of values."""
        d = self.mgr.to_dict_values()
        self.assertEqual(set(d), {'short', 'param'})

    def test_assign_values(self):
        """assign_values writes new values into the underlying states."""
        self.mgr.assign_values({'short': jnp.full((2,), 5.0)})
        self.assertTrue(bool(jnp.allclose(self.short.value, jnp.full((2,), 5.0))))

    def test_assign_values_requires_dict(self):
        """assign_values rejects non-dict arguments."""
        with self.assertRaises(AssertionError):
            self.mgr.assign_values([('short', jnp.ones((2,)))])

    def test_split_values_by_type(self):
        """split_values groups values by the requested state types."""
        params, rest = self.mgr.split_values(brainstate.ParamState)
        self.assertIn('param', params)
        self.assertIn('short', rest)


class TestTreefyStateCoverage(unittest.TestCase):
    """Cover the TreefyState pytree wrapper (to_state, replace, copy, metadata, name)."""

    def setUp(self):
        """Create a TreefyState reference from a named state."""
        self.state = brainstate.State(jnp.ones((2, 2)), name='x')
        self.ref = self.state.to_state_ref()

    def test_to_state_reconstructs(self):
        """to_state rebuilds a State of the original type and value."""
        st = self.ref.to_state()
        self.assertIs(type(st), brainstate.State)
        self.assertTrue(bool(jnp.allclose(st.value, jnp.ones((2, 2)))))

    def test_name_get_set(self):
        """TreefyState exposes and updates the name."""
        self.assertEqual(self.ref.name, 'x')
        self.ref.name = 'y'
        self.assertEqual(self.ref.name, 'y')

    def test_replace_value(self):
        """replace produces a TreefyState carrying the new value."""
        ref2 = self.ref.replace(jnp.zeros((2, 2)))
        self.assertTrue(bool(jnp.allclose(ref2.value, jnp.zeros((2, 2)))))

    def test_copy(self):
        """copy yields an equal-valued TreefyState."""
        ref3 = self.ref.copy()
        self.assertTrue(bool(jnp.allclose(ref3.value, self.ref.value)))

    def test_get_metadata_excludes_type_and_value(self):
        """get_metadata returns a dict without the type/value keys."""
        md = self.ref.get_metadata()
        self.assertIsInstance(md, dict)
        self.assertNotIn('type', md)
        self.assertNotIn('value', md)

    def test_repr(self):
        """repr renders without error and includes the name."""
        self.assertIn('x', repr(self.ref))


class TestNewStateCatcherCoverage(unittest.TestCase):
    """Cover the NewStateCatcher collection API and catch_new_states auto-capture."""

    def test_catch_new_states_auto_captures(self):
        """States created inside catch_new_states are captured and tagged."""
        with brainstate.catch_new_states('grp') as catcher:
            s = brainstate.State(jnp.ones((2,)))
        self.assertIn(s, catcher)
        self.assertGreaterEqual(len(catcher), 1)
        self.assertIn('grp', s.tag)

    def test_catcher_collection_methods(self):
        """append/contains/len/iter/getitem/get_states/values behave consistently."""
        from brainstate._state import NewStateCatcher
        catcher = NewStateCatcher(state_tag='t')
        s1 = brainstate.State(jnp.ones((2,)))
        s2 = brainstate.State(jnp.zeros((2,)))
        catcher.append(s1)
        catcher.append(s1)  # duplicate ignored
        catcher.append(s2)
        self.assertEqual(len(catcher), 2)
        self.assertIn(s1, catcher)
        self.assertIs(catcher[0], s1)
        self.assertEqual(list(catcher), [s1, s2])
        self.assertEqual(catcher.get_states(), [s1, s2])
        self.assertEqual(len(catcher.values), 2)
        self.assertEqual(len(catcher.get_state_values()), 2)
        self.assertIsInstance(catcher.get_by_tag('t'), list)

    def test_catcher_remove_and_clear(self):
        """remove drops a single state and clear empties the catcher."""
        from brainstate._state import NewStateCatcher
        catcher = NewStateCatcher(state_tag='t', state_to_exclude=None)
        s1 = brainstate.State(jnp.ones((2,)))
        s2 = brainstate.State(jnp.zeros((2,)))
        catcher.append(s1)
        catcher.append(s2)
        catcher.remove(s1)
        self.assertNotIn(s1, catcher)
        self.assertEqual(len(catcher), 1)
        catcher.clear()
        self.assertEqual(len(catcher), 0)

    def test_catcher_decrease_stack_level(self):
        """decrease_stack_level lowers the level of every caught state."""
        from brainstate._state import NewStateCatcher
        catcher = NewStateCatcher(state_tag='t')
        s = brainstate.State(jnp.ones((2,)))
        s.stack_level = 3
        catcher.append(s)
        catcher.decrease_stack_level()
        self.assertEqual(s.stack_level, 2)


class TestStateSubclassesCoverage(unittest.TestCase):
    """Cover the thin State subclasses used as type markers."""

    def test_delay_and_nonbatch_states(self):
        """DelayState and NonBatchState construct and store values."""
        from brainstate._state import NonBatchState
        ds = brainstate.DelayState(jnp.ones((2,)))
        nb = NonBatchState(jnp.full((2,), 2.0))
        self.assertTrue(bool(jnp.allclose(ds.value, jnp.ones((2,)))))
        self.assertTrue(bool(jnp.allclose(nb.value, jnp.full((2,), 2.0))))
