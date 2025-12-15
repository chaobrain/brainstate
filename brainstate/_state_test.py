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

import braintools
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

import brainstate


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
            'v': np.random.randn(10, 10) * u.mV,
            'i': np.random.randn(10, 10) * u.mA,
        }
        state = brainstate.HiddenTreeState(value)

        new_v = np.ones((10, 10)) * u.mV
        state.set_value({'v': new_v})

        retrieved = state.get_value('v')
        np.testing.assert_array_almost_equal(retrieved.mantissa, new_v.mantissa)

    def test_hidden_tree_state_set_value_list(self):
        """Test setting values with list."""
        value = [
            np.random.randn(10, 10) * u.mV,
            np.random.randn(10, 10) * u.mA,
        ]
        state = brainstate.HiddenTreeState(value)

        new_vals = [
            np.ones((10, 10)) * u.mV,
            np.zeros((10, 10)) * u.mA,
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
        ref = brainstate.TreefyState(brainstate.State, jnp.array([1.0, 2.0]), _name='test')
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
        self.assertEqual(state1.tag, 'test_tag')
        self.assertEqual(state2.tag, 'test_tag')

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
    """Test StateCatcher functionality via catch_new_states context manager."""

    def test_catch_new_states_basic(self):
        """Test basic catch_new_states functionality."""
        with brainstate.catch_new_states('test') as catcher:
            state = brainstate.State(jnp.zeros(3))

        self.assertEqual(len(catcher), 1)
        self.assertEqual(state.tag, 'test')

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
            'v': np.random.randn(10, 10) * u.mV,
            'u': np.random.randn(10, 10) * u.mV,
            'g': np.random.randn(10, 10) * u.mS,
        })

        # Verify structure
        self.assertEqual(traces.num_state, 3)
        self.assertEqual(traces.varshape, (10, 10))

        # Update individual traces
        new_v = np.ones((10, 10)) * u.mV
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

