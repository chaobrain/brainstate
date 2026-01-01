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
from dataclasses import is_dataclass

import jax.numpy as jnp
import numpy as np

from brainstate.nn import HiData


class TestAutoDataclass(unittest.TestCase):
    """Test that subclasses of HiData automatically become dataclasses."""

    def test_subclass_is_dataclass(self):
        class MyState(HiData):
            x: float
            y: float

        self.assertTrue(is_dataclass(MyState))

    def test_subclass_can_instantiate(self):
        class MyState(HiData):
            x: float
            y: float

        state = MyState(x=1.0, y=2.0)
        self.assertEqual(state.x, 1.0)
        self.assertEqual(state.y, 2.0)

    def test_nested_inheritance(self):
        class BaseState(HiData):
            x: float

        class DerivedState(BaseState):
            y: float

        self.assertTrue(is_dataclass(BaseState))
        self.assertTrue(is_dataclass(DerivedState))


class TestDataMethods(unittest.TestCase):
    """Test HiData class methods."""

    def setUp(self):
        class SimpleState(HiData):
            x: jnp.ndarray
            y: jnp.ndarray

        self.StateClass = SimpleState
        self.state = SimpleState(
            x=jnp.array([1.0, 2.0]),
            y=jnp.array([3.0, 4.0])
        )

    def test_to_dict(self):
        d = self.state.to_dict()
        self.assertIn('x', d)
        self.assertIn('y', d)
        np.testing.assert_array_equal(d['x'], jnp.array([1.0, 2.0]))
        np.testing.assert_array_equal(d['y'], jnp.array([3.0, 4.0]))

    def test_from_dict(self):
        d = {'x': jnp.array([5.0, 6.0]), 'y': jnp.array([7.0, 8.0])}
        state = self.StateClass.from_dict(d)
        np.testing.assert_array_equal(state.x, jnp.array([5.0, 6.0]))
        np.testing.assert_array_equal(state.y, jnp.array([7.0, 8.0]))

    def test_to_dict_from_dict_roundtrip(self):
        d = self.state.to_dict()
        state2 = self.StateClass.from_dict(d)
        np.testing.assert_array_equal(state2.x, self.state.x)
        np.testing.assert_array_equal(state2.y, self.state.y)

    def test_dtype(self):
        self.assertEqual(self.state.dtype, jnp.float32)

    def test_dtype_no_tensors_raises(self):
        class EmptyState(HiData):
            name: str

        state = EmptyState(name="test")
        with self.assertRaises(ValueError):
            _ = state.dtype

    def test_state_size(self):
        self.assertEqual(self.state.state_size, 2)

    def test_replace(self):
        new_x = jnp.array([10.0, 20.0])
        new_state = self.state.replace(x=new_x)
        np.testing.assert_array_equal(new_state.x, new_x)
        np.testing.assert_array_equal(new_state.y, self.state.y)
        # Original unchanged
        np.testing.assert_array_equal(self.state.x, jnp.array([1.0, 2.0]))

    def test_replace_multiple(self):
        new_x = jnp.array([10.0, 20.0])
        new_y = jnp.array([30.0, 40.0])
        new_state = self.state.replace(x=new_x, y=new_y)
        np.testing.assert_array_equal(new_state.x, new_x)
        np.testing.assert_array_equal(new_state.y, new_y)


class TestDataWithNone(unittest.TestCase):
    """Test HiData class with None values."""

    def test_to_dict_with_none(self):
        class OptionalState(HiData):
            x: jnp.ndarray
            y: jnp.ndarray = None

        state = OptionalState(x=jnp.array([1.0]), y=None)
        d = state.to_dict()
        self.assertIsNone(d['y'])

    def test_dtype_skips_none(self):
        class OptionalState(HiData):
            x: jnp.ndarray = None
            y: jnp.ndarray = None

        state = OptionalState(x=None, y=jnp.array([1.0]))
        self.assertEqual(state.dtype, jnp.float32)


class TestComposedParamData(unittest.TestCase):
    """Test HiData class."""

    def test_is_dataclass(self):
        self.assertTrue(is_dataclass(HiData))

    def test_empty_init(self):
        composed = HiData()
        self.assertEqual(len(composed.children), 0)

    def test_init_with_children(self):
        class ChildState(HiData):
            val: jnp.ndarray

        child = ChildState(val=jnp.array([1.0]))
        composed = HiData(children={'child1': child})
        self.assertIn('child1', composed)

    def test_getitem(self):
        class ChildState(HiData):
            val: jnp.ndarray

        child = ChildState(val=jnp.array([1.0]))
        composed = HiData(children={'child1': child})
        np.testing.assert_array_equal(composed['child1'].val, jnp.array([1.0]))

    def test_contains(self):
        composed = HiData(children={'a': 1, 'b': 2})
        self.assertIn('a', composed)
        self.assertIn('b', composed)
        self.assertNotIn('c', composed)

    def test_keys_items_values(self):
        composed = HiData(children={'a': 1, 'b': 2})
        self.assertEqual(set(composed.keys()), {'a', 'b'})
        self.assertEqual(set(composed.values()), {1, 2})
        self.assertEqual(set(composed.items()), {('a', 1), ('b', 2)})

    def test_state_size(self):
        class ChildState(HiData):
            x: jnp.ndarray
            y: jnp.ndarray

        child1 = ChildState(x=jnp.array([1.0]), y=jnp.array([2.0]))
        child2 = ChildState(x=jnp.array([3.0]), y=jnp.array([4.0]))
        composed = HiData(children={'c1': child1, 'c2': child2})
        self.assertEqual(composed.state_size, 4)  # 2 fields * 2 children

    def test_state_size_with_none(self):
        class ChildState(HiData):
            x: jnp.ndarray

        child = ChildState(x=jnp.array([1.0]))
        composed = HiData(children={'c1': child, 'c2': None})
        self.assertEqual(composed.state_size, 1)

    def test_dtype(self):
        class ChildState(HiData):
            x: jnp.ndarray

        child = ChildState(x=jnp.array([1.0], dtype=jnp.float32))
        composed = HiData(children={'c1': child})
        self.assertEqual(composed.dtype, jnp.float32)

    def test_dtype_no_children_raises(self):
        composed = HiData()
        with self.assertRaises(ValueError):
            _ = composed.dtype

    def test_replace(self):
        class ChildState(HiData):
            x: jnp.ndarray

        child1 = ChildState(x=jnp.array([1.0]))
        child2 = ChildState(x=jnp.array([2.0]))
        composed = HiData(children={'c1': child1})

        new_composed = composed.replace(c1=child2)
        np.testing.assert_array_equal(new_composed['c1'].x, jnp.array([2.0]))
        # Original unchanged
        np.testing.assert_array_equal(composed['c1'].x, jnp.array([1.0]))

    def test_clone(self):
        class ChildState(HiData):
            x: jnp.ndarray

            def clone(self):
                return ChildState(x=self.x.copy())

        child = ChildState(x=jnp.array([1.0]))
        composed = HiData(children={'c1': child, 'c2': None})
        cloned = composed.clone()

        self.assertIsNot(cloned, composed)
        self.assertIsNot(cloned.children, composed.children)
        np.testing.assert_array_equal(cloned['c1'].x, child.x)
        self.assertIsNone(cloned['c2'])

    def test_clone_without_clone_method(self):
        composed = HiData(children={'c1': 'simple_value'})
        cloned = composed.clone()
        self.assertEqual(cloned['c1'], 'simple_value')


class TestComposedDataKwargsInit(unittest.TestCase):
    """Test HiData kwargs initialization and attribute access."""

    def test_init_with_kwargs(self):
        class ChildState(HiData):
            val: jnp.ndarray

        child1 = ChildState(val=jnp.array([1.0]))
        child2 = ChildState(val=jnp.array([2.0]))
        composed = HiData(key1=child1, key2=child2)

        self.assertIn('key1', composed)
        self.assertIn('key2', composed)

    def test_getattr_access(self):
        class ChildState(HiData):
            val: jnp.ndarray

        child = ChildState(val=jnp.array([1.0]))
        composed = HiData(mykey=child)

        np.testing.assert_array_equal(composed.mykey.val, jnp.array([1.0]))

    def test_getattr_missing_raises(self):
        composed = HiData()
        with self.assertRaises(AttributeError):
            _ = composed.nonexistent

    def test_mixed_init_children_and_kwargs(self):
        composed = HiData(children={'a': 1}, b=2, c=3)
        self.assertEqual(composed.a, 1)
        self.assertEqual(composed.b, 2)
        self.assertEqual(composed.c, 3)

    def test_kwargs_override_children(self):
        composed = HiData(children={'a': 1}, a=2)
        self.assertEqual(composed.a, 2)

    def test_attribute_and_item_access_equivalent(self):
        composed = HiData(key1='value1', key2='value2')
        self.assertEqual(composed.key1, composed['key1'])
        self.assertEqual(composed.key2, composed['key2'])

    def test_children_attribute_accessible(self):
        composed = HiData(a=1, b=2)
        self.assertIsInstance(composed.children, dict)
        self.assertEqual(composed.children, {'a': 1, 'b': 2})


class TestDataHierarchicalRepr(unittest.TestCase):
    """Test hierarchical __repr__ functionality."""

    def test_simple_data_repr(self):
        """Test repr with simple values."""
        data = HiData(name='config', x=1, y=2.5, z='hello')
        result = repr(data)
        # Root level uses '='
        self.assertIn("name='config'", result)
        self.assertIn("x=1", result)
        self.assertIn("y=2.5", result)
        self.assertIn("z='hello'", result)

    def test_data_with_none_value(self):
        """Test repr with None values."""
        data = HiData(name='test', value=None)
        result = repr(data)
        # Root level uses '='
        self.assertIn("value=None", result)

    def test_nested_data_repr(self):
        """Test repr with nested HiData objects."""
        inner = HiData(name='inner', a=1, b=2)
        outer = HiData(name='outer', child=inner, x=10)
        result = repr(outer)

        # All levels use '='
        self.assertIn("name='outer'", result)
        self.assertIn("child=", result)
        self.assertIn("x=10", result)
        self.assertIn("name='inner'", result)
        self.assertIn("a=1", result)
        self.assertIn("b=2", result)

    def test_deeply_nested_data_repr(self):
        """Test repr with multiple levels of nesting."""
        level3 = HiData(name='level3', value=42)
        level2 = HiData(name='level2', deep=level3)
        level1 = HiData(name='level1', nested=level2)
        root = HiData(name='root', child=level1)

        result = repr(root)
        # All Data objects should have their name
        self.assertIn("name='root'", result)
        self.assertIn("name='level1'", result)
        self.assertIn("name='level2'", result)
        self.assertIn("name='level3'", result)
        # All levels use '='
        self.assertIn("value=42", result)

    def test_array_formatting_numpy(self):
        """Test repr with numpy arrays."""
        arr = np.array([1, 2, 3, 4, 5])
        data = HiData(name='arrays', vector=arr)
        result = repr(data)

        # Root level uses '='
        self.assertIn("name='arrays'", result)
        self.assertIn("vector=Array(shape=(5,), dtype=int", result)

    def test_array_formatting_jax(self):
        """Test repr with JAX arrays."""
        arr = jnp.array([1.0, 2.0, 3.0])
        data = HiData(name='arrays', vector=arr)
        result = repr(data)

        # Root level uses '='
        self.assertIn("name='arrays'", result)
        self.assertIn("vector=Array(shape=(3,), dtype=float", result)

    def test_multidimensional_array_formatting(self):
        """Test repr with 2D arrays."""
        arr = np.zeros((3, 4))
        data = HiData(name='test', matrix=arr)
        result = repr(data)

        # Root level uses '='
        self.assertIn("matrix=Array(shape=(3, 4), dtype=float", result)

    def test_long_string_truncation(self):
        """Test that long strings are truncated."""
        long_string = 'x' * 100
        data = HiData(name='test', long=long_string)
        result = repr(data)

        # Should be truncated to 60 chars - root level uses '='
        self.assertIn("long=", result)
        self.assertIn("...", result)

    def test_mixed_children_repr(self):
        """Test repr with mixed types of children."""
        nested = HiData(name='nested', inner_val=99)
        arr = np.array([1.0, 2.0, 3.0])
        data = HiData(
            name='mixed',
            number=42,
            text='hello',
            array=arr,
            nested=nested,
            none_val=None
        )
        result = repr(data)

        # All levels use '='
        self.assertIn("name='mixed'", result)
        self.assertIn("number=42", result)
        self.assertIn("text='hello'", result)
        self.assertIn("array=Array(shape=(3,), dtype=float", result)
        self.assertIn("nested=", result)
        self.assertIn("none_val=None", result)
        self.assertIn("name='nested'", result)
        self.assertIn("inner_val=99", result)

    def test_empty_nested_data(self):
        """Test nested HiData with no children."""
        inner = HiData(name='empty')
        outer = HiData(name='outer', child=inner)
        result = repr(outer)

        self.assertIn("name='outer'", result)
        self.assertIn("name='empty'", result)

    def test_many_children(self):
        """Test HiData with many children."""
        data = HiData(name='many', **{f'item{i}': i for i in range(20)})
        result = repr(data)

        # Root level uses '='
        self.assertIn("item0=", result)
        self.assertIn("item19=", result)

    def test_repr_with_complex_nested_structure(self):
        """Test repr with a realistic complex nested structure."""
        # Simulate a realistic configuration structure
        optimizer_config = HiData(
            name='adam',
            learning_rate=0.001,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8
        )

        model_config = HiData(
            name='neural_net',
            layers=3,
            hidden_size=128,
            weights=np.random.randn(10, 5)
        )

        training_config = HiData(
            name='training',
            optimizer=optimizer_config,
            model=model_config,
            epochs=100,
            batch_size=32
        )

        result = repr(training_config)

        # Check all components are present
        self.assertIn("name='training'", result)
        self.assertIn("name='adam'", result)
        self.assertIn("name='neural_net'", result)
        # All levels use '='
        self.assertIn("learning_rate=0.001", result)
        self.assertIn("layers=3", result)
        self.assertIn("epochs=100", result)
        self.assertIn("weights=Array(shape=(10, 5), dtype=float", result)


if __name__ == '__main__':
    unittest.main()
