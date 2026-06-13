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

"""
Comprehensive tests for the others module.
"""

import pickle
import threading
import unittest
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp

from brainstate.util._others import (
    DictManager,
    DotDict,
    NameContext,
    clear_buffer_memory,
    flatten_dict,
    get_unique_name,
    is_instance_eval,
    merge_dicts,
    not_instance_eval,
    split_total,
    unflatten_dict,
)


class TestSplitTotal(unittest.TestCase):
    """Test cases for split_total function."""

    def test_float_fraction(self):
        """Test with float fraction values."""
        self.assertEqual(split_total(100, 0.5), 50)
        self.assertEqual(split_total(100, 0.25), 25)
        self.assertEqual(split_total(100, 0.75), 75)
        self.assertEqual(split_total(100, 0.0), 0)
        self.assertEqual(split_total(100, 1.0), 100)

    def test_int_fraction(self):
        """Test with integer fraction values."""
        self.assertEqual(split_total(100, 25), 25)
        self.assertEqual(split_total(100, 0), 0)
        self.assertEqual(split_total(100, 100), 100)
        self.assertEqual(split_total(50, 30), 30)

    def test_edge_cases(self):
        """Test edge cases."""
        self.assertEqual(split_total(1, 0.5), 0)  # int(0.5) = 0
        self.assertEqual(split_total(1, 1), 1)
        self.assertEqual(split_total(10, 0.99), 9)  # int(9.9) = 9

    def test_type_errors(self):
        """Test type error handling."""
        with self.assertRaises(TypeError) as ctx:
            split_total("100", 0.5)
        self.assertIn("must be an integer", str(ctx.exception))

        with self.assertRaises(TypeError) as ctx:
            split_total(100, "0.5")
        self.assertIn("must be an integer or float", str(ctx.exception))

        with self.assertRaises(TypeError) as ctx:
            split_total(100.5, 0.5)
        self.assertIn("must be an integer", str(ctx.exception))

    def test_value_errors(self):
        """Test value error handling."""
        # Negative total
        with self.assertRaises(ValueError) as ctx:
            split_total(-10, 0.5)
        self.assertIn("must be a positive integer", str(ctx.exception))

        # Zero total
        with self.assertRaises(ValueError) as ctx:
            split_total(0, 0.5)
        self.assertIn("must be a positive integer", str(ctx.exception))

        # Negative fraction (float)
        with self.assertRaises(ValueError) as ctx:
            split_total(100, -0.5)
        self.assertIn("cannot be negative", str(ctx.exception))

        # Fraction > 1 (float)
        with self.assertRaises(ValueError) as ctx:
            split_total(100, 1.5)
        self.assertIn("cannot be greater than 1", str(ctx.exception))

        # Negative fraction (int)
        with self.assertRaises(ValueError) as ctx:
            split_total(100, -10)
        self.assertIn("cannot be negative", str(ctx.exception))

        # Fraction > total (int)
        with self.assertRaises(ValueError) as ctx:
            split_total(100, 150)
        self.assertIn("cannot be greater than total", str(ctx.exception))

    def test_bool_rejected(self):
        """Reject bool arguments so they are never treated as 0/1 or leak into the int return.

        ``bool`` is a subclass of ``int``; without an explicit guard
        ``split_total(100, True)`` would return ``True`` and ``split_total(True, 0.5)``
        would silently treat the total as ``1``.
        """
        # bool fraction must raise (would otherwise be int 0/1 and return a bool)
        for frac in (True, False):
            with self.assertRaises(TypeError) as ctx:
                split_total(100, frac)
            self.assertIn("must be an integer or float", str(ctx.exception))

        # bool total must raise (would otherwise pass the int check as 0/1)
        for tot in (True, False):
            with self.assertRaises(TypeError) as ctx:
                split_total(tot, 0.5)
            self.assertIn("must be an integer", str(ctx.exception))

    def test_return_type_is_strict_int(self):
        """Return a strict ``int`` (never a ``bool``) for all valid inputs."""
        for r in (split_total(100, 0.5), split_total(100, 25), split_total(100, 100), split_total(100, 0)):
            self.assertIs(type(r), int)


class TestNameContext(unittest.TestCase):
    """Test cases for NameContext and get_unique_name."""

    def setUp(self):
        """Reset the global NAME context before each test."""
        global NAME
        from brainstate.util._others import NAME
        NAME.typed_names.clear()

    def test_get_unique_name_basic(self):
        """Test basic unique name generation."""
        name1 = get_unique_name('layer')
        name2 = get_unique_name('layer')
        name3 = get_unique_name('layer')

        self.assertEqual(name1, 'layer0')
        self.assertEqual(name2, 'layer1')
        self.assertEqual(name3, 'layer2')

    def test_get_unique_name_with_prefix(self):
        """Test unique name generation with prefix."""
        name1 = get_unique_name('layer', 'conv_')
        name2 = get_unique_name('layer', 'conv_')
        name3 = get_unique_name('layer', 'dense_')

        self.assertEqual(name1, 'conv_layer0')
        self.assertEqual(name2, 'conv_layer1')
        self.assertEqual(name3, 'dense_layer2')

    def test_different_types(self):
        """Test unique names for different types."""
        layer1 = get_unique_name('layer')
        neuron1 = get_unique_name('neuron')
        layer2 = get_unique_name('layer')
        neuron2 = get_unique_name('neuron')

        self.assertEqual(layer1, 'layer0')
        self.assertEqual(neuron1, 'neuron0')
        self.assertEqual(layer2, 'layer1')
        self.assertEqual(neuron2, 'neuron1')

    def test_thread_local_context(self):
        """Test that NameContext is thread-local."""
        results = {}

        def worker(thread_id):
            name1 = get_unique_name(f'type_{thread_id}')
            name2 = get_unique_name(f'type_{thread_id}')
            results[thread_id] = (name1, name2)

        threads = []
        for i in range(3):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Each thread should have independent counters
        for thread_id in range(3):
            self.assertEqual(results[thread_id][0], f'type_{thread_id}0')
            self.assertEqual(results[thread_id][1], f'type_{thread_id}1')

    def test_name_context_reset(self):
        """Test resetting name context."""
        context = NameContext()
        context.typed_names['test'] = 5

        # Reset specific type
        context.reset('test')
        self.assertEqual(context.typed_names.get('test'), 0)

        # Add more names
        context.typed_names['test1'] = 1
        context.typed_names['test2'] = 2

        # Reset all
        context.reset()
        self.assertEqual(len(context.typed_names), 0)


class TestDictManager(unittest.TestCase):
    """Test cases for DictManager class."""

    def test_initialization(self):
        """Test DictManager initialization."""
        # Empty initialization
        dm1 = DictManager()
        self.assertEqual(len(dm1), 0)

        # From dict
        dm2 = DictManager({'a': 1, 'b': 2})
        self.assertEqual(dm2['a'], 1)
        self.assertEqual(dm2['b'], 2)

        # From kwargs
        dm3 = DictManager(a=1, b=2)
        self.assertEqual(dm3['a'], 1)
        self.assertEqual(dm3['b'], 2)

        # From items
        dm4 = DictManager([('a', 1), ('b', 2)])
        self.assertEqual(dm4['a'], 1)
        self.assertEqual(dm4['b'], 2)

    def test_subset(self):
        """Test subset filtering."""
        dm = DictManager({'a': 1, 'b': 2.0, 'c': 'text', 'd': 3})

        # By type
        int_subset = dm.subset(int)
        self.assertEqual(dict(int_subset), {'a': 1, 'd': 3})

        # By multiple types
        num_subset = dm.subset((int, float))
        self.assertEqual(dict(num_subset), {'a': 1, 'b': 2.0, 'd': 3})

        # By predicate
        large_subset = dm.subset(lambda x: isinstance(x, (int, float)) and x > 1.5)
        self.assertEqual(dict(large_subset), {'b': 2.0, 'd': 3})

    def test_not_subset(self):
        """Test not_subset filtering."""
        dm = DictManager({'a': 1, 'b': 2.0, 'c': 'text', 'd': 3})

        not_int = dm.not_subset(int)
        self.assertEqual(dict(not_int), {'b': 2.0, 'c': 'text'})

        not_num = dm.not_subset((int, float))
        self.assertEqual(dict(not_num), {'c': 'text'})

    def test_add_unique_key(self):
        """Test adding unique keys."""
        dm = DictManager()
        obj1 = object()
        obj2 = object()

        # Add new key
        dm.add_unique_key('key1', obj1)
        self.assertIs(dm['key1'], obj1)

        # Add same key with same object (should work)
        dm.add_unique_key('key1', obj1)
        self.assertIs(dm['key1'], obj1)

        # Add same key with different object (should fail)
        with self.assertRaises(ValueError) as ctx:
            dm.add_unique_key('key1', obj2)
        self.assertIn("already exists with a different value", str(ctx.exception))

    def test_add_unique_value(self):
        """Test adding unique values."""
        dm = DictManager()
        obj1 = object()

        # First addition should succeed
        result1 = dm.add_unique_value('key1', obj1)
        self.assertTrue(result1)
        self.assertIs(dm['key1'], obj1)

        # Adding same value with different key should fail
        result2 = dm.add_unique_value('key2', obj1)
        self.assertFalse(result2)
        self.assertNotIn('key2', dm)

        # Adding different value should succeed
        obj2 = object()
        result3 = dm.add_unique_value('key2', obj2)
        self.assertTrue(result3)
        self.assertIs(dm['key2'], obj2)

    def test_add_unique_value_allows_readd_after_delete(self):
        """Keep the identity index in sync after direct item deletion."""
        dm = DictManager()
        obj = object()
        self.assertTrue(dm.add_unique_value('key1', obj))

        del dm['key1']

        self.assertTrue(dm.add_unique_value('key2', obj))
        self.assertIs(dm['key2'], obj)

    def test_add_unique_value_indexes_existing_direct_mutations(self):
        """Rebuild stale identity indexes when values were added without helper APIs."""
        dm = DictManager()
        obj = object()
        self.assertTrue(dm.add_unique_value('key1', object()))

        dm['direct'] = obj

        self.assertFalse(dm.add_unique_value('duplicate', obj))
        self.assertNotIn('duplicate', dm)

    def test_add_unique_value_rejects_existing_key_with_different_value(self):
        """Do not overwrite an existing key when the new value is unique."""
        dm = DictManager()
        obj1 = object()
        obj2 = object()
        self.assertTrue(dm.add_unique_value('key1', obj1))

        with self.assertRaises(ValueError):
            dm.add_unique_value('key1', obj2)

        self.assertIs(dm['key1'], obj1)

    def test_unique(self):
        """Test getting unique values."""
        obj1 = object()
        obj2 = object()
        dm = DictManager({'a': obj1, 'b': obj2, 'c': obj1, 'd': obj2, 'e': obj1})

        unique_dm = dm.unique()
        self.assertEqual(len(unique_dm), 2)
        # Check that each object appears only once
        values = list(unique_dm.values())
        self.assertEqual(len(set(id(v) for v in values)), 2)

    def test_unique_inplace(self):
        """Test in-place unique operation."""
        obj1 = object()
        obj2 = object()
        dm = DictManager({'a': obj1, 'b': obj2, 'c': obj1})

        result = dm.unique_()
        self.assertIs(result, dm)  # Should return self
        self.assertEqual(len(dm), 2)  # One duplicate removed

    def test_assign(self):
        """Test assign method."""
        dm = DictManager({'a': 1})

        # Assign from dict
        dm.assign({'b': 2, 'c': 3})
        self.assertEqual(dict(dm), {'a': 1, 'b': 2, 'c': 3})

        # Assign from multiple dicts
        dm.assign({'d': 4}, {'e': 5})
        self.assertEqual(len(dm), 5)

        # Assign with kwargs
        dm.assign(f=6, g=7)
        self.assertEqual(dm['f'], 6)
        self.assertEqual(dm['g'], 7)

        # Invalid argument
        with self.assertRaises(TypeError):
            dm.assign([1, 2, 3])

    def test_split(self):
        """Test splitting by types."""
        dm = DictManager({
            'a': 1, 'b': 2.0, 'c': 'text',
            'd': 3, 'e': 4.5, 'f': [1, 2]
        })

        int_dm, float_dm, rest = dm.split(int, float)

        self.assertEqual(dict(int_dm), {'a': 1, 'd': 3})
        self.assertEqual(dict(float_dm), {'b': 2.0, 'e': 4.5})
        self.assertEqual(dict(rest), {'c': 'text', 'f': [1, 2]})

    def test_filter_by_predicate(self):
        """Test filtering with predicate."""
        dm = DictManager({'a': 1, 'b': 2, 'c': 3, 'd': 4})

        # Filter by key
        filtered = dm.filter_by_predicate(lambda k, v: k in ['a', 'c'])
        self.assertEqual(dict(filtered), {'a': 1, 'c': 3})

        # Filter by value
        filtered = dm.filter_by_predicate(lambda k, v: v > 2)
        self.assertEqual(dict(filtered), {'c': 3, 'd': 4})

        # Filter by both
        filtered = dm.filter_by_predicate(lambda k, v: k == 'a' or v == 4)
        self.assertEqual(dict(filtered), {'a': 1, 'd': 4})

    def test_map_values(self):
        """Test mapping function to values."""
        dm = DictManager({'a': 1, 'b': 2, 'c': 3})

        doubled = dm.map_values(lambda x: x * 2)
        self.assertEqual(dict(doubled), {'a': 2, 'b': 4, 'c': 6})

        # Original should be unchanged
        self.assertEqual(dict(dm), {'a': 1, 'b': 2, 'c': 3})

    def test_map_keys(self):
        """Test mapping function to keys."""
        dm = DictManager({'a': 1, 'b': 2, 'c': 3})

        upper = dm.map_keys(str.upper)
        self.assertEqual(dict(upper), {'A': 1, 'B': 2, 'C': 3})

        # Test duplicate key error
        with self.assertRaises(ValueError) as ctx:
            dm.map_keys(lambda x: 'same')
        self.assertIn("duplicate", str(ctx.exception))

    def test_pop_by_keys(self):
        """Test removing items by keys."""
        dm = DictManager({'a': 1, 'b': 2, 'c': 3, 'd': 4})

        dm.pop_by_keys(['b', 'd'])
        self.assertEqual(dict(dm), {'a': 1, 'c': 3})

        # Pop non-existent keys (should not raise)
        dm.pop_by_keys(['x', 'y'])
        self.assertEqual(dict(dm), {'a': 1, 'c': 3})

    def test_pop_by_values(self):
        """Test removing items by values."""
        obj1, obj2, obj3 = object(), object(), object()
        dm = DictManager({'a': obj1, 'b': obj2, 'c': obj3})

        # By identity
        dm_copy = DictManager(dm)
        dm_copy.pop_by_values([obj2], by='id')
        self.assertEqual(len(dm_copy), 2)
        self.assertNotIn('b', dm_copy)

        # By value equality
        dm2 = DictManager({'a': 1, 'b': 2, 'c': 3})
        dm2.pop_by_values([2, 3], by='value')
        self.assertEqual(dict(dm2), {'a': 1})

        # Invalid method
        with self.assertRaises(ValueError):
            dm.pop_by_values([obj1], by='invalid')

    def test_difference_operations(self):
        """Test difference operations."""
        dm = DictManager({'a': 1, 'b': 2, 'c': 3, 'd': 4})

        # Difference by keys
        diff = dm.difference_by_keys(['b', 'd'])
        self.assertEqual(dict(diff), {'a': 1, 'c': 3})

        # Difference by values
        diff = dm.difference_by_values([2, 4], by='value')
        self.assertEqual(dict(diff), {'a': 1, 'c': 3})

    def test_intersection_operations(self):
        """Test intersection operations."""
        dm = DictManager({'a': 1, 'b': 2, 'c': 3, 'd': 4})

        # Intersection by keys
        inter = dm.intersection_by_keys(['a', 'c', 'e'])
        self.assertEqual(dict(inter), {'a': 1, 'c': 3})

        # Intersection by values
        inter = dm.intersection_by_values([1, 3, 5], by='value')
        self.assertEqual(dict(inter), {'a': 1, 'c': 3})

    def test_operators(self):
        """Test operator overloading."""
        dm1 = DictManager({'a': 1, 'b': 2})
        dm2 = DictManager({'c': 3, 'd': 4})

        # Addition
        dm3 = dm1 + dm2
        self.assertEqual(dict(dm3), {'a': 1, 'b': 2, 'c': 3, 'd': 4})
        self.assertIsNot(dm3, dm1)  # New object

        # Addition with regular dict
        dm4 = dm1 + {'e': 5}
        self.assertEqual(dm4['e'], 5)

        # Union operator (Python 3.9+) - only test if available
        import sys
        if sys.version_info >= (3, 9):
            dm5 = dm1 | dm2
            self.assertEqual(dict(dm5), {'a': 1, 'b': 2, 'c': 3, 'd': 4})

            # In-place union
            dm1_copy = DictManager(dm1)
            dm1_copy |= dm2
            self.assertEqual(dict(dm1_copy), {'a': 1, 'b': 2, 'c': 3, 'd': 4})

        # Invalid operations should raise TypeError or return NotImplemented
        # The actual behavior depends on whether __add__ returns NotImplemented
        # or lets Python raise TypeError
        with self.assertRaises(TypeError):
            _ = dm1 + 123
        if sys.version_info >= (3, 9):
            with self.assertRaises(TypeError):
                _ = dm1 | 123

    def test_copy_operations(self):
        """Test copy operations."""
        obj = object()
        dm1 = DictManager({'a': 1, 'b': obj})

        # Shallow copy
        dm2 = dm1.__copy__()
        self.assertIsNot(dm2, dm1)
        self.assertEqual(dict(dm2), dict(dm1))
        self.assertIs(dm2['b'], obj)  # Same object reference

        # Deep copy
        dm3 = dm1.__deepcopy__({})
        self.assertIsNot(dm3, dm1)
        self.assertEqual(dm3['a'], 1)
        # Note: object() can't be deep copied, but other values work

    def test_jax_pytree(self):
        """Test JAX pytree registration."""
        dm = DictManager({'a': 1, 'b': 2, 'c': 3})

        # Flatten
        values, keys = dm.tree_flatten()
        self.assertEqual(len(values), 3)
        self.assertEqual(len(keys), 3)

        # Unflatten
        dm2 = DictManager.tree_unflatten(keys, values)
        self.assertEqual(dict(dm2), dict(dm))

        # Test with JAX tree operations
        dm3 = DictManager({'x': jnp.array([1, 2]), 'y': jnp.array([3, 4])})
        doubled = jax.tree_util.tree_map(lambda x: x * 2, dm3)
        self.assertTrue(jnp.allclose(doubled['x'], jnp.array([2, 4])))
        self.assertTrue(jnp.allclose(doubled['y'], jnp.array([6, 8])))

    def test_repr(self):
        """Test string representation."""
        dm = DictManager({'a': 1, 'b': 'text'})
        repr_str = repr(dm)
        self.assertIn('DictManager', repr_str)
        self.assertIn("'a': 1", repr_str)
        self.assertIn("'b': 'text'", repr_str)


class TestDotDict(unittest.TestCase):
    """Test cases for DotDict class."""

    def test_initialization(self):
        """Test DotDict initialization."""
        # From dict
        dd1 = DotDict({'a': 1, 'b': 2})
        self.assertEqual(dd1.a, 1)
        self.assertEqual(dd1['b'], 2)

        # From kwargs
        dd2 = DotDict(a=1, b=2)
        self.assertEqual(dd2.a, 1)
        self.assertEqual(dd2.b, 2)

        # From tuple
        dd3 = DotDict(('key', 'value'))
        self.assertEqual(dd3.key, 'value')

        # From items
        dd4 = DotDict([('a', 1), ('b', 2)])
        self.assertEqual(dd4.a, 1)
        self.assertEqual(dd4.b, 2)

        # Empty
        dd5 = DotDict()
        self.assertEqual(len(dd5), 0)

        # Invalid argument
        with self.assertRaises(TypeError):
            DotDict(123)

    def test_dot_access(self):
        """Test dot notation access."""
        dd = DotDict({'a': 1, 'b': {'c': 2, 'd': 3}})

        # Read access
        self.assertEqual(dd.a, 1)
        self.assertEqual(dd.b.c, 2)
        self.assertEqual(dd.b.d, 3)

        # Write access
        dd.a = 10
        dd.b.c = 20
        self.assertEqual(dd['a'], 10)
        self.assertEqual(dd['b']['c'], 20)

        # Add new attributes
        dd.e = 5
        self.assertEqual(dd['e'], 5)

        # Direct item assignment should still apply nested DotDict conversion.
        dd['nested'] = {'x': 1}
        self.assertIsInstance(dd.nested, DotDict)
        self.assertEqual(dd.nested.x, 1)

        # Attribute assignment should use the same conversion path.
        dd.another = {'y': 2}
        self.assertIsInstance(dd.another, DotDict)
        self.assertEqual(dd.another.y, 2)

    def test_nested_dict_conversion(self):
        """Test automatic nested dict conversion."""
        dd = DotDict({
            'level1': {
                'level2': {
                    'level3': 'value'
                }
            }
        })

        self.assertIsInstance(dd.level1, DotDict)
        self.assertIsInstance(dd.level1.level2, DotDict)
        self.assertEqual(dd.level1.level2.level3, 'value')

    def test_list_tuple_conversion(self):
        """Test conversion of lists and tuples containing dicts."""
        dd = DotDict({
            'list': [{'a': 1}, {'b': 2}],
            'tuple': ({'c': 3}, {'d': 4})
        })

        # List items should be DotDict
        self.assertIsInstance(dd.list[0], DotDict)
        self.assertEqual(dd.list[0].a, 1)
        self.assertIsInstance(dd.list[1], DotDict)
        self.assertEqual(dd.list[1].b, 2)

        # Tuple items should be DotDict
        self.assertIsInstance(dd.tuple[0], DotDict)
        self.assertEqual(dd.tuple[0].c, 3)

    def test_attribute_errors(self):
        """Test attribute error handling."""
        dd = DotDict({'a': 1})

        # Non-existent attribute
        with self.assertRaises(AttributeError) as ctx:
            _ = dd.nonexistent
        self.assertIn("has no attribute 'nonexistent'", str(ctx.exception))

        # Delete non-existent attribute
        with self.assertRaises(AttributeError):
            del dd.nonexistent

        # Try to set built-in method
        with self.assertRaises(AttributeError) as ctx:
            dd.keys = 'value'
        self.assertIn("built-in method", str(ctx.exception))

    def test_dir_method(self):
        """Test __dir__ method."""
        dd = DotDict({'a': 1, 'b': 2})
        attrs = dir(dd)

        self.assertIn('a', attrs)
        self.assertIn('b', attrs)
        self.assertIn('keys', attrs)  # Built-in method
        self.assertIn('values', attrs)  # Built-in method

    def test_get_method(self):
        """Test get method with default."""
        dd = DotDict({'a': 1})

        self.assertEqual(dd.get('a'), 1)
        self.assertEqual(dd.get('b'), None)
        self.assertEqual(dd.get('b', 'default'), 'default')

    def test_copy_operations(self):
        """Test copy operations."""
        dd1 = DotDict({'a': 1, 'b': {'c': 2}})

        # Shallow copy
        dd2 = dd1.copy()
        self.assertIsNot(dd2, dd1)
        self.assertEqual(dd2.a, 1)
        dd2.a = 10
        self.assertEqual(dd1.a, 1)  # Original unchanged

        # Deep copy
        dd3 = dd1.deepcopy()
        self.assertIsNot(dd3, dd1)
        self.assertIsNot(dd3.b, dd1.b)
        dd3.b.c = 20
        self.assertEqual(dd1.b.c, 2)  # Original unchanged

    def test_to_dict_from_dict(self):
        """Test conversion to/from standard dict."""
        dd1 = DotDict({
            'a': 1,
            'b': {'c': 2, 'd': {'e': 3}},
            'list': [{'f': 4}]
        })

        # Convert to dict
        d = dd1.to_dict()
        self.assertIsInstance(d, dict)
        self.assertNotIsInstance(d, DotDict)
        self.assertIsInstance(d['b'], dict)
        self.assertNotIsInstance(d['b'], DotDict)
        self.assertEqual(d['b']['d']['e'], 3)

        # Convert from dict
        dd2 = DotDict.from_dict(d)
        self.assertIsInstance(dd2, DotDict)
        self.assertIsInstance(dd2.b, DotDict)
        self.assertEqual(dd2.b.d.e, 3)

    def test_namedtuple_values(self):
        """Regression (H20): namedtuples must survive _hook and to_dict."""
        from collections import namedtuple
        Point = namedtuple('Point', ['x', 'y'])

        # Construction must not raise on a namedtuple value.
        dd = DotDict({'p': Point(1, 2)})
        self.assertIsInstance(dd.p, Point)
        self.assertEqual(dd.p, Point(1, 2))

        # Direct item assignment goes through __setitem__ -> _hook.
        dd['q'] = Point(3, 4)
        self.assertIsInstance(dd.q, Point)
        self.assertEqual(dd.q, Point(3, 4))

        # Nested dicts inside a namedtuple are converted to DotDict.
        dd2 = DotDict({'p': Point({'a': 1}, 2)})
        self.assertIsInstance(dd2.p, Point)
        self.assertIsInstance(dd2.p.x, DotDict)
        self.assertEqual(dd2.p.x.a, 1)

        # to_dict must mirror the same reconstruction (no TypeError, fields kept).
        d = dd2.to_dict()
        self.assertIsInstance(d['p'], Point)
        self.assertNotIsInstance(d['p'].x, DotDict)
        self.assertEqual(d['p'].x, {'a': 1})
        self.assertEqual(d['p'].y, 2)

        # Plain tuples and lists still round-trip.
        dd3 = DotDict({'t': ({'a': 1},), 'l': [{'b': 2}]})
        self.assertIsInstance(dd3.t, tuple)
        self.assertNotIsInstance(dd3.t, Point)
        self.assertIsInstance(dd3.t[0], DotDict)
        self.assertIsInstance(dd3.l, list)
        self.assertIsInstance(dd3.l[0], DotDict)
        d3 = dd3.to_dict()
        self.assertEqual(d3['t'], ({'a': 1},))
        self.assertEqual(d3['l'], [{'b': 2}])

    def test_update_accepts_iterable_of_pairs(self):
        """Regression (M44): update must accept an iterable of (k, v) pairs."""
        dd = DotDict({'a': 1})
        dd.update([('b', 2)])
        self.assertEqual(dict(dd), {'a': 1, 'b': 2})

        # Recursive-merge path still produces nested DotDict from pairs.
        dd2 = DotDict({'a': {'x': 1}})
        dd2.update([('a', {'y': 2})])
        self.assertIsInstance(dd2.a, DotDict)
        self.assertEqual(dd2.a.x, 1)
        self.assertEqual(dd2.a.y, 2)

        # Generators (an iterable of pairs) are accepted too.
        dd3 = DotDict()
        dd3.update((k, v) for k, v in [('c', 3), ('d', 4)])
        self.assertEqual(dict(dd3), {'c': 3, 'd': 4})

        # Bad input raises a clear TypeError, not an opaque AttributeError.
        with self.assertRaises(TypeError):
            DotDict().update(123)

    def test_parent_kwarg_no_longer_special(self):
        """Regression (L26): __parent/__key are ordinary keys, not reserved."""
        # Previously crashed inside __setitem__ with TypeError.
        dd = DotDict(__parent=5, a=1)
        self.assertEqual(set(dd.keys()), {'__parent', 'a'})
        self.assertEqual(dd['__parent'], 5)
        self.assertEqual(dd['a'], 1)

        # Behaves identically regardless of the supplied value.
        dd2 = DotDict(__parent='x', value=1)
        self.assertEqual(set(dd2.keys()), {'__parent', 'value'})
        self.assertEqual(dd2['__parent'], 'x')

    def test_update_method(self):
        """Test update with recursive merge."""
        dd = DotDict({'a': 1, 'b': {'c': 2, 'd': 3}})

        # Simple update
        dd.update({'a': 10})
        self.assertEqual(dd.a, 10)

        # Recursive merge
        dd.update({'b': {'d': 30, 'e': 4}})
        self.assertEqual(dd.b.c, 2)  # Preserved
        self.assertEqual(dd.b.d, 30)  # Updated
        self.assertEqual(dd.b.e, 4)  # Added

        # Update with kwargs
        dd.update(f=5, g=6)
        self.assertEqual(dd.f, 5)
        self.assertEqual(dd.g, 6)

        # Multiple arguments error
        with self.assertRaises(TypeError):
            dd.update({}, {})

    def test_setdefault(self):
        """Test setdefault method."""
        dd = DotDict({'a': 1})

        # Existing key
        result = dd.setdefault('a', 10)
        self.assertEqual(result, 1)
        self.assertEqual(dd.a, 1)

        # New key
        result = dd.setdefault('b', 2)
        self.assertEqual(result, 2)
        self.assertEqual(dd.b, 2)

        # New key with None default
        result = dd.setdefault('c')
        self.assertIsNone(result)
        self.assertIsNone(dd.c)

        result = dd.setdefault('nested', {'value': 3})
        self.assertIsInstance(result, DotDict)
        self.assertEqual(result.value, 3)

    def test_or_operator_preserves_type_and_hooks(self):
        """``DotDict | dict`` returns a DotDict and recursively hooks nested dicts.

        Without explicit ``__or__`` this delegates to ``dict.__or__`` and returns a
        plain ``dict`` (losing both the type and the nested-conversion hook).
        """
        dd = DotDict({'a': 1})
        result = dd | {'b': {'c': 2}}
        self.assertIsInstance(result, DotDict)
        self.assertIsInstance(result['b'], DotDict)
        self.assertEqual(result.b.c, 2)
        # original is unchanged
        self.assertNotIn('b', dd)
        # non-mapping right operand is not handled
        self.assertEqual(dd.__or__(123), NotImplemented)

    def test_ior_operator_preserves_type_and_hooks(self):
        """``DotDict |= dict`` updates in place via the hook-aware ``update``."""
        dd = DotDict({'a': 1})
        dd |= {'b': {'c': 2}}
        self.assertIsInstance(dd, DotDict)
        self.assertIsInstance(dd['b'], DotDict)
        self.assertEqual(dd.b.c, 2)

    def test_ror_operator_with_generic_mapping(self):
        """``mapping | DotDict`` (non-dict left operand) yields a hooked DotDict."""
        import collections.abc as abc

        class ReadOnlyMapping(abc.Mapping):
            def __init__(self, data):
                self._data = dict(data)

            def __getitem__(self, key):
                return self._data[key]

            def __iter__(self):
                return iter(self._data)

            def __len__(self):
                return len(self._data)

        result = ReadOnlyMapping({'b': {'c': 2}}) | DotDict({'a': 1})
        self.assertIsInstance(result, DotDict)
        self.assertIsInstance(result['b'], DotDict)
        self.assertEqual(result.a, 1)
        self.assertEqual(result.b.c, 2)

    def test_ror_operator_with_non_mapping_returns_notimplemented(self):
        """``__ror__`` returns ``NotImplemented`` for a non-Mapping left operand.

        This lets Python fall through to the other operand / raise ``TypeError``
        rather than mis-handling a non-mapping. Exercises the guard at the top of
        ``DotDict.__ror__``.
        """
        dd = DotDict({'a': 1})
        self.assertIs(dd.__ror__(5), NotImplemented)
        self.assertIs(dd.__ror__("not a mapping"), NotImplemented)
        # As a consequence, ``non_mapping | dotdict`` raises TypeError.
        with self.assertRaises(TypeError):
            _ = 5 | dd

    def test_ior_operator_with_non_mapping_returns_notimplemented(self):
        """``__ior__`` returns ``NotImplemented`` for a non-Mapping right operand.

        Exercises the guard at the top of ``DotDict.__ior__``; the original
        DotDict must be left untouched.
        """
        dd = DotDict({'a': 1})
        self.assertIs(dd.__ior__(5), NotImplemented)
        self.assertEqual(dict(dd), {'a': 1})  # unchanged
        # ``dotdict |= non_mapping`` therefore raises TypeError.
        with self.assertRaises(TypeError):
            dd |= 5

    def test_pickling(self):
        """Test pickling/unpickling."""
        dd1 = DotDict({'a': 1, 'b': {'c': 2}})

        # Pickle and unpickle
        pickled = pickle.dumps(dd1)
        dd2 = pickle.loads(pickled)

        self.assertIsNot(dd2, dd1)
        self.assertEqual(dd2.a, 1)
        self.assertEqual(dd2.b.c, 2)
        self.assertIsInstance(dd2, DotDict)
        self.assertIsInstance(dd2.b, DotDict)

    def test_jax_pytree(self):
        """Test JAX pytree registration."""
        dd = DotDict({'a': jnp.array([1, 2]), 'b': jnp.array([3, 4])})

        # Tree operations
        doubled = jax.tree_util.tree_map(lambda x: x * 2, dd)
        self.assertTrue(jnp.allclose(doubled.a, jnp.array([2, 4])))
        self.assertTrue(jnp.allclose(doubled.b, jnp.array([6, 8])))

    def test_repr(self):
        """Test string representation."""
        dd = DotDict({'a': 1, 'b': 'text'})
        repr_str = repr(dd)
        self.assertIn('DotDict', repr_str)
        self.assertIn("'a': 1", repr_str)
        self.assertIn("'b': 'text'", repr_str)


class TestUtilityFunctions(unittest.TestCase):
    """Test cases for utility functions."""

    def test_merge_dicts_basic(self):
        """Test basic dict merging."""
        d1 = {'a': 1, 'b': 2}
        d2 = {'b': 3, 'c': 4}
        d3 = {'d': 5}

        result = merge_dicts(d1, d2, d3)
        self.assertEqual(result, {'a': 1, 'b': 3, 'c': 4, 'd': 5})

        # Original dicts should be unchanged
        self.assertEqual(d1, {'a': 1, 'b': 2})

    def test_merge_dicts_recursive(self):
        """Test recursive dict merging."""
        d1 = {'a': 1, 'b': {'c': 2, 'd': 3}}
        d2 = {'b': {'d': 4, 'e': 5}, 'f': 6}

        result = merge_dicts(d1, d2, recursive=True)
        self.assertEqual(result, {
            'a': 1,
            'b': {'c': 2, 'd': 4, 'e': 5},
            'f': 6
        })

    def test_merge_dicts_non_recursive(self):
        """Test non-recursive dict merging."""
        d1 = {'a': 1, 'b': {'c': 2}}
        d2 = {'b': {'d': 3}}

        result = merge_dicts(d1, d2, recursive=False)
        self.assertEqual(result, {'a': 1, 'b': {'d': 3}})

    def test_merge_dicts_errors(self):
        """Test merge_dicts error handling."""
        with self.assertRaises(TypeError):
            merge_dicts({'a': 1}, [1, 2, 3])

    def test_flatten_dict(self):
        """Test dictionary flattening."""
        nested = {
            'a': 1,
            'b': {
                'c': 2,
                'd': {
                    'e': 3,
                    'f': 4
                }
            },
            'g': 5
        }

        flat = flatten_dict(nested)
        self.assertEqual(flat, {
            'a': 1,
            'b.c': 2,
            'b.d.e': 3,
            'b.d.f': 4,
            'g': 5
        })

        # Custom separator
        flat_dash = flatten_dict(nested, sep='-')
        self.assertEqual(flat_dash, {
            'a': 1,
            'b-c': 2,
            'b-d-e': 3,
            'b-d-f': 4,
            'g': 5
        })

    def test_unflatten_dict(self):
        """Test dictionary unflattening."""
        flat = {
            'a': 1,
            'b.c': 2,
            'b.d.e': 3,
            'b.d.f': 4,
            'g': 5
        }

        nested = unflatten_dict(flat)
        self.assertEqual(nested, {
            'a': 1,
            'b': {
                'c': 2,
                'd': {
                    'e': 3,
                    'f': 4
                }
            },
            'g': 5
        })

        # Custom separator
        flat_dash = {'a': 1, 'b-c': 2, 'b-d': 3}
        nested_dash = unflatten_dict(flat_dash, sep='-')
        self.assertEqual(nested_dash, {
            'a': 1,
            'b': {'c': 2, 'd': 3}
        })

    def test_unflatten_dict_rejects_prefix_conflicts(self):
        """Reject flattened inputs that would overwrite a scalar with a mapping."""
        with self.assertRaises(ValueError):
            unflatten_dict({'a': 1, 'a.b': 2})

        with self.assertRaises(ValueError):
            unflatten_dict({'a.b': 2, 'a': 1})

    def test_flatten_dict_rejects_non_mapping(self):
        """Raise a clear TypeError for non-dictionary inputs."""
        with self.assertRaises(TypeError):
            flatten_dict([('a', 1)])

    def test_flatten_unflatten_roundtrip(self):
        """Test that flatten/unflatten is reversible."""
        original = {
            'level1': {
                'level2': {
                    'level3': 'value',
                    'another': 42
                },
                'sibling': 'data'
            },
            'root': 'element'
        }

        flattened = flatten_dict(original)
        unflattened = unflatten_dict(flattened)
        self.assertEqual(unflattened, original)

    def test_flatten_dict_preserves_empty_subdicts(self):
        """Regression (M43): empty sub-dicts must not be dropped on flatten."""
        # Top-level empty dict.
        self.assertEqual(flatten_dict({'a': 1, 'b': {}}), {'a': 1, 'b': {}})
        # Entirely-empty content.
        self.assertEqual(flatten_dict({'b': {}}), {'b': {}})

        # Round-trip with empty sub-dicts at top level and nested.
        for x in (
            {'a': 1, 'b': {}},
            {'b': {'c': {}}},
            {'a': 1, 'b': {'c': {}, 'd': 2}},
        ):
            self.assertEqual(unflatten_dict(flatten_dict(x)), x)

    def test_unflatten_dict_copies_dict_leaves(self):
        """Regression (M43): dict leaf values must not alias the input."""
        flat = {'b': {}}
        out = unflatten_dict(flat)
        out['b']['mutated'] = 1
        self.assertEqual(flat, {'b': {}})  # input unchanged

    def test_flatten_dict_rejects_key_collision(self):
        """Regression (L25): colliding joined keys must raise, not silently drop."""
        with self.assertRaises(ValueError):
            flatten_dict({'a.b': 1, 'a': {'b': 2}})
        with self.assertRaises(ValueError):
            flatten_dict({'a': {'b': 2}, 'a.b': 1})

    def test_flatten_dict_rejects_empty_subdict_key_collision(self):
        """An *empty* sub-dict whose emitted key collides with an already-present
        key must also raise, not silently overwrite.

        Empty sub-dicts take a dedicated branch (they emit a ``{}`` placeholder so
        flatten/unflatten round-trips). That branch must still honour the
        duplicate-key guard: here a non-empty sibling ``'a'`` first emits ``'a.b'``,
        then the empty sibling ``'a.b': {}`` would re-emit the same joined key.
        """
        with self.assertRaisesRegex(ValueError, "duplicate key 'a.b'"):
            flatten_dict({'a': {'b': 1}, 'a.b': {}})

    def test_flatten_dict_rejects_non_string_nested_keys(self):
        """Regression (L25): non-string nested keys must raise, not coerce."""
        with self.assertRaises(TypeError):
            flatten_dict({1: 'a', 2: {3: 'b'}})
        # A non-string top-level key (no nesting) is preserved verbatim.
        self.assertEqual(flatten_dict({1: 'a', 2: 'b'}), {1: 'a', 2: 'b'})

    def test_is_instance_eval(self):
        """Test is_instance_eval function."""
        # Single type
        is_int = is_instance_eval(int)
        self.assertTrue(is_int(5))
        self.assertFalse(is_int("5"))
        self.assertFalse(is_int(5.0))

        # Multiple types
        is_number = is_instance_eval(int, float)
        self.assertTrue(is_number(5))
        self.assertTrue(is_number(5.0))
        self.assertFalse(is_number("5"))

        # With subclasses
        class MyList(list):
            pass

        is_list = is_instance_eval(list)
        self.assertTrue(is_list([1, 2, 3]))
        self.assertTrue(is_list(MyList([1, 2, 3])))
        self.assertFalse(is_list((1, 2, 3)))

    def test_not_instance_eval(self):
        """Test not_instance_eval function."""
        # Single type
        not_int = not_instance_eval(int)
        self.assertFalse(not_int(5))
        self.assertTrue(not_int("5"))
        self.assertTrue(not_int(5.0))

        # Multiple types
        not_number = not_instance_eval(int, float)
        self.assertFalse(not_number(5))
        self.assertFalse(not_number(5.0))
        self.assertTrue(not_number("5"))
        self.assertTrue(not_number([1, 2, 3]))


class TestJaxIntegration(unittest.TestCase):
    """Test JAX integration for DictManager and DotDict."""

    def test_dictmanager_pytree_operations(self):
        """Test DictManager with JAX tree operations."""
        dm = DictManager({
            'weights': jnp.array([1.0, 2.0, 3.0]),
            'bias': jnp.array([0.1, 0.2])
        })

        # Tree map
        scaled = jax.tree_util.tree_map(lambda x: x * 2, dm)
        self.assertTrue(jnp.allclose(scaled['weights'], jnp.array([2.0, 4.0, 6.0])))
        self.assertTrue(jnp.allclose(scaled['bias'], jnp.array([0.2, 0.4])))

        # Tree reduce
        total = jax.tree_util.tree_reduce(lambda x, y: x + y.sum(), dm, 0.0)
        self.assertAlmostEqual(total, 6.3, places=5)

    def test_dotdict_pytree_operations(self):
        """Test DotDict with JAX tree operations."""
        dd = DotDict({
            'model': {
                'weights': jnp.array([[1.0, 2.0], [3.0, 4.0]]),
                'bias': jnp.array([0.1, 0.2])
            },
            'optimizer': {
                'lr': 0.01
            }
        })

        # Tree map on nested structure
        def scale_arrays(x):
            return x * 2 if isinstance(x, jnp.ndarray) else x

        scaled = jax.tree_util.tree_map(scale_arrays, dd)
        self.assertTrue(jnp.allclose(
            scaled.model.weights,
            jnp.array([[2.0, 4.0], [6.0, 8.0]])
        ))
        self.assertEqual(scaled.optimizer.lr, 0.01)  # Non-array unchanged

    def test_mixed_pytree_structures(self):
        """Test mixing DictManager and DotDict in pytree operations."""
        structure = {
            'dict_manager': DictManager({'a': jnp.array([1, 2])})
        }

        doubled = jax.tree_util.tree_map(lambda x: x * 2, structure)
        self.assertTrue(jnp.allclose(
            doubled['dict_manager']['a'],
            jnp.array([2, 4])
        ))


class TestClearBufferMemory(unittest.TestCase):
    """Test cases for clear_buffer_memory error handling (item 3)."""

    def test_runs_without_error_when_buffers_delete_cleanly(self):
        """Complete normally when every live buffer deletes cleanly (happy path).

        Mocks the backend instead of clearing the real one: a real
        ``clear_buffer_memory()`` deletes JAX's process-global device buffers,
        including the ordered-effect runtime token that ``jit_error_if`` relies
        on. Tearing that down corrupts unrelated tests later in the same process
        (``BlockHostUntilReady() called on deleted or donated buffer``). The real
        deletion loop and its error handling are covered by the mocked tests below.
        """
        buf1, buf2 = MagicMock(), MagicMock()
        backend = MagicMock()
        backend.live_buffers.return_value = [buf1, buf2]
        with patch("brainstate._compatible_import.get_backend", return_value=backend):
            clear_buffer_memory()  # should not raise
        buf1.delete.assert_called_once()
        buf2.delete.assert_called_once()

    def test_per_buffer_delete_failure_warns_and_continues(self):
        """Warn per failing buffer deletion but still delete the others.

        Previously every error was swallowed under a single warning; now each
        ``delete()`` failure is isolated so one bad buffer cannot abort the loop.
        """
        good = MagicMock()
        bad = MagicMock()
        bad.delete.side_effect = RuntimeError("cannot free")
        another = MagicMock()
        backend = MagicMock()
        backend.live_buffers.return_value = [good, bad, another]

        with patch("brainstate._compatible_import.get_backend", return_value=backend):
            with self.assertWarns(RuntimeWarning):
                clear_buffer_memory()

        # the failing buffer did not prevent the others from being deleted
        good.delete.assert_called_once()
        another.delete.assert_called_once()

    def test_backend_acquisition_error_is_not_swallowed(self):
        """Surface setup errors (e.g. unknown platform) instead of hiding them.

        The old implementation caught *all* exceptions, masking a genuine
        misconfiguration; backend acquisition failures should now propagate.
        """
        with patch(
            "brainstate._compatible_import.get_backend",
            side_effect=RuntimeError("unknown platform"),
        ):
            with self.assertRaises(RuntimeError):
                clear_buffer_memory(platform="definitely_not_a_platform")

    def test_array_false_skips_buffer_clearing(self):
        """Skip buffer clearing entirely when ``array=False``."""
        backend = MagicMock()
        with patch("brainstate._compatible_import.get_backend", return_value=backend) as gb:
            clear_buffer_memory(array=False)
        gb.assert_not_called()


if __name__ == '__main__':
    unittest.main()
