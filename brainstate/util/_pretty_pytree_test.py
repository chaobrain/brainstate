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
Comprehensive test suite for the pretty_pytree module.

This test module provides extensive coverage of the pretty printing and tree
manipulation functionality, including:
- PrettyObject and pretty representation
- Nested and flattened dictionary structures
- Mapping flattening and unflattening
- Split, filter, and merge operations
- JAX pytree integration
- State management utilities
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

import brainstate
from brainstate.util._pretty_pytree import (
    PrettyObject,
    PrettyDict,
    NestedDict,
    FlattedDict,
    PrettyList,
    flat_mapping,
    nest_mapping,
    empty_node,
    _EmptyNode,
)


class TestNestedMapping(absltest.TestCase):
    def test_create_state(self):
        state = brainstate.util.NestedDict({'a': brainstate.ParamState(1), 'b': {'c': brainstate.ParamState(2)}})

        assert state['a'].value == 1
        assert state['b']['c'].value == 2

    def test_get_attr(self):
        state = brainstate.util.NestedDict({'a': brainstate.ParamState(1), 'b': {'c': brainstate.ParamState(2)}})

        assert state.a.value == 1
        assert state.b['c'].value == 2

    def test_set_attr(self):
        state = brainstate.util.NestedDict({'a': brainstate.ParamState(1), 'b': {'c': brainstate.ParamState(2)}})

        state.a.value = 3
        state.b['c'].value = 4

        assert state['a'].value == 3
        assert state['b']['c'].value == 4

    def test_set_attr_variables(self):
        state = brainstate.util.NestedDict({'a': brainstate.ParamState(1), 'b': {'c': brainstate.ParamState(2)}})

        state.a.value = 3
        state.b['c'].value = 4

        assert isinstance(state.a, brainstate.ParamState)
        assert state.a.value == 3
        assert isinstance(state.b['c'], brainstate.ParamState)
        assert state.b['c'].value == 4

    def test_add_nested_attr(self):
        state = brainstate.util.NestedDict({'a': brainstate.ParamState(1), 'b': {'c': brainstate.ParamState(2)}})
        state.b['d'] = brainstate.ParamState(5)

        assert state['b']['d'].value == 5

    def test_delete_nested_attr(self):
        state = brainstate.util.NestedDict({'a': brainstate.ParamState(1), 'b': {'c': brainstate.ParamState(2)}})
        del state['b']['c']

        assert 'c' not in state['b']

    def test_integer_access(self):
        class Foo(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = [brainstate.nn.Linear(1, 2), brainstate.nn.Linear(2, 3)]

        module = Foo()
        state_refs = brainstate.graph.treefy_states(module)

        assert module.layers[0].weight.value['weight'].shape == (1, 2)
        assert state_refs.layers[0]['weight'].value['weight'].shape == (1, 2)
        assert module.layers[1].weight.value['weight'].shape == (2, 3)
        assert state_refs.layers[1]['weight'].value['weight'].shape == (2, 3)

    def test_pure_dict(self):
        module = brainstate.nn.Linear(4, 5)
        state_map = brainstate.graph.treefy_states(module)
        pure_dict = state_map.to_pure_dict()
        assert isinstance(pure_dict, dict)
        assert isinstance(pure_dict['weight'].value['weight'], jax.Array)
        assert isinstance(pure_dict['weight'].value['bias'], jax.Array)


class TestSplit(unittest.TestCase):
    def test_split(self):
        class Model(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.batchnorm = brainstate.nn.BatchNorm1d([10, 3])
                self.linear = brainstate.nn.Linear([10, 3], [10, 4])

            def __call__(self, x):
                return self.linear(self.batchnorm(x))

        with brainstate.environ.context(fit=True):
            model = Model()
            x = brainstate.random.randn(1, 10, 3)
            y = model(x)
            self.assertEqual(y.shape, (1, 10, 4))

        state_map = brainstate.graph.treefy_states(model)

        with self.assertRaises(ValueError):
            params, others = state_map.split(brainstate.ParamState)

        params, others = state_map.split(brainstate.ParamState, ...)
        print()
        print(params)
        print(others)

        self.assertTrue(len(params.to_flat()) == 2)
        self.assertTrue(len(others.to_flat()) == 2)


class TestStateMap2(unittest.TestCase):
    def test1(self):
        class Model(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.batchnorm = brainstate.nn.BatchNorm1d([10, 3])
                self.linear = brainstate.nn.Linear([10, 3], [10, 4])

            def __call__(self, x):
                return self.linear(self.batchnorm(x))

        with brainstate.environ.context(fit=True):
            model = Model()
            state_map = brainstate.graph.treefy_states(model).to_flat()
            state_map = brainstate.util.NestedDict(state_map)


class TestFlattedMapping(unittest.TestCase):
    def test1(self):
        class Model(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.batchnorm = brainstate.nn.BatchNorm1d([10, 3])
                self.linear = brainstate.nn.Linear([10, 3], [10, 4])

            def __call__(self, x):
                return self.linear(self.batchnorm(x))

        model = Model()
        # print(model.states())
        # print(brainstate.graph.states(model))
        self.assertTrue(model.states() == brainstate.graph.states(model))

        print(model.nodes())
        # print(brainstate.graph.nodes(model))
        self.assertTrue(model.nodes() == brainstate.graph.nodes(model))


class TestPrettyObject(unittest.TestCase):
    """Test PrettyObject functionality."""

    def test_pretty_object_basic(self):
        """Test basic PrettyObject creation and representation."""
        class MyObject(PrettyObject):
            def __init__(self, value):
                self.value = value
                self.name = "test"

        obj = MyObject(42)
        repr_str = repr(obj)
        self.assertIsInstance(repr_str, str)
        self.assertIn("MyObject", repr_str)
        self.assertIn("value", repr_str)
        self.assertIn("42", repr_str)

    def test_pretty_repr_item_filtering(self):
        """Test __pretty_repr_item__ filtering."""
        class FilteredObject(PrettyObject):
            def __init__(self):
                self.visible = "show"
                self.hidden = "hide"

            def __pretty_repr_item__(self, k, v):
                if k == "hidden":
                    return None
                return k, v

        obj = FilteredObject()
        repr_str = repr(obj)
        self.assertIn("visible", repr_str)
        self.assertNotIn("hidden", repr_str)

    def test_pretty_repr_item_transformation(self):
        """Test __pretty_repr_item__ value transformation."""
        class TransformObject(PrettyObject):
            def __init__(self):
                self.value = 100

            def __pretty_repr_item__(self, k, v):
                if k == "value":
                    return k, v * 2
                return k, v

        obj = TransformObject()
        repr_str = repr(obj)
        self.assertIn("200", repr_str)


class TestFlatAndNestMapping(unittest.TestCase):
    """Test flat_mapping and nest_mapping functions."""

    def test_flat_mapping_basic(self):
        """Test basic flattening of nested dict."""
        nested = {'a': 1, 'b': {'c': 2, 'd': 3}}
        flat = flat_mapping(nested)

        self.assertIsInstance(flat, FlattedDict)
        self.assertEqual(flat[('a',)], 1)
        self.assertEqual(flat[('b', 'c')], 2)
        self.assertEqual(flat[('b', 'd')], 3)

    def test_flat_mapping_with_separator(self):
        """Test flattening with string separator."""
        nested = {'a': 1, 'b': {'c': 2}}
        flat = flat_mapping(nested, sep='/')

        self.assertEqual(flat['a'], 1)
        self.assertEqual(flat['b/c'], 2)

    def test_flat_mapping_empty_nodes(self):
        """Test flattening with keep_empty_nodes."""
        nested = {'a': 1, 'b': {}}
        flat = flat_mapping(nested, keep_empty_nodes=True)

        self.assertEqual(flat[('a',)], 1)
        self.assertIsInstance(flat[('b',)], _EmptyNode)

    def test_flat_mapping_without_empty_nodes(self):
        """Test flattening without keeping empty nodes."""
        nested = {'a': 1, 'b': {}}
        flat = flat_mapping(nested, keep_empty_nodes=False)

        self.assertIn(('a',), flat)
        self.assertNotIn(('b',), flat)

    def test_flat_mapping_is_leaf(self):
        """Test flattening with custom is_leaf function."""
        nested = {'a': 1, 'b': {'c': 2, 'd': 3}}

        def is_leaf(path, value):
            return len(path) >= 1

        flat = flat_mapping(nested, is_leaf=is_leaf)
        self.assertEqual(flat[('a',)], 1)
        self.assertEqual(flat[('b',)], {'c': 2, 'd': 3})

    def test_nest_mapping_basic(self):
        """Test basic unflattening."""
        flat = {('a',): 1, ('b', 'c'): 2, ('b', 'd'): 3}
        nested = nest_mapping(flat)

        self.assertIsInstance(nested, NestedDict)
        self.assertEqual(nested['a'], 1)
        self.assertEqual(nested['b']['c'], 2)
        self.assertEqual(nested['b']['d'], 3)

    def test_nest_mapping_with_separator(self):
        """Test unflattening with string separator."""
        flat = {'a': 1, 'b/c': 2, 'b/d': 3}
        nested = nest_mapping(flat, sep='/')

        self.assertEqual(nested['a'], 1)
        self.assertEqual(nested['b']['c'], 2)
        self.assertEqual(nested['b']['d'], 3)

    def test_nest_mapping_with_empty_node(self):
        """Test unflattening with empty nodes."""
        flat = {('a',): 1, ('b',): empty_node}
        nested = nest_mapping(flat)

        self.assertEqual(nested['a'], 1)
        self.assertEqual(nested['b'], {})

    def test_round_trip(self):
        """Test flatten -> unflatten round trip."""
        original = {'a': 1, 'b': {'c': 2, 'd': {'e': 3}}}
        flat = flat_mapping(original)
        restored = nest_mapping(flat)

        self.assertEqual(restored.to_dict(), original)


class TestPrettyDict(unittest.TestCase):
    """Test PrettyDict functionality."""

    def test_pretty_dict_creation(self):
        """Test PrettyDict creation."""
        d = PrettyDict({'a': 1, 'b': 2})
        self.assertEqual(d['a'], 1)
        self.assertEqual(d['b'], 2)

    def test_pretty_dict_attribute_access(self):
        """Test accessing dict items as attributes."""
        d = PrettyDict({'a': 1, 'b': 2})
        self.assertEqual(d.a, 1)
        self.assertEqual(d.b, 2)

    def test_pretty_dict_repr(self):
        """Test PrettyDict representation."""
        d = PrettyDict({'a': 1, 'b': 2})
        repr_str = repr(d)
        self.assertIsInstance(repr_str, str)
        self.assertIn('a', repr_str)

    def test_to_dict(self):
        """Test conversion to regular dict."""
        d = PrettyDict({'a': 1, 'b': 2})
        regular = d.to_dict()
        self.assertIsInstance(regular, dict)
        self.assertEqual(regular, {'a': 1, 'b': 2})


class TestNestedDictOperations(unittest.TestCase):
    """Test NestedDict additional operations."""

    def test_or_operator(self):
        """Test | operator for merging."""
        d1 = NestedDict({'a': 1})
        d2 = NestedDict({'b': 2})
        merged = d1 | d2

        self.assertIsInstance(merged, NestedDict)
        self.assertEqual(merged['a'], 1)
        self.assertEqual(merged['b'], 2)

    def test_sub_operator(self):
        """Test - operator for difference."""
        d1 = NestedDict({'a': 1, 'b': 2, 'c': 3})
        d2 = NestedDict({'b': 2})
        diff = d1 - d2

        flat_diff = diff.to_flat()
        self.assertIn(('a',), flat_diff.keys())
        self.assertIn(('c',), flat_diff.keys())
        # b should not be in diff
        has_b = any('b' in key for key in flat_diff.keys())
        self.assertFalse(has_b)

    def test_merge_static_method(self):
        """Test static merge method."""
        d1 = NestedDict({'a': 1})
        d2 = NestedDict({'b': 2})
        d3 = NestedDict({'c': 3})
        merged = NestedDict.merge(d1, d2, d3)

        self.assertEqual(merged['a'], 1)
        self.assertEqual(merged['b'], 2)
        self.assertEqual(merged['c'], 3)

    def test_to_pure_dict(self):
        """Test conversion to pure dict."""
        nested = NestedDict({'a': 1, 'b': {'c': 2}})
        pure = nested.to_pure_dict()

        self.assertIsInstance(pure, dict)
        self.assertNotIsInstance(pure, NestedDict)
        self.assertEqual(pure['a'], 1)
        self.assertEqual(pure['b']['c'], 2)


class TestFlattedDictOperations(unittest.TestCase):
    """Test FlattedDict additional operations."""

    def test_or_operator(self):
        """Test | operator for merging."""
        d1 = FlattedDict({('a',): 1})
        d2 = FlattedDict({('b',): 2})
        merged = d1 | d2

        self.assertIsInstance(merged, FlattedDict)
        self.assertEqual(merged[('a',)], 1)
        self.assertEqual(merged[('b',)], 2)

    def test_sub_operator(self):
        """Test - operator for difference."""
        d1 = FlattedDict({('a',): 1, ('b',): 2, ('c',): 3})
        d2 = FlattedDict({('b',): 2})
        diff = d1 - d2

        self.assertIn(('a',), diff)
        self.assertIn(('c',), diff)
        self.assertNotIn(('b',), diff)

    def test_merge_static_method(self):
        """Test static merge method."""
        d1 = FlattedDict({('a',): 1})
        d2 = FlattedDict({('b',): 2})
        merged = FlattedDict.merge(d1, d2)

        self.assertEqual(merged[('a',)], 1)
        self.assertEqual(merged[('b',)], 2)

    def test_to_dict_values(self):
        """Test conversion to dictionary of values."""
        flat = FlattedDict({
            ('a',): brainstate.ParamState(jnp.array([1, 2, 3])),
            ('b',): 42
        })
        values = flat.to_dict_values()

        self.assertIsInstance(values[('a',)], jnp.ndarray)
        np.testing.assert_array_equal(values[('a',)], jnp.array([1, 2, 3]))
        self.assertEqual(values[('b',)], 42)

    def test_assign_dict_values(self):
        """Test assigning dictionary values."""
        flat = FlattedDict({
            ('a',): brainstate.ParamState(jnp.array([1, 2, 3])),
            ('b',): 42
        })

        new_values = {
            ('a',): jnp.array([4, 5, 6]),
            ('b',): 100
        }

        flat.assign_dict_values(new_values)

        np.testing.assert_array_equal(flat[('a',)].value, jnp.array([4, 5, 6]))
        self.assertEqual(flat[('b',)], 100)

    def test_assign_dict_values_missing_key(self):
        """Test assigning with missing key raises error."""
        flat = FlattedDict({('a',): 1})

        with self.assertRaises(KeyError):
            flat.assign_dict_values({('b',): 2})


class TestPrettyList(unittest.TestCase):
    """Test PrettyList functionality."""

    def test_pretty_list_creation(self):
        """Test PrettyList creation."""
        lst = PrettyList([1, 2, 3])
        self.assertEqual(lst[0], 1)
        self.assertEqual(lst[1], 2)
        self.assertEqual(lst[2], 3)

    def test_pretty_list_repr(self):
        """Test PrettyList representation."""
        lst = PrettyList([1, 2, {'a': 3}])
        repr_str = repr(lst)
        self.assertIsInstance(repr_str, str)
        self.assertIn('1', repr_str)

    def test_tree_flatten(self):
        """Test JAX tree flattening."""
        lst = PrettyList([1, 2, 3])
        leaves, aux = lst.tree_flatten()
        self.assertEqual(leaves, [1, 2, 3])
        self.assertEqual(aux, ())

    def test_tree_unflatten(self):
        """Test JAX tree unflattening."""
        children = [1, 2, 3]
        lst = PrettyList.tree_unflatten((), children)
        self.assertIsInstance(lst, PrettyList)
        self.assertEqual(list(lst), [1, 2, 3])


class TestFilterOperations(unittest.TestCase):
    """Test filter operations."""

    def test_nested_dict_filter(self):
        """Test filtering NestedDict."""
        nested = NestedDict({
            'a': 1,
            'b': 2,
            'c': 3
        })

        filtered = nested.filter(lambda path, val: val >= 2)

        flat = filtered.to_flat()
        # Check that filtered values are present
        values = [v for v in flat.values()]
        self.assertIn(2, values)
        self.assertIn(3, values)

    def test_flatted_dict_filter(self):
        """Test filtering FlattedDict."""
        flat = FlattedDict({
            ('a',): 1,
            ('b',): 2,
            ('c',): 3
        })

        filtered = flat.filter(lambda path, val: val % 2 == 0)
        self.assertIn(('b',), filtered)
        self.assertNotIn(('a',), filtered)

    def test_ellipsis_filter_position(self):
        """Test that ... can only be used as last filter."""
        nested = NestedDict({'a': 1, 'b': 2, 'c': 3})

        with self.assertRaises(ValueError):
            # ... in middle should raise error
            nested.split(..., lambda path, val: val > 1)


class TestJAXPytreeIntegration(unittest.TestCase):
    """Test JAX pytree integration."""

    def test_nested_dict_pytree_flatten(self):
        """Test NestedDict can be flattened as pytree."""
        nested = NestedDict({'a': 1, 'b': 2})
        leaves, treedef = jax.tree.flatten(nested)

        self.assertEqual(sorted(leaves), [1, 2])

    def test_nested_dict_pytree_unflatten(self):
        """Test NestedDict can be unflattened as pytree."""
        nested = NestedDict({'a': 1, 'b': 2})
        leaves, treedef = jax.tree.flatten(nested)
        restored = jax.tree.unflatten(treedef, leaves)

        self.assertIsInstance(restored, NestedDict)
        self.assertEqual(restored['a'], 1)
        self.assertEqual(restored['b'], 2)

    def test_flatted_dict_pytree_flatten(self):
        """Test FlattedDict can be flattened as pytree."""
        flat = FlattedDict({('a',): 1, ('b',): 2})
        leaves, treedef = jax.tree.flatten(flat)

        self.assertEqual(sorted(leaves), [1, 2])

    def test_flatted_dict_pytree_unflatten(self):
        """Test FlattedDict can be unflattened as pytree."""
        flat = FlattedDict({('a',): 1, ('b',): 2})
        leaves, treedef = jax.tree.flatten(flat)
        restored = jax.tree.unflatten(treedef, leaves)

        self.assertIsInstance(restored, FlattedDict)
        self.assertEqual(restored[('a',)], 1)

    def test_pretty_list_pytree(self):
        """Test PrettyList pytree operations."""
        lst = PrettyList([1, 2, 3])
        leaves, treedef = jax.tree.flatten(lst)
        restored = jax.tree.unflatten(treedef, leaves)

        self.assertIsInstance(restored, PrettyList)
        self.assertEqual(list(restored), [1, 2, 3])

    def test_jax_tree_map_nested_dict(self):
        """Test jax.tree.map with NestedDict."""
        nested = NestedDict({'a': 1, 'b': {'c': 2}})
        doubled = jax.tree.map(lambda x: x * 2, nested)

        self.assertEqual(doubled['a'], 2)
        self.assertEqual(doubled['b']['c'], 4)

    def test_jax_tree_map_flatted_dict(self):
        """Test jax.tree.map with FlattedDict."""
        flat = FlattedDict({('a',): 1, ('b', 'c'): 2})
        doubled = jax.tree.map(lambda x: x * 2, flat)

        self.assertEqual(doubled[('a',)], 2)
        self.assertEqual(doubled[('b', 'c')], 4)

    def test_jax_tree_map_pretty_list(self):
        """Test jax.tree.map with PrettyList."""
        lst = PrettyList([1, 2, 3])
        doubled = jax.tree.map(lambda x: x * 2, lst)

        self.assertEqual(list(doubled), [2, 4, 6])


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def test_empty_nested_dict(self):
        """Test empty NestedDict."""
        nested = NestedDict({})
        flat = nested.to_flat()
        self.assertEqual(len(flat), 0)

    def test_empty_flatted_dict(self):
        """Test empty FlattedDict."""
        flat = FlattedDict({})
        nested = flat.to_nest()
        self.assertEqual(len(nested), 0)

    def test_deeply_nested_structure(self):
        """Test deeply nested structure."""
        nested = NestedDict({
            'a': {
                'b': {
                    'c': {
                        'd': {
                            'e': 42
                        }
                    }
                }
            }
        })
        flat = nested.to_flat()
        self.assertEqual(flat[('a', 'b', 'c', 'd', 'e')], 42)

    def test_mixed_types_in_nested(self):
        """Test nested dict with mixed types."""
        nested = NestedDict({
            'int': 1,
            'float': 2.5,
            'str': 'hello',
            'list': [1, 2, 3],
            'dict': {'nested': True}
        })
        flat = nested.to_flat()

        self.assertEqual(flat[('int',)], 1)
        self.assertEqual(flat[('float',)], 2.5)
        self.assertEqual(flat[('str',)], 'hello')

    def test_numeric_keys(self):
        """Test handling of numeric keys."""
        nested = NestedDict({
            1: 'one',
            2: {'a': 'two-a'}
        })
        flat = nested.to_flat()

        self.assertEqual(flat[(1,)], 'one')
        self.assertEqual(flat[(2, 'a')], 'two-a')

    def test_merge_with_overlapping_keys(self):
        """Test merging with overlapping keys."""
        d1 = NestedDict({'a': 1, 'b': 2})
        d2 = NestedDict({'b': 3, 'c': 4})
        merged = NestedDict.merge(d1, d2)

        # Later values should override
        self.assertEqual(merged['b'], 3)
        self.assertEqual(merged['a'], 1)
        self.assertEqual(merged['c'], 4)
