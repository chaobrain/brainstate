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
    NestedStateRepr,
    flat_mapping,
    nest_mapping,
    empty_node,
    _EmptyNode,
    _default_compare,
    _default_process,
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


class TestGetattrProtocol(absltest.TestCase):
    """``__getattr__`` honours the attribute protocol for dunder probes.

    Regression test: ``__getattr__`` used to ``return self[key]`` for every
    name, so probing a missing dunder (e.g. ``__deepcopy__``) raised
    ``KeyError`` instead of ``AttributeError``. That broke ``copy.deepcopy``,
    ``pickle``, and ``hasattr`` on any :class:`PrettyDict`.
    """

    def test_missing_dunder_raises_attribute_error(self):
        """A missing dunder attribute raises ``AttributeError`` (not ``KeyError``)."""
        d = NestedDict({'a': 1})
        with self.assertRaises(AttributeError):
            getattr(d, '__deepcopy__')
        self.assertFalse(hasattr(d, '__deepcopy__'))

    def test_missing_non_dunder_key_still_keyerror(self):
        """A missing ordinary key is still surfaced as ``KeyError``."""
        d = NestedDict({'a': 1})
        with self.assertRaises(KeyError):
            getattr(d, 'does_not_exist')

    def test_deepcopy_roundtrips(self):
        """``copy.deepcopy`` reproduces the mapping contents."""
        import copy
        d = NestedDict({'a': 1, 'b': {'c': 2}})
        d2 = copy.deepcopy(d)
        self.assertEqual(d2.to_dict(), {'a': 1, 'b': {'c': 2}})

    def test_existing_key_attribute_access_still_works(self):
        """Accessing an existing key as an attribute still returns its value."""
        d = NestedDict({'a': 5})
        self.assertEqual(d.a, 5)


class TestPrettyPytreeRoundtrips(unittest.TestCase):
    """Verify flat/nest roundtrips and JAX pytree registration consistency."""

    def test_flat_nest_roundtrip(self):
        """Reproduce the nested structure via ``nest_mapping(flat_mapping(x))``."""
        nested = {'a': 1, 'b': {'c': 2, 'd': {'e': 3}}}
        self.assertEqual(nest_mapping(flat_mapping(nested)).to_dict(), nested)

    def test_nesteddict_flatten_unflatten_consistent(self):
        """Flatten and unflatten a NestedDict to an equal JAX structure."""
        nd = NestedDict({'a': brainstate.ParamState(1), 'b': {'c': brainstate.ParamState(2)}})
        leaves, treedef = jax.tree.flatten(nd)
        rebuilt = jax.tree.unflatten(treedef, leaves)
        self.assertEqual(jax.tree.structure(nd), jax.tree.structure(rebuilt))

    def test_empty_nesteddict(self):
        """Flatten an empty NestedDict to zero leaves."""
        self.assertEqual(len(jax.tree.leaves(NestedDict({}))), 0)


class TestFlatMappingEmptyWhole(unittest.TestCase):
    """Exercise the whole-empty-input branch of ``flat_mapping``."""

    def test_flat_mapping_whole_empty_keep_empty_nodes(self):
        """Return an empty FlattedDict when the whole input is empty even with keep_empty_nodes."""
        flat = flat_mapping({}, keep_empty_nodes=True)
        self.assertIsInstance(flat, FlattedDict)
        self.assertEqual(len(flat), 0)


class TestDefaultIdentityHelpers(unittest.TestCase):
    """Cover the module-level identity helper functions."""

    def test_default_process_returns_id(self):
        """Return the object identity from ``_default_process``."""
        obj = object()
        self.assertEqual(_default_process(obj), id(obj))

    def test_default_compare_membership(self):
        """Report membership of an object's id in a set via ``_default_compare``."""
        obj = object()
        self.assertTrue(_default_compare(obj, {id(obj)}))
        self.assertFalse(_default_compare(obj, set()))


class TestPrettyDictAbstractAndUtils(unittest.TestCase):
    """Cover PrettyDict abstract methods, subset, and treefy_state."""

    def test_split_not_implemented(self):
        """Raise NotImplementedError from the abstract ``PrettyDict.split``."""
        d = PrettyDict({'a': 1})
        with self.assertRaises(NotImplementedError):
            d.split(lambda p, v: True)

    def test_filter_not_implemented(self):
        """Raise NotImplementedError from the abstract ``PrettyDict.filter``."""
        d = PrettyDict({'a': 1})
        with self.assertRaises(NotImplementedError):
            d.filter(lambda p, v: True)

    def test_merge_not_implemented(self):
        """Raise NotImplementedError from the abstract ``PrettyDict.merge``."""
        d = PrettyDict({'a': 1})
        with self.assertRaises(NotImplementedError):
            d.merge(PrettyDict({'b': 2}))

    def test_subset_delegates_to_filter(self):
        """Subset a NestedDict by delegating to ``filter``."""
        nd = NestedDict({'a': 1, 'b': 2, 'c': 3})
        subset = nd.subset(lambda p, v: v > 1)
        flat = subset.to_flat()
        self.assertIn(('b',), flat)
        self.assertIn(('c',), flat)
        self.assertNotIn(('a',), flat)

    def test_treefy_state_converts_state_objects(self):
        """Map State objects through ``to_state_ref`` while preserving structure."""
        d = PrettyDict({'a': brainstate.ParamState(1), 'b': 2})
        ref_tree = d.treefy_state()
        # The flatten/map/unflatten roundtrip preserves the dict structure and
        # passes non-State leaves through unchanged.
        self.assertEqual(set(ref_tree.keys()), {'a', 'b'})
        self.assertEqual(ref_tree['b'], 2)
        # State leaves remain State-like (their reference form).
        self.assertEqual(ref_tree['a'].value, 1)


class TestNestedStateReprTreescope(unittest.TestCase):
    """Cover NestedStateRepr treescope and nested wrapping behaviour."""

    def test_treescope_repr_wraps_nested_dicts(self):
        """Render nested PrettyDict children through a subtree renderer."""
        nsr = NestedStateRepr(NestedDict({'a': 1, 'b': NestedDict({'c': 2})}))
        captured = {}

        def renderer(children, path):
            captured['children'] = children
            captured['path'] = path
            return 'rendered'

        result = nsr.__treescope_repr__('PATH', renderer)
        self.assertEqual(result, 'rendered')
        self.assertEqual(captured['path'], 'PATH')
        # Nested PrettyDict value should have been wrapped in NestedStateRepr.
        self.assertIsInstance(captured['children']['b'], NestedStateRepr)
        self.assertEqual(captured['children']['a'], 1)

    def test_nested_state_repr_pretty_output(self):
        """Produce a compact brace-delimited repr for a wrapped NestedDict."""
        from brainstate.util._pretty_repr import pretty_repr_object
        nsr = NestedStateRepr(NestedDict({'a': 1, 'b': NestedDict({'c': 2})}))
        out = pretty_repr_object(nsr)
        self.assertIn('a', out)
        self.assertIn('c', out)


class TestNestedDictMergeAndOps(unittest.TestCase):
    """Cover NestedDict merge variants and empty-operand operators."""

    def test_or_with_empty_other_returns_self(self):
        """Return self unchanged when OR-ing with an empty NestedDict."""
        d1 = NestedDict({'a': 1})
        result = d1 | NestedDict({})
        self.assertIs(result, d1)

    def test_sub_with_empty_other_returns_self(self):
        """Return self unchanged when subtracting an empty NestedDict."""
        d1 = NestedDict({'a': 1})
        result = d1 - NestedDict({})
        self.assertIs(result, d1)

    def test_merge_accepts_flatted_dict(self):
        """Merge a FlattedDict argument into a NestedDict result."""
        nd = NestedDict({'a': 1})
        fd = FlattedDict({('b',): 2})
        merged = NestedDict.merge(nd, fd)
        self.assertEqual(merged['a'], 1)
        self.assertEqual(merged['b'], 2)

    def test_merge_rejects_invalid_type(self):
        """Raise TypeError when merging a non-mapping into a NestedDict."""
        with self.assertRaises(TypeError):
            NestedDict.merge(NestedDict({'a': 1}), [('b', 2)])

    def test_filter_single_filter_returns_single(self):
        """Return a single NestedDict when ``filter`` is given one filter."""
        nd = NestedDict({'a': 1, 'b': 2})
        result = nd.filter(lambda p, v: v > 1)
        self.assertIsInstance(result, NestedDict)
        self.assertIn(('b',), result.to_flat())

    def test_split_single_filter_returns_single(self):
        """Return a single NestedDict when ``split`` resolves to one state."""
        nd = NestedDict({'a': 1, 'b': 2})
        result = nd.split(...)
        self.assertIsInstance(result, NestedDict)
        self.assertEqual(len(result.to_flat()), 2)

    def test_filter_multiple_filters_returns_tuple(self):
        """Return a tuple of NestedDicts when filtering with two filters."""
        nd = NestedDict({'a': 1, 'b': 2, 'c': 3})
        evens, odds = nd.filter(lambda p, v: v % 2 == 0, lambda p, v: v % 2 == 1)
        self.assertIn(('b',), evens.to_flat())
        self.assertIn(('a',), odds.to_flat())
        self.assertIn(('c',), odds.to_flat())


class TestNestedDictReplaceByPureDict(unittest.TestCase):
    """Cover NestedDict.replace_by_pure_dict for State and plain values."""

    def test_replace_state_values(self):
        """Replace State values via their ``replace`` method by default."""
        nd = NestedDict({'a': brainstate.ParamState(1), 'b': brainstate.ParamState(2)})
        nd.replace_by_pure_dict({'a': 10, 'b': 20})
        self.assertEqual(nd['a'].value, 10)
        self.assertEqual(nd['b'].value, 20)

    def test_replace_plain_values(self):
        """Replace plain (non-State) values by direct assignment."""
        nd = NestedDict({'a': 5, 'b': 6})
        nd.replace_by_pure_dict({'a': 99, 'b': 100})
        self.assertEqual(nd['a'], 99)
        self.assertEqual(nd['b'], 100)

    def test_replace_with_custom_replace_fn(self):
        """Apply a custom replace function to combine old and new values."""
        nd = NestedDict({'a': 5})
        nd.replace_by_pure_dict({'a': 3}, replace_fn=lambda old, new: old + new)
        self.assertEqual(nd['a'], 8)

    def test_replace_missing_key_raises(self):
        """Raise ValueError when a pure_dict key is absent from the state."""
        nd = NestedDict({'a': 1})
        with self.assertRaises(ValueError):
            nd.replace_by_pure_dict({'z': 9})


class TestFlattedDictMergeAndOps(unittest.TestCase):
    """Cover FlattedDict merge variants, operators, and conversions."""

    def test_or_with_empty_other_returns_self(self):
        """Return self unchanged when OR-ing with an empty FlattedDict."""
        d1 = FlattedDict({('a',): 1})
        result = d1 | FlattedDict({})
        self.assertIs(result, d1)

    def test_sub_with_empty_other_returns_self(self):
        """Return self unchanged when subtracting an empty FlattedDict."""
        d1 = FlattedDict({('a',): 1})
        result = d1 - FlattedDict({})
        self.assertIs(result, d1)

    def test_from_nest_classmethod(self):
        """Build a FlattedDict from a nested mapping via ``from_nest``."""
        fd = FlattedDict.from_nest({'a': 1, 'b': {'c': 2}})
        self.assertIsInstance(fd, FlattedDict)
        self.assertEqual(fd[('a',)], 1)
        self.assertEqual(fd[('b', 'c')], 2)

    def test_split_exhaustive(self):
        """Split a FlattedDict exhaustively into two FlattedDicts."""
        fd = FlattedDict({('a',): 1, ('b',): 2})
        big, rest = fd.split(lambda p, v: v > 1, ...)
        self.assertIn(('b',), big)
        self.assertIn(('a',), rest)

    def test_split_non_exhaustive_raises(self):
        """Raise ValueError for non-exhaustive FlattedDict split filters."""
        fd = FlattedDict({('a',): 1, ('b',): 2})
        with self.assertRaises(ValueError):
            fd.split(lambda p, v: v > 1)

    def test_split_single_filter_returns_single(self):
        """Return a single FlattedDict when split resolves to one state."""
        fd = FlattedDict({('a',): 1, ('b',): 2})
        result = fd.split(...)
        self.assertIsInstance(result, FlattedDict)
        self.assertEqual(len(result), 2)

    def test_filter_single_filter_returns_single(self):
        """Return a single FlattedDict when ``filter`` is given one filter."""
        fd = FlattedDict({('a',): 1, ('b',): 2})
        result = fd.filter(lambda p, v: v > 1)
        self.assertIsInstance(result, FlattedDict)
        self.assertIn(('b',), result)

    def test_filter_multiple_filters_returns_tuple(self):
        """Return a tuple of FlattedDicts when filtering with two filters."""
        fd = FlattedDict({('a',): 1, ('b',): 2, ('c',): 3})
        evens, odds = fd.filter(lambda p, v: v % 2 == 0, lambda p, v: v % 2 == 1)
        self.assertIn(('b',), evens)
        self.assertIn(('a',), odds)
        self.assertIn(('c',), odds)

    def test_merge_accepts_nested_dict(self):
        """Merge a NestedDict argument into a FlattedDict result."""
        fd = FlattedDict({('a',): 1})
        nd = NestedDict({'b': 2})
        merged = FlattedDict.merge(fd, nd)
        self.assertEqual(merged[('a',)], 1)
        self.assertEqual(merged[('b',)], 2)

    def test_merge_rejects_invalid_type(self):
        """Raise TypeError when merging a non-mapping into a FlattedDict."""
        with self.assertRaises(TypeError):
            FlattedDict.merge(FlattedDict({('a',): 1}), 123)


class TestSplitErrorPaths(unittest.TestCase):
    """Cover the ``...``/``True`` filter-position validation branches."""

    def test_nested_ellipsis_then_ellipsis_no_error(self):
        """Allow trailing ``...`` filters after an earlier ``...`` in NestedDict.split."""
        nd = NestedDict({'a': 1, 'b': 2})
        first, second = nd.split(..., ...)
        self.assertIsInstance(first, NestedDict)
        self.assertIsInstance(second, NestedDict)

    def test_nested_ellipsis_before_real_filter_raises(self):
        """Raise ValueError when ``...`` precedes a real filter in NestedDict.split."""
        nd = NestedDict({'a': 1, 'b': 2})
        with self.assertRaises(ValueError):
            nd.split(..., lambda p, v: v > 1)

    def test_flatted_ellipsis_then_ellipsis_no_error(self):
        """Allow trailing ``...`` filters after an earlier ``...`` in FlattedDict.split."""
        fd = FlattedDict({('a',): 1, ('b',): 2})
        first, second = fd.split(..., ...)
        self.assertIsInstance(first, FlattedDict)
        self.assertIsInstance(second, FlattedDict)

    def test_flatted_ellipsis_before_real_filter_raises(self):
        """Raise ValueError when ``...`` precedes a real filter in FlattedDict.split."""
        fd = FlattedDict({('a',): 1, ('b',): 2})
        with self.assertRaises(ValueError):
            fd.split(..., lambda p, v: v > 1)


class TestReprAttributeBranches(unittest.TestCase):
    """Cover list/dict conversion and key-filtering branches in repr helpers."""

    def test_pretty_dict_with_list_value_repr(self):
        """Render a PrettyDict that holds a list value (PrettyList conversion)."""
        d = PrettyDict({'a': [1, 2, 3], 'b': 1})
        out = repr(d)
        self.assertIsInstance(out, str)
        self.assertIn('1', out)

    def test_pretty_list_of_lists_repr(self):
        """Render a PrettyList that holds nested list items."""
        pl = PrettyList([[1, 2], [3, 4]])
        out = repr(pl)
        self.assertIsInstance(out, str)
        self.assertIn('3', out)

    def test_pretty_object_with_list_and_dict_attrs(self):
        """Render a PrettyObject whose attributes are a list and a dict."""
        class Obj(PrettyObject):
            def __init__(self):
                self.mylist = [1, 2, 3]
                self.mydict = {'x': 1}

        out = repr(Obj())
        self.assertIn('mylist', out)
        self.assertIn('mydict', out)

    def test_pretty_object_item_returns_none_key_is_skipped(self):
        """Skip an attribute when ``__pretty_repr_item__`` returns a ``None`` key."""
        class Obj(PrettyObject):
            def __init__(self):
                self.keep = 'visible'
                self.drop = 'gone'

            def __pretty_repr_item__(self, k, v):
                if k == 'drop':
                    return None, v
                return k, v

        out = repr(Obj())
        self.assertIn('keep', out)
        self.assertNotIn('gone', out)

    def test_pretty_object_without_item_hook(self):
        """Render a PrettyObject lacking ``__pretty_repr_item__`` (AttributeError path)."""
        # A bare PrettyObject still defines the hook, so use a value attribute
        # to exercise the general attribute path with the default hook.
        class Obj(PrettyObject):
            def __init__(self):
                self.value = 7

        out = repr(Obj())
        self.assertIn('value', out)
        self.assertIn('7', out)

    def test_general_repr_object_missing_item_hook(self):
        """Exercise the AttributeError fallback when a node lacks the item hook."""
        from brainstate.util._pretty_repr import (
            PrettyRepr,
            yield_unique_pretty_repr_items,
            pretty_repr_object,
        )
        from brainstate.util._pretty_pytree import (
            _repr_object_general,
            _repr_attribute_general,
        )

        class Bare(PrettyRepr):
            # Deliberately does NOT define ``__pretty_repr_item__`` so the
            # ``except AttributeError`` branch in ``_repr_attribute_general`` runs.
            def __init__(self):
                self.x = 5
                self.items_list = [1, 2]

            def __pretty_repr__(self):
                yield from yield_unique_pretty_repr_items(
                    self, _repr_object_general, _repr_attribute_general
                )

        out = pretty_repr_object(Bare())
        self.assertIn('x', out)
        self.assertIn('items_list', out)
