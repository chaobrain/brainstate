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
from collections.abc import Callable
from threading import Thread

import jax
import jax.numpy as jnp
from absl.testing import absltest, parameterized

import pytest
pytest.skip("skipping tests", allow_module_level=True)

import brainstate
import braintools
import brainpy


class TestIter(unittest.TestCase):
    def test1(self):
        class Model(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = brainstate.nn.Linear(1, 2)
                self.b = brainstate.nn.Linear(2, 3)
                self.c = [brainstate.nn.Linear(3, 4), brainstate.nn.Linear(4, 5)]
                self.d = {'x': brainstate.nn.Linear(5, 6), 'y': brainstate.nn.Linear(6, 7)}
                self.b.a = brainpy.LIF(2)

        for path, node in brainstate.graph.iter_leaf(Model()):
            print(path, node)
        for path, node in brainstate.graph.iter_node(Model()):
            print(path, node)
        for path, node in brainstate.graph.iter_node(Model(), allowed_hierarchy=(1, 1)):
            print(path, node)
        for path, node in brainstate.graph.iter_node(Model(), allowed_hierarchy=(2, 2)):
            print(path, node)

    def test_iter_leaf_v1(self):
        class Linear(brainstate.nn.Module):
            def __init__(self, din, dout):
                super().__init__()
                self.weight = brainstate.ParamState(brainstate.random.randn(din, dout))
                self.bias = brainstate.ParamState(brainstate.random.randn(dout))
                self.a = 1

        module = Linear(3, 4)
        graph = [module, module]

        num = 0
        for path, value in brainstate.graph.iter_leaf(graph):
            print(path, type(value).__name__)
            num += 1

        assert num == 3

    def test_iter_node_v1(self):
        class Model(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = brainstate.nn.Linear(1, 2)
                self.b = brainstate.nn.Linear(2, 3)
                self.c = [brainstate.nn.Linear(3, 4), brainstate.nn.Linear(4, 5)]
                self.d = {'x': brainstate.nn.Linear(5, 6), 'y': brainstate.nn.Linear(6, 7)}
                self.b.a = brainpy.LIF(2)

        model = Model()

        num = 0
        for path, node in brainstate.graph.iter_node([model, model]):
            print(path, node.__class__.__name__)
            num += 1
        assert num == 8


class List(brainstate.nn.Module):
    def __init__(self, items):
        super().__init__()
        self.items = list(items)

    def __getitem__(self, idx):
        return self.items[idx]

    def __setitem__(self, idx, value):
        self.items[idx] = value


class Dict(brainstate.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.items = dict(*args, **kwargs)

    def __getitem__(self, key):
        return self.items[key]

    def __setitem__(self, key, value):
        self.items[key] = value


class StatefulLinear(brainstate.nn.Module):
    def __init__(self, din, dout):
        super().__init__()
        self.w = brainstate.ParamState(brainstate.random.rand(din, dout))
        self.b = brainstate.ParamState(jnp.zeros((dout,)))
        self.count = brainstate.State(jnp.array(0, dtype=jnp.uint32))

    def increment(self):
        self.count.value += 1

    def __call__(self, x):
        self.count.value += 1
        return x @ self.w.value + self.b.value


class TestGraphUtils(absltest.TestCase):
    def test_flatten_treey_state(self):
        a = {'a': 1, 'b': brainstate.ParamState(2)}
        g = [a, 3, a, brainstate.ParamState(4)]

        refmap = brainstate.graph.RefMap()
        graphdef, states = brainstate.graph.flatten(g, ref_index=refmap, treefy_state=True)

        states[0]['b'].value = 2
        states[3].value = 4

        assert isinstance(states[0]['b'], brainstate.TreefyState)
        assert isinstance(states[3], brainstate.TreefyState)
        assert isinstance(states, brainstate.util.NestedDict)
        assert len(refmap) == 2
        assert a['b'] in refmap
        assert g[3] in refmap

    def test_flatten(self):
        a = {'a': 1, 'b': brainstate.ParamState(2)}
        g = [a, 3, a, brainstate.ParamState(4)]

        refmap = brainstate.graph.RefMap()
        graphdef, states = brainstate.graph.flatten(g, ref_index=refmap, treefy_state=False)

        states[0]['b'].value = 2
        states[3].value = 4

        assert isinstance(states[0]['b'], brainstate.State)
        assert isinstance(states[3], brainstate.State)
        assert len(refmap) == 2
        assert a['b'] in refmap
        assert g[3] in refmap

    def test_unflatten_pytree(self):
        a = {'a': 1, 'b': brainstate.ParamState(2)}
        g = [a, 3, a, brainstate.ParamState(4)]

        graphdef, references = brainstate.graph.treefy_split(g)
        g = brainstate.graph.treefy_merge(graphdef, references)

        assert g[0] is not g[2]

    def test_unflatten_empty(self):
        a = Dict({'a': 1, 'b': brainstate.ParamState(2)})
        g = List([a, 3, a, brainstate.ParamState(4)])

        graphdef, references = brainstate.graph.treefy_split(g)

        with self.assertRaisesRegex(ValueError, 'Expected key'):
            brainstate.graph.unflatten(graphdef, brainstate.util.NestedDict({}))

    def test_module_list(self):
        ls = [
            brainstate.nn.Linear(2, 2),
            brainstate.nn.BatchNorm1d([10, 2]),
        ]
        graphdef, statetree = brainstate.graph.treefy_split(ls)

        assert statetree[0]['weight'].value['weight'].shape == (2, 2)
        assert statetree[0]['weight'].value['bias'].shape == (2,)
        assert statetree[1]['weight'].value['scale'].shape == (1, 2,)
        assert statetree[1]['weight'].value['bias'].shape == (1, 2,)
        assert statetree[1]['running_mean'].value.shape == (1, 2,)
        assert statetree[1]['running_var'].value.shape == (1, 2)

    def test_shared_variables(self):
        v = brainstate.ParamState(1)
        g = [v, v]

        graphdef, statetree = brainstate.graph.treefy_split(g)
        assert len(statetree.to_flat()) == 1

        g2 = brainstate.graph.treefy_merge(graphdef, statetree)
        assert g2[0] is g2[1]

    def test_tied_weights(self):
        class Foo(brainstate.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.bar = brainstate.nn.Linear(2, 2)
                self.baz = brainstate.nn.Linear(2, 2)

                # tie the weights
                self.baz.weight = self.bar.weight

        node = Foo()
        graphdef, state = brainstate.graph.treefy_split(node)

        assert len(state.to_flat()) == 1

        node2 = brainstate.graph.treefy_merge(graphdef, state)

        assert node2.bar.weight is node2.baz.weight

    def test_tied_weights_example(self):
        class LinearTranspose(brainstate.nn.Module):
            def __init__(self, dout: int, din: int, ) -> None:
                super().__init__()
                self.kernel = brainstate.ParamState(braintools.init.LecunNormal()((dout, din)))

            def __call__(self, x):
                return x @ self.kernel.value.T

        class Encoder(brainstate.nn.Module):
            def __init__(self, ) -> None:
                super().__init__()
                self.embed = brainstate.nn.Embedding(10, 2)
                self.linear_out = LinearTranspose(10, 2)

                # tie the weights
                self.linear_out.kernel = self.embed.weight

            def __call__(self, x):
                x = self.embed(x)
                return self.linear_out(x)

        model = Encoder()
        graphdef, state = brainstate.graph.treefy_split(model)

        assert len(state.to_flat()) == 1

        x = jax.random.randint(jax.random.key(0), (2,), 0, 10)
        y = model(x)

        assert y.shape == (2, 10)

    def test_state_variables_not_shared_with_graph(self):
        class Foo(brainstate.graph.Node):
            def __init__(self):
                self.a = brainstate.ParamState(1)

        m = Foo()
        graphdef, statetree = brainstate.graph.treefy_split(m)

        assert isinstance(m.a, brainstate.ParamState)
        assert issubclass(statetree.a.type, brainstate.ParamState)
        assert m.a is not statetree.a
        assert m.a.value == statetree.a.value

        m2 = brainstate.graph.treefy_merge(graphdef, statetree)

        assert isinstance(m2.a, brainstate.ParamState)
        assert issubclass(statetree.a.type, brainstate.ParamState)
        assert m2.a is not statetree.a
        assert m2.a.value == statetree.a.value

    def test_shared_state_variables_not_shared_with_graph(self):
        class Foo(brainstate.graph.Node):
            def __init__(self):
                p = brainstate.ParamState(1)
                self.a = p
                self.b = p

        m = Foo()
        graphdef, state = brainstate.graph.treefy_split(m)

        assert isinstance(m.a, brainstate.ParamState)
        assert isinstance(m.b, brainstate.ParamState)
        assert issubclass(state.a.type, brainstate.ParamState)
        assert 'b' not in state
        assert m.a is not state.a
        assert m.b is not state.a
        assert m.a.value == state.a.value
        assert m.b.value == state.a.value

        m2 = brainstate.graph.treefy_merge(graphdef, state)

        assert isinstance(m2.a, brainstate.ParamState)
        assert isinstance(m2.b, brainstate.ParamState)
        assert issubclass(state.a.type, brainstate.ParamState)
        assert m2.a is not state.a
        assert m2.b is not state.a
        assert m2.a.value == state.a.value
        assert m2.b.value == state.a.value
        assert m2.a is m2.b

    def test_pytree_node(self):
        @brainstate.util.dataclass
        class Tree:
            a: brainstate.ParamState
            b: str = brainstate.util.field(pytree_node=False)

        class Foo(brainstate.graph.Node):
            def __init__(self):
                self.tree = Tree(brainstate.ParamState(1), 'a')

        m = Foo()

        graphdef, state = brainstate.graph.treefy_split(m)

        assert 'tree' in state
        assert 'a' in state.tree
        assert graphdef.subgraphs['tree'].type.__name__ == 'PytreeType'

        m2 = brainstate.graph.treefy_merge(graphdef, state)

        assert isinstance(m2.tree, Tree)
        assert m2.tree.a.value == 1
        assert m2.tree.b == 'a'
        assert m2.tree.a is not m.tree.a
        assert m2.tree is not m.tree


class SimpleModule(brainstate.nn.Module):
    pass


class SimplePyTreeModule(brainstate.nn.Module):
    pass


class TestThreading(parameterized.TestCase):

    @parameterized.parameters(
        (SimpleModule,),
        (SimplePyTreeModule,),
    )
    def test_threading(self, module_fn: Callable[[], brainstate.nn.Module]):
        x = module_fn()

        class MyThread(Thread):

            def run(self) -> None:
                brainstate.graph.treefy_split(x)

        thread = MyThread()
        thread.start()
        thread.join()


class TestGraphOperation(unittest.TestCase):
    def test1(self):
        class MyNode(brainstate.graph.Node):
            def __init__(self):
                self.a = brainstate.nn.Linear(2, 3)
                self.b = brainstate.nn.Linear(3, 2)
                self.c = [brainstate.nn.Linear(1, 2), brainstate.nn.Linear(1, 3)]
                self.d = {'x': brainstate.nn.Linear(1, 3), 'y': brainstate.nn.Linear(1, 4)}

        graphdef, statetree = brainstate.graph.flatten(MyNode())
        # print(graphdef)
        print(statetree)
        # print(brainstate.graph.unflatten(graphdef, statetree))

    def test_split(self):
        class Foo(brainstate.graph.Node):
            def __init__(self):
                self.a = brainstate.nn.Linear(2, 2)
                self.b = brainstate.nn.BatchNorm1d([10, 2])

        node = Foo()
        graphdef, params, others = brainstate.graph.treefy_split(node, brainstate.ParamState, ...)

        print(params)
        print(jax.tree.map(jnp.shape, params))

        print(jax.tree.map(jnp.shape, others))

    def test_merge(self):
        class Foo(brainstate.graph.Node):
            def __init__(self):
                self.a = brainstate.nn.Linear(2, 2)
                self.b = brainstate.nn.BatchNorm1d([10, 2])

        node = Foo()
        graphdef, params, others = brainstate.graph.treefy_split(node, brainstate.ParamState, ...)

        new_node = brainstate.graph.treefy_merge(graphdef, params, others)

        assert isinstance(new_node, Foo)
        assert isinstance(new_node.b, brainstate.nn.BatchNorm1d)
        assert isinstance(new_node.a, brainstate.nn.Linear)

    def test_update_states(self):
        x = jnp.ones((1, 2))
        y = jnp.ones((1, 3))
        model = brainstate.nn.Linear(2, 3)

        def loss_fn(x, y):
            return jnp.mean((y - model(x)) ** 2)

        def sgd(ps, gs):
            updates = jax.tree.map(lambda p, g: p - 0.1 * g, ps.value, gs)
            ps.value = updates

        prev_loss = loss_fn(x, y)
        weights = model.states()
        grads = brainstate.transform.grad(loss_fn, weights)(x, y)
        for key, val in grads.items():
            sgd(weights[key], val)
        assert loss_fn(x, y) < prev_loss

    def test_pop_states(self):
        class Model(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = brainstate.nn.Linear(2, 3)
                self.b = brainpy.LIF([10, 2])

        model = Model()
        with brainstate.catch_new_states('new'):
            brainstate.nn.init_all_states(model)
        # print(model.states())
        self.assertTrue(len(model.states()) == 2)
        model_states = brainstate.graph.pop_states(model, 'new')
        print(model_states)
        self.assertTrue(len(model.states()) == 1)
        assert not hasattr(model.b, 'V')
        # print(model.states())

    def test_treefy_split(self):
        class MLP(brainstate.graph.Node):
            def __init__(self, din: int, dmid: int, dout: int, n_layer: int = 3):
                self.input = brainstate.nn.Linear(din, dmid)
                self.layers = [brainstate.nn.Linear(dmid, dmid) for _ in range(n_layer)]
                self.output = brainstate.nn.Linear(dmid, dout)

            def __call__(self, x):
                x = brainstate.functional.relu(self.input(x))
                for layer in self.layers:
                    x = brainstate.functional.relu(layer(x))
                return self.output(x)

        model = MLP(2, 1, 3)
        graph_def, treefy_states = brainstate.graph.treefy_split(model)

        print(graph_def)
        print(treefy_states)

        # states = brainstate.graph.states(model)
        # print(states)
        # nest_states = states.to_nest()
        # print(nest_states)

    def test_states(self):
        class MLP(brainstate.graph.Node):
            def __init__(self, din: int, dmid: int, dout: int, n_layer: int = 3):
                self.input = brainstate.nn.Linear(din, dmid)
                self.layers = [brainstate.nn.Linear(dmid, dmid) for _ in range(n_layer)]
                self.output = brainpy.LIF(dout)

            def __call__(self, x):
                x = brainstate.functional.relu(self.input(x))
                for layer in self.layers:
                    x = brainstate.functional.relu(layer(x))
                return self.output(x)

        model = brainstate.nn.init_all_states(MLP(2, 1, 3))
        states = brainstate.graph.states(model)
        print(states)
        nest_states = states.to_nest()
        print(nest_states)

        params, others = brainstate.graph.states(model, brainstate.ParamState, brainstate.ShortTermState)
        print(params)
        print(others)


class TestRefMap(unittest.TestCase):
    """Test RefMap class functionality."""

    def test_refmap_basic_operations(self):
        """Test basic RefMap operations."""
        ref_map = brainstate.graph.RefMap()

        # Test empty RefMap
        self.assertEqual(len(ref_map), 0)
        self.assertFalse(object() in ref_map)

        # Test adding items
        obj1 = object()
        obj2 = object()
        ref_map[obj1] = 'value1'
        ref_map[obj2] = 'value2'

        self.assertEqual(len(ref_map), 2)
        self.assertTrue(obj1 in ref_map)
        self.assertTrue(obj2 in ref_map)
        self.assertEqual(ref_map[obj1], 'value1')
        self.assertEqual(ref_map[obj2], 'value2')

        # Test iteration
        keys = list(ref_map)
        self.assertIn(obj1, keys)
        self.assertIn(obj2, keys)

        # Test deletion
        del ref_map[obj1]
        self.assertEqual(len(ref_map), 1)
        self.assertFalse(obj1 in ref_map)
        self.assertTrue(obj2 in ref_map)

    def test_refmap_initialization_with_mapping(self):
        """Test RefMap initialization with a mapping."""
        obj1, obj2 = object(), object()
        mapping = {obj1: 'value1', obj2: 'value2'}
        ref_map = brainstate.graph.RefMap(mapping)

        self.assertEqual(len(ref_map), 2)
        self.assertEqual(ref_map[obj1], 'value1')
        self.assertEqual(ref_map[obj2], 'value2')

    def test_refmap_initialization_with_iterable(self):
        """Test RefMap initialization with an iterable."""
        obj1, obj2 = object(), object()
        pairs = [(obj1, 'value1'), (obj2, 'value2')]
        ref_map = brainstate.graph.RefMap(pairs)

        self.assertEqual(len(ref_map), 2)
        self.assertEqual(ref_map[obj1], 'value1')
        self.assertEqual(ref_map[obj2], 'value2')

    def test_refmap_same_object_different_instances(self):
        """Test RefMap handles same content objects with different ids."""
        # Create two lists with same content but different ids
        list1 = [1, 2, 3]
        list2 = [1, 2, 3]

        ref_map = brainstate.graph.RefMap()
        ref_map[list1] = 'list1'
        ref_map[list2] = 'list2'

        # Should have 2 entries since they have different ids
        self.assertEqual(len(ref_map), 2)
        self.assertEqual(ref_map[list1], 'list1')
        self.assertEqual(ref_map[list2], 'list2')

    def test_refmap_update(self):
        """Test RefMap update method."""
        obj1, obj2, obj3 = object(), object(), object()
        ref_map = brainstate.graph.RefMap()
        ref_map[obj1] = 'value1'

        # Update with mapping
        ref_map.update({obj2: 'value2', obj3: 'value3'})
        self.assertEqual(len(ref_map), 3)

        # Update existing key
        ref_map[obj1] = 'new_value1'
        self.assertEqual(ref_map[obj1], 'new_value1')

    def test_refmap_str_repr(self):
        """Test RefMap string representation."""
        ref_map = brainstate.graph.RefMap()
        obj = object()
        ref_map[obj] = 'value'

        str_repr = str(ref_map)
        self.assertIsInstance(str_repr, str)
        # Check that __str__ calls __repr__
        self.assertEqual(str_repr, repr(ref_map))


class TestHelperFunctions(unittest.TestCase):
    """Test helper functions in the _operation module."""

    def test_is_state_leaf(self):
        """Test _is_state_leaf function."""
        from brainstate.graph._operation import _is_state_leaf

        # Create TreefyState instance
        state = brainstate.ParamState(1)
        treefy_state = state.to_state_ref()

        self.assertTrue(_is_state_leaf(treefy_state))
        self.assertFalse(_is_state_leaf(state))
        self.assertFalse(_is_state_leaf(1))
        self.assertFalse(_is_state_leaf("string"))
        self.assertFalse(_is_state_leaf(None))

    def test_is_node_leaf(self):
        """Test _is_node_leaf function."""
        from brainstate.graph._operation import _is_node_leaf

        state = brainstate.ParamState(1)

        self.assertTrue(_is_node_leaf(state))
        self.assertFalse(_is_node_leaf(1))
        self.assertFalse(_is_node_leaf("string"))
        self.assertFalse(_is_node_leaf(None))

    def test_is_node(self):
        """Test _is_node function."""
        from brainstate.graph._operation import _is_node

        # Test with graph nodes
        node = brainstate.nn.Module()
        self.assertTrue(_is_node(node))

        # Test with pytree nodes
        self.assertTrue(_is_node([1, 2, 3]))
        self.assertTrue(_is_node({'a': 1}))

        # Test with non-nodes
        self.assertFalse(_is_node(1))
        self.assertFalse(_is_node("string"))

    def test_is_pytree_node(self):
        """Test _is_pytree_node function."""
        from brainstate.graph._operation import _is_pytree_node

        self.assertTrue(_is_pytree_node([1, 2, 3]))
        self.assertTrue(_is_pytree_node({'a': 1}))
        self.assertTrue(_is_pytree_node((1, 2)))

        self.assertFalse(_is_pytree_node(1))
        self.assertFalse(_is_pytree_node("string"))
        self.assertFalse(_is_pytree_node(jnp.array([1, 2])))

    def test_is_graph_node(self):
        """Test _is_graph_node function."""
        from brainstate.graph._operation import _is_graph_node

        # Register a custom type for testing
        class CustomNode:
            pass

        # Graph nodes are those registered with register_graph_node_type
        node = brainstate.nn.Module()
        self.assertTrue(_is_graph_node(node))

        # Non-registered types
        self.assertFalse(_is_graph_node([1, 2, 3]))
        self.assertFalse(_is_graph_node({'a': 1}))
        self.assertFalse(_is_graph_node(CustomNode()))


class TestRegisterGraphNodeType(unittest.TestCase):
    """Test register_graph_node_type functionality."""

    def test_register_custom_node_type(self):
        """Test registering a custom graph node type."""
        from brainstate.graph._operation import _is_graph_node, _get_node_impl

        class CustomNode:
            def __init__(self):
                self.data = {}

        def flatten_custom(node):
            return list(node.data.items()), None

        def set_key_custom(node, key, value):
            node.data[key] = value

        def pop_key_custom(node, key):
            return node.data.pop(key)

        def create_empty_custom(metadata):
            return CustomNode()

        def clear_custom(node):
            node.data.clear()

        # Register the custom node type
        brainstate.graph.register_graph_node_type(
            CustomNode,
            flatten_custom,
            set_key_custom,
            pop_key_custom,
            create_empty_custom,
            clear_custom
        )

        # Test that the node is recognized
        node = CustomNode()
        self.assertTrue(_is_graph_node(node))

        # Test node operations
        node.data['key1'] = 'value1'
        node_impl = _get_node_impl(node)

        # Test flatten
        items, metadata = node_impl.flatten(node)
        self.assertEqual(list(items), [('key1', 'value1')])

        # Test set_key
        node_impl.set_key(node, 'key2', 'value2')
        self.assertEqual(node.data['key2'], 'value2')

        # Test pop_key
        value = node_impl.pop_key(node, 'key1')
        self.assertEqual(value, 'value1')
        self.assertNotIn('key1', node.data)

        # Test create_empty
        new_node = node_impl.create_empty(None)
        self.assertIsInstance(new_node, CustomNode)
        self.assertEqual(new_node.data, {})

        # Test clear
        node_impl.clear(node)
        self.assertEqual(node.data, {})


class TestHashableMapping(unittest.TestCase):
    """Test HashableMapping class."""

    def test_hashable_mapping_basic(self):
        """Test basic HashableMapping operations."""
        from brainstate.graph._operation import HashableMapping

        mapping = {'a': 1, 'b': 2}
        hm = HashableMapping(mapping)

        # Test basic operations
        self.assertEqual(len(hm), 2)
        self.assertTrue('a' in hm)
        self.assertFalse('c' in hm)
        self.assertEqual(hm['a'], 1)
        self.assertEqual(hm['b'], 2)

        # Test iteration
        keys = list(hm)
        self.assertEqual(set(keys), {'a', 'b'})

    def test_hashable_mapping_hash(self):
        """Test HashableMapping hashing."""
        from brainstate.graph._operation import HashableMapping

        hm1 = HashableMapping({'a': 1, 'b': 2})
        hm2 = HashableMapping({'a': 1, 'b': 2})
        hm3 = HashableMapping({'a': 1, 'b': 3})

        # Equal mappings should have same hash
        self.assertEqual(hash(hm1), hash(hm2))
        self.assertEqual(hm1, hm2)

        # Different mappings should not be equal
        self.assertNotEqual(hm1, hm3)

        # Can be used in sets
        s = {hm1, hm2, hm3}
        self.assertEqual(len(s), 2)  # hm1 and hm2 are the same

    def test_hashable_mapping_from_iterable(self):
        """Test HashableMapping creation from iterable."""
        from brainstate.graph._operation import HashableMapping

        pairs = [('a', 1), ('b', 2)]
        hm = HashableMapping(pairs)

        self.assertEqual(len(hm), 2)
        self.assertEqual(hm['a'], 1)
        self.assertEqual(hm['b'], 2)


class TestNodeDefAndNodeRef(unittest.TestCase):
    """Test NodeDef and NodeRef classes."""

    def test_noderef_creation(self):
        """Test NodeRef creation and attributes."""
        node_ref = brainstate.graph.NodeRef(
            type=brainstate.nn.Module,
            index=42
        )

        self.assertEqual(node_ref.type, brainstate.nn.Module)
        self.assertEqual(node_ref.index, 42)

    def test_nodedef_creation(self):
        """Test NodeDef creation and attributes."""
        from brainstate.graph._operation import HashableMapping

        nodedef = brainstate.graph.NodeDef.create(
            type=brainstate.nn.Module,
            index=1,
            attributes=('a', 'b'),
            subgraphs=[],
            static_fields=[('static', 'value')],
            leaves=[],
            metadata=None,
            index_mapping=None
        )

        self.assertEqual(nodedef.type, brainstate.nn.Module)
        self.assertEqual(nodedef.index, 1)
        self.assertEqual(nodedef.attributes, ('a', 'b'))
        self.assertIsInstance(nodedef.subgraphs, HashableMapping)
        self.assertIsInstance(nodedef.static_fields, HashableMapping)
        self.assertEqual(nodedef.static_fields['static'], 'value')
        self.assertIsNone(nodedef.metadata)
        self.assertIsNone(nodedef.index_mapping)

    def test_nodedef_with_index_mapping(self):
        """Test NodeDef with index_mapping."""
        nodedef = brainstate.graph.NodeDef.create(
            type=brainstate.nn.Module,
            index=1,
            attributes=(),
            subgraphs=[],
            static_fields=[],
            leaves=[],
            metadata=None,
            index_mapping={1: 2, 3: 4}
        )

        self.assertIsNotNone(nodedef.index_mapping)
        self.assertEqual(nodedef.index_mapping[1], 2)
        self.assertEqual(nodedef.index_mapping[3], 4)


class TestGraphDefAndClone(unittest.TestCase):
    """Test graphdef and clone functions."""

    def test_graphdef_function(self):
        """Test graphdef function returns correct GraphDef."""
        model = brainstate.nn.Linear(2, 3)
        graphdef = brainstate.graph.graphdef(model)

        self.assertIsInstance(graphdef, brainstate.graph.NodeDef)
        self.assertEqual(graphdef.type, brainstate.nn.Linear)

        # Compare with flatten result
        graphdef2, _ = brainstate.graph.flatten(model)
        self.assertEqual(graphdef, graphdef2)

    def test_clone_function(self):
        """Test clone creates a deep copy."""
        model = brainstate.nn.Linear(2, 3)
        cloned = brainstate.graph.clone(model)

        # Check types
        self.assertIsInstance(cloned, brainstate.nn.Linear)
        self.assertIsNot(model, cloned)

        # Check that states are not shared
        self.assertIsNot(model.weight, cloned.weight)

        # Modify original and check clone is unaffected
        original_weight = cloned.weight.value['weight'].copy()
        model.weight.value = jax.tree.map(lambda x: x + 1, model.weight.value)

        # Clone should be unchanged
        self.assertTrue(jnp.allclose(cloned.weight.value['weight'], original_weight))

    def test_clone_with_shared_variables(self):
        """Test cloning preserves shared variable structure."""

        class SharedModel(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.shared_weight = brainstate.ParamState(jnp.ones((2, 2)))
                self.layer1 = brainstate.nn.Linear(2, 2)
                self.layer2 = brainstate.nn.Linear(2, 2)
                # Share weights
                self.layer2.weight = self.layer1.weight

        model = SharedModel()
        cloned = brainstate.graph.clone(model)

        # Check that sharing is preserved
        self.assertIs(cloned.layer1.weight, cloned.layer2.weight)
        # But not shared with original
        self.assertIsNot(cloned.layer1.weight, model.layer1.weight)


class TestNodesFunction(unittest.TestCase):
    """Test nodes function for filtering graph nodes."""

    def test_nodes_without_filters(self):
        """Test nodes function without filters."""

        class Model(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = brainstate.nn.Linear(2, 3)
                self.b = brainstate.nn.Linear(3, 4)

        model = Model()
        all_nodes = brainstate.graph.nodes(model)

        # Should return all nodes as FlattedDict
        self.assertIsInstance(all_nodes, brainstate.util.FlattedDict)

        # Check that nodes are present
        paths = [path for path, _ in all_nodes.items()]
        self.assertIn(('a',), paths)
        self.assertIn(('b',), paths)
        self.assertIn((), paths)  # The model itself

    def test_nodes_with_filter(self):
        """Test nodes function with a single filter."""

        class CustomModule(brainstate.nn.Module):
            pass

        class Model(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = brainstate.nn.Linear(2, 3)
                self.custom = CustomModule()

        model = Model()

        # Filter for Linear modules
        linear_nodes = brainstate.graph.nodes(
            model,
            lambda path, node: isinstance(node, brainstate.nn.Linear)
        )

        self.assertIsInstance(linear_nodes, brainstate.util.FlattedDict)
        # Should only contain the Linear module
        nodes_list = list(linear_nodes.values())
        self.assertEqual(len(nodes_list), 1)
        self.assertIsInstance(nodes_list[0], brainstate.nn.Linear)

    def test_nodes_with_hierarchy(self):
        """Test nodes function with hierarchy limits."""

        class Model(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = brainstate.nn.Linear(2, 3)
                self.layer1.sublayer = brainstate.nn.Linear(3, 3)

        model = Model()

        # Get only level 1 nodes
        level1_nodes = brainstate.graph.nodes(model, allowed_hierarchy=(1, 1))
        paths = [path for path, _ in level1_nodes.items()]

        self.assertIn(('layer1',), paths)
        # Sublayer should not be included at level 1
        self.assertNotIn(('layer1', 'sublayer'), paths)


class TestStatic(unittest.TestCase):
    """Test Static class functionality."""

    def test_static_basic(self):
        """Test basic Static wrapper."""
        from brainstate.graph._operation import Static

        value = {'key': 'value'}
        static = Static(value)

        self.assertEqual(static.value, value)
        self.assertIs(static.value, value)

    def test_static_is_pytree_leaf(self):
        """Test that Static is treated as a pytree leaf."""
        from brainstate.graph._operation import Static

        static = Static({'key': 'value'})

        # Should be treated as a leaf in pytree operations
        leaves, treedef = jax.tree_util.tree_flatten(static)
        self.assertEqual(len(leaves), 0)  # Static has no leaves

        # Test in a structure
        tree = {'a': 1, 'b': static, 'c': [2, 3]}
        leaves, treedef = jax.tree_util.tree_flatten(tree)

        # static should not be in leaves since it's registered as static
        self.assertNotIn(static, leaves)

    def test_static_equality_and_hash(self):
        """Test Static equality and hashing."""
        from brainstate.graph._operation import Static

        static1 = Static(42)
        static2 = Static(42)
        static3 = Static(43)

        # Dataclass frozen=True provides equality
        self.assertEqual(static1, static2)
        self.assertNotEqual(static1, static3)

        # Can be hashed due to frozen=True
        self.assertEqual(hash(static1), hash(static2))
        self.assertNotEqual(hash(static1), hash(static3))


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases."""

    def test_flatten_with_invalid_ref_index(self):
        """Test flatten with invalid ref_index."""
        model = brainstate.nn.Linear(2, 3)

        # Should raise assertion error with non-RefMap
        with self.assertRaises(AssertionError):
            brainstate.graph.flatten(model, ref_index={})

    def test_unflatten_with_invalid_graphdef(self):
        """Test unflatten with invalid graphdef."""
        state = brainstate.util.NestedDict({})

        # Should raise assertion error with non-GraphDef
        with self.assertRaises(AssertionError):
            brainstate.graph.unflatten("not_a_graphdef", state)

    def test_pop_states_without_filters(self):
        """Test pop_states raises error without filters."""
        model = brainstate.nn.Linear(2, 3)

        with self.assertRaises(ValueError) as context:
            brainstate.graph.pop_states(model)

        self.assertIn('Expected at least one filter', str(context.exception))

    def test_update_states_immutable_node(self):
        """Test update_states on immutable pytree node."""
        # Create a pytree node (tuple is immutable)
        node = (1, 2, brainstate.ParamState(3))
        state = brainstate.util.NestedDict({0: brainstate.TreefyState(int, 10)})

        # Should raise ValueError when trying to update immutable node
        with self.assertRaises(ValueError):
            brainstate.graph.update_states(node, state)

    def test_get_node_impl_with_state(self):
        """Test _get_node_impl raises error for State objects."""
        from brainstate.graph._operation import _get_node_impl

        state = brainstate.ParamState(1)

        with self.assertRaises(ValueError) as context:
            _get_node_impl(state)

        self.assertIn('State is not a node', str(context.exception))

    def test_split_with_non_exhaustive_filters(self):
        """Test split with non-exhaustive filters."""
        from brainstate.graph._operation import _split_flatted

        flatted = [(('a',), 1), (('b',), 2)]
        filters = (lambda path, value: value == 1,)  # Only matches first item

        # Should raise ValueError for non-exhaustive filters
        with self.assertRaises(ValueError) as context:
            _split_flatted(flatted, filters)

        self.assertIn('Non-exhaustive filters', str(context.exception))

    def test_invalid_filter_order(self):
        """Test filters with ... not at the end."""
        from brainstate.graph._operation import _filters_to_predicates

        # ... must be the last filter
        filters = (..., lambda p, v: True)

        with self.assertRaises(ValueError) as context:
            _filters_to_predicates(filters)

        self.assertIn('can only be used as the last filters', str(context.exception))


class TestIntegration(unittest.TestCase):
    """Integration tests for complex scenarios."""

    def test_complex_graph_operations(self):
        """Test complex graph with multiple levels and shared references."""

        class SubModule(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = brainstate.ParamState(jnp.ones((2, 2)))

        class ComplexModel(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.shared = SubModule()
                self.layer1 = brainstate.nn.Linear(2, 3)
                self.layer2 = brainstate.nn.Linear(3, 4)
                self.layer2.shared_ref = self.shared  # Create a reference
                self.nested = {
                    'a': brainstate.nn.Linear(4, 5),
                    'b': [brainstate.nn.Linear(5, 6), self.shared]  # Another reference
                }

        model = ComplexModel()

        # Test flatten/unflatten preserves structure
        graphdef, state = brainstate.graph.treefy_split(model)
        reconstructed = brainstate.graph.treefy_merge(graphdef, state)

        # Check shared references are preserved
        self.assertIs(reconstructed.shared, reconstructed.layer2.shared_ref)
        self.assertIs(reconstructed.shared, reconstructed.nested['b'][1])

        # Test state updates
        new_state = jax.tree.map(lambda x: x * 2, state)
        brainstate.graph.update_states(model, new_state)

        # Verify updates applied
        self.assertTrue(jnp.allclose(
            model.shared.weight.value,
            jnp.ones((2, 2)) * 2
        ))

    def test_recursive_structure(self):
        """Test handling of recursive/circular references."""

        class RecursiveModule(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = brainstate.ParamState(1)
                self.child = None

        # Create circular reference
        parent = RecursiveModule()
        child = RecursiveModule()
        parent.child = child
        child.child = parent  # Circular reference

        # Should handle circular references without infinite recursion
        graphdef, state = brainstate.graph.treefy_split(parent)

        # Should be able to reconstruct
        reconstructed = brainstate.graph.treefy_merge(graphdef, state)

        # Check structure is preserved
        self.assertIsNotNone(reconstructed.child)
        self.assertIs(reconstructed.child.child, reconstructed)


if __name__ == '__main__':
    absltest.main()
