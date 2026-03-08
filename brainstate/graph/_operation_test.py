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

from __future__ import annotations

import unittest
from collections.abc import Callable
from threading import Thread

import jax
import jax.numpy as jnp
import pytest
from absl.testing import absltest, parameterized

import brainstate
import brainstate.util


# ---------------------------------------------------------------------------
# Module-level helpers used across test classes
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Iteration tests
# ---------------------------------------------------------------------------

class TestIter(unittest.TestCase):
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
            num += 1
        assert num == 3

    def test_iter_leaf_nested_modules(self):
        class SubModule(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = brainstate.ParamState(jnp.ones((2,)))

        class Model(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = brainstate.nn.Linear(1, 2)
                self.b = brainstate.nn.Linear(2, 3)
                self.c = [brainstate.nn.Linear(3, 4), brainstate.nn.Linear(4, 5)]
                self.d = {'x': brainstate.nn.Linear(5, 6), 'y': brainstate.nn.Linear(6, 7)}
                self.b.sub = SubModule()

        model = Model()
        for path, leaf in brainstate.graph.iter_leaf(model):
            assert path is not None
        for path, node in brainstate.graph.iter_node(model):
            assert path is not None

    def test_iter_node_hierarchy(self):
        class Model(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = brainstate.nn.Linear(1, 2)
                self.b = brainstate.nn.Linear(2, 3)

        model = Model()
        nodes_all = list(brainstate.graph.iter_node(model))
        nodes_level1 = list(brainstate.graph.iter_node(model, allowed_hierarchy=(1, 1)))
        assert len(nodes_level1) <= len(nodes_all)

    def test_iter_node_count(self):
        class SubModule(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = brainstate.ParamState(jnp.ones((2,)))

        class Model(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = brainstate.nn.Linear(1, 2)
                self.b = brainstate.nn.Linear(2, 3)
                self.c = [brainstate.nn.Linear(3, 4), brainstate.nn.Linear(4, 5)]
                self.d = {'x': brainstate.nn.Linear(5, 6), 'y': brainstate.nn.Linear(6, 7)}
                self.b.sub = SubModule()

        model = Model()
        # iter_node on [model, model] should visit each node only once
        num = 0
        for path, node in brainstate.graph.iter_node([model, model]):
            num += 1
        # Model itself + a, b, b.sub, c[0], c[1], d[x], d[y] = 8
        assert num == 8


# ---------------------------------------------------------------------------
# Core graph utility tests
# ---------------------------------------------------------------------------

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
                self.baz.weight = self.bar.weight

        node = Foo()
        graphdef, state = brainstate.graph.treefy_split(node)

        assert len(state.to_flat()) == 1

        node2 = brainstate.graph.treefy_merge(graphdef, state)

        assert node2.bar.weight is node2.baz.weight

    def test_tied_weights_example_with_braintools(self):
        braintools = pytest.importorskip('braintools')

        class LinearTranspose(brainstate.nn.Module):
            def __init__(self, dout: int, din: int) -> None:
                super().__init__()
                self.kernel = brainstate.ParamState(braintools.init.LecunNormal()((dout, din)))

            def __call__(self, x):
                return x @ self.kernel.value.T

        class Encoder(brainstate.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed = brainstate.nn.Embedding(10, 2)
                self.linear_out = LinearTranspose(10, 2)
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


# ---------------------------------------------------------------------------
# Graph operation tests
# ---------------------------------------------------------------------------

class TestGraphOperation(unittest.TestCase):
    def test_flatten_basic(self):
        class MyNode(brainstate.graph.Node):
            def __init__(self):
                self.a = brainstate.nn.Linear(2, 3)
                self.b = brainstate.nn.Linear(3, 2)
                self.c = [brainstate.nn.Linear(1, 2), brainstate.nn.Linear(1, 3)]
                self.d = {'x': brainstate.nn.Linear(1, 3), 'y': brainstate.nn.Linear(1, 4)}

        graphdef, statetree = brainstate.graph.flatten(MyNode())
        assert statetree is not None

    def test_split(self):
        class Foo(brainstate.graph.Node):
            def __init__(self):
                self.a = brainstate.nn.Linear(2, 2)
                self.b = brainstate.nn.BatchNorm1d([10, 2])

        node = Foo()
        graphdef, params, others = brainstate.graph.treefy_split(node, brainstate.ParamState, ...)

        assert params is not None
        assert others is not None

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

        model = Model()
        # Add a dynamically tagged state inside a catch_new_states context
        with brainstate.catch_new_states('new'):
            model.dynamic = brainstate.ShortTermState(jnp.zeros(5))

        before_count = len(model.states())
        self.assertGreater(before_count, 0)

        popped = brainstate.graph.pop_states(model, 'new')
        self.assertEqual(len(model.states()), before_count - 1)
        self.assertFalse(hasattr(model, 'dynamic'))

    def test_treefy_split(self):
        class MLP(brainstate.graph.Node):
            def __init__(self, din: int, dmid: int, dout: int, n_layer: int = 3):
                self.input = brainstate.nn.Linear(din, dmid)
                self.layers = [brainstate.nn.Linear(dmid, dmid) for _ in range(n_layer)]
                self.output = brainstate.nn.Linear(dmid, dout)

            def __call__(self, x):
                x = jax.nn.relu(self.input(x))
                for layer in self.layers:
                    x = jax.nn.relu(layer(x))
                return self.output(x)

        model = MLP(2, 1, 3)
        graph_def, treefy_states = brainstate.graph.treefy_split(model)
        assert graph_def is not None
        assert treefy_states is not None

    def test_states(self):
        class MLP(brainstate.graph.Node):
            def __init__(self, din: int, dmid: int, dout: int, n_layer: int = 3):
                self.input = brainstate.nn.Linear(din, dmid)
                self.layers = [brainstate.nn.Linear(dmid, dmid) for _ in range(n_layer)]
                self.output = brainstate.nn.Linear(dmid, dout)

            def __call__(self, x):
                x = jax.nn.relu(self.input(x))
                for layer in self.layers:
                    x = jax.nn.relu(layer(x))
                return self.output(x)

        model = MLP(2, 1, 3)
        all_states = brainstate.graph.states(model)
        assert all_states is not None

        params, others = brainstate.graph.states(model, brainstate.ParamState, ...)
        assert params is not None
        assert others is not None


# ---------------------------------------------------------------------------
# RefMap tests
# ---------------------------------------------------------------------------

class TestRefMap(unittest.TestCase):
    def test_refmap_basic_operations(self):
        ref_map = brainstate.graph.RefMap()

        self.assertEqual(len(ref_map), 0)
        self.assertFalse(object() in ref_map)

        obj1 = object()
        obj2 = object()
        ref_map[obj1] = 'value1'
        ref_map[obj2] = 'value2'

        self.assertEqual(len(ref_map), 2)
        self.assertTrue(obj1 in ref_map)
        self.assertTrue(obj2 in ref_map)
        self.assertEqual(ref_map[obj1], 'value1')
        self.assertEqual(ref_map[obj2], 'value2')

        keys = list(ref_map)
        self.assertIn(obj1, keys)
        self.assertIn(obj2, keys)

        del ref_map[obj1]
        self.assertEqual(len(ref_map), 1)
        self.assertFalse(obj1 in ref_map)
        self.assertTrue(obj2 in ref_map)

    def test_refmap_initialization_with_mapping(self):
        obj1, obj2 = object(), object()
        ref_map = brainstate.graph.RefMap({obj1: 'value1', obj2: 'value2'})

        self.assertEqual(len(ref_map), 2)
        self.assertEqual(ref_map[obj1], 'value1')
        self.assertEqual(ref_map[obj2], 'value2')

    def test_refmap_initialization_with_iterable(self):
        obj1, obj2 = object(), object()
        ref_map = brainstate.graph.RefMap([(obj1, 'value1'), (obj2, 'value2')])

        self.assertEqual(len(ref_map), 2)
        self.assertEqual(ref_map[obj1], 'value1')
        self.assertEqual(ref_map[obj2], 'value2')

    def test_refmap_same_content_different_identity(self):
        list1 = [1, 2, 3]
        list2 = [1, 2, 3]

        ref_map = brainstate.graph.RefMap()
        ref_map[list1] = 'list1'
        ref_map[list2] = 'list2'

        self.assertEqual(len(ref_map), 2)
        self.assertEqual(ref_map[list1], 'list1')
        self.assertEqual(ref_map[list2], 'list2')

    def test_refmap_update(self):
        obj1, obj2, obj3 = object(), object(), object()
        ref_map = brainstate.graph.RefMap()
        ref_map[obj1] = 'value1'

        ref_map.update({obj2: 'value2', obj3: 'value3'})
        self.assertEqual(len(ref_map), 3)

        ref_map[obj1] = 'new_value1'
        self.assertEqual(ref_map[obj1], 'new_value1')

    def test_refmap_str_repr(self):
        ref_map = brainstate.graph.RefMap()
        obj = object()
        ref_map[obj] = 'value'

        str_repr = str(ref_map)
        self.assertIsInstance(str_repr, str)
        self.assertEqual(str_repr, repr(ref_map))


# ---------------------------------------------------------------------------
# Internal helper function tests
# ---------------------------------------------------------------------------

class TestHelperFunctions(unittest.TestCase):
    def test_is_state_leaf(self):
        from brainstate.graph._operation import _is_state_leaf

        state = brainstate.ParamState(1)
        treefy_state = state.to_state_ref()

        self.assertTrue(_is_state_leaf(treefy_state))
        self.assertFalse(_is_state_leaf(state))
        self.assertFalse(_is_state_leaf(1))
        self.assertFalse(_is_state_leaf(None))

    def test_is_node_leaf(self):
        from brainstate.graph._operation import _is_node_leaf

        state = brainstate.ParamState(1)

        self.assertTrue(_is_node_leaf(state))
        self.assertFalse(_is_node_leaf(1))
        self.assertFalse(_is_node_leaf(None))

    def test_is_node(self):
        from brainstate.graph._operation import _is_node

        node = brainstate.nn.Module()
        self.assertTrue(_is_node(node))
        self.assertTrue(_is_node([1, 2, 3]))
        self.assertTrue(_is_node({'a': 1}))
        self.assertFalse(_is_node(1))
        self.assertFalse(_is_node("string"))

    def test_is_pytree_node(self):
        from brainstate.graph._operation import _is_pytree_node

        self.assertTrue(_is_pytree_node([1, 2, 3]))
        self.assertTrue(_is_pytree_node({'a': 1}))
        self.assertTrue(_is_pytree_node((1, 2)))
        self.assertFalse(_is_pytree_node(1))
        self.assertFalse(_is_pytree_node("string"))
        self.assertFalse(_is_pytree_node(jnp.array([1, 2])))

    def test_is_graph_node(self):
        from brainstate.graph._operation import _is_graph_node

        class CustomNode:
            pass

        node = brainstate.nn.Module()
        self.assertTrue(_is_graph_node(node))
        self.assertFalse(_is_graph_node([1, 2, 3]))
        self.assertFalse(_is_graph_node({'a': 1}))
        self.assertFalse(_is_graph_node(CustomNode()))


# ---------------------------------------------------------------------------
# register_graph_node_type tests
# ---------------------------------------------------------------------------

class TestRegisterGraphNodeType(unittest.TestCase):
    def test_register_custom_node_type(self):
        from brainstate.graph._operation import _is_graph_node, _get_node_impl

        class CustomNode:
            def __init__(self):
                self.data = {}

        brainstate.graph.register_graph_node_type(
            CustomNode,
            flatten=lambda node: (list(node.data.items()), None),
            set_key=lambda node, key, value: node.data.__setitem__(key, value),
            pop_key=lambda node, key: node.data.pop(key),
            create_empty=lambda metadata: CustomNode(),
            clear=lambda node: node.data.clear(),
        )

        node = CustomNode()
        self.assertTrue(_is_graph_node(node))

        node.data['key1'] = 'value1'
        node_impl = _get_node_impl(node)

        items, metadata = node_impl.flatten(node)
        self.assertEqual(list(items), [('key1', 'value1')])

        node_impl.set_key(node, 'key2', 'value2')
        self.assertEqual(node.data['key2'], 'value2')

        value = node_impl.pop_key(node, 'key1')
        self.assertEqual(value, 'value1')
        self.assertNotIn('key1', node.data)

        new_node = node_impl.create_empty(None)
        self.assertIsInstance(new_node, CustomNode)

        node_impl.clear(node)
        self.assertEqual(node.data, {})


# ---------------------------------------------------------------------------
# HashableMapping tests
# ---------------------------------------------------------------------------

class TestHashableMapping(unittest.TestCase):
    def test_hashable_mapping_basic(self):
        from brainstate.graph._operation import HashableMapping

        hm = HashableMapping({'a': 1, 'b': 2})

        self.assertEqual(len(hm), 2)
        self.assertTrue('a' in hm)
        self.assertFalse('c' in hm)
        self.assertEqual(hm['a'], 1)
        self.assertEqual(set(hm), {'a', 'b'})

    def test_hashable_mapping_hash(self):
        from brainstate.graph._operation import HashableMapping

        hm1 = HashableMapping({'a': 1, 'b': 2})
        hm2 = HashableMapping({'a': 1, 'b': 2})
        hm3 = HashableMapping({'a': 1, 'b': 3})

        self.assertEqual(hash(hm1), hash(hm2))
        self.assertEqual(hm1, hm2)
        self.assertNotEqual(hm1, hm3)
        self.assertEqual(len({hm1, hm2, hm3}), 2)

    def test_hashable_mapping_from_iterable(self):
        from brainstate.graph._operation import HashableMapping

        hm = HashableMapping([('a', 1), ('b', 2)])

        self.assertEqual(len(hm), 2)
        self.assertEqual(hm['a'], 1)
        self.assertEqual(hm['b'], 2)


# ---------------------------------------------------------------------------
# NodeDef / NodeRef tests
# ---------------------------------------------------------------------------

class TestNodeDefAndNodeRef(unittest.TestCase):
    def test_noderef_creation(self):
        node_ref = brainstate.graph.NodeRef(type=brainstate.nn.Module, index=42)

        self.assertEqual(node_ref.type, brainstate.nn.Module)
        self.assertEqual(node_ref.index, 42)

    def test_nodedef_creation(self):
        from brainstate.graph._operation import HashableMapping

        nodedef = brainstate.graph.NodeDef.create(
            type=brainstate.nn.Module,
            index=1,
            attributes=('a', 'b'),
            subgraphs=[],
            static_fields=[('static', 'value')],
            leaves=[],
            metadata=None,
            index_mapping=None,
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
        nodedef = brainstate.graph.NodeDef.create(
            type=brainstate.nn.Module,
            index=1,
            attributes=(),
            subgraphs=[],
            static_fields=[],
            leaves=[],
            metadata=None,
            index_mapping={1: 2, 3: 4},
        )

        self.assertIsNotNone(nodedef.index_mapping)
        self.assertEqual(nodedef.index_mapping[1], 2)
        self.assertEqual(nodedef.index_mapping[3], 4)


# ---------------------------------------------------------------------------
# graphdef / clone tests
# ---------------------------------------------------------------------------

class TestGraphDefAndClone(unittest.TestCase):
    def test_graphdef_function(self):
        model = brainstate.nn.Linear(2, 3)
        gdef = brainstate.graph.graphdef(model)

        self.assertIsInstance(gdef, brainstate.graph.NodeDef)
        self.assertEqual(gdef.type, brainstate.nn.Linear)

        gdef2, _ = brainstate.graph.flatten(model)
        self.assertEqual(gdef, gdef2)

    def test_clone_function(self):
        model = brainstate.nn.Linear(2, 3)
        cloned = brainstate.graph.clone(model)

        self.assertIsInstance(cloned, brainstate.nn.Linear)
        self.assertIsNot(model, cloned)
        self.assertIsNot(model.weight, cloned.weight)

        original_weight = cloned.weight.value['weight'].copy()
        model.weight.value = jax.tree.map(lambda x: x + 1, model.weight.value)
        self.assertTrue(jnp.allclose(cloned.weight.value['weight'], original_weight))

    def test_clone_preserves_shared_variables(self):
        class SharedModel(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.shared_weight = brainstate.ParamState(jnp.ones((2, 2)))
                self.layer1 = brainstate.nn.Linear(2, 2)
                self.layer2 = brainstate.nn.Linear(2, 2)
                self.layer2.weight = self.layer1.weight

        model = SharedModel()
        cloned = brainstate.graph.clone(model)

        self.assertIs(cloned.layer1.weight, cloned.layer2.weight)
        self.assertIsNot(cloned.layer1.weight, model.layer1.weight)


# ---------------------------------------------------------------------------
# nodes() function tests
# ---------------------------------------------------------------------------

class TestNodesFunction(unittest.TestCase):
    def test_nodes_without_filters(self):
        class Model(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = brainstate.nn.Linear(2, 3)
                self.b = brainstate.nn.Linear(3, 4)

        model = Model()
        all_nodes = brainstate.graph.nodes(model)

        self.assertIsInstance(all_nodes, brainstate.util.FlattedDict)
        paths = [path for path, _ in all_nodes.items()]
        self.assertIn(('a',), paths)
        self.assertIn(('b',), paths)
        self.assertIn((), paths)

    def test_nodes_with_filter(self):
        class CustomModule(brainstate.nn.Module):
            pass

        class Model(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = brainstate.nn.Linear(2, 3)
                self.custom = CustomModule()

        model = Model()
        linear_nodes = brainstate.graph.nodes(
            model,
            lambda path, node: isinstance(node, brainstate.nn.Linear),
        )

        self.assertIsInstance(linear_nodes, brainstate.util.FlattedDict)
        nodes_list = list(linear_nodes.values())
        self.assertEqual(len(nodes_list), 1)
        self.assertIsInstance(nodes_list[0], brainstate.nn.Linear)

    def test_nodes_with_hierarchy(self):
        class Model(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = brainstate.nn.Linear(2, 3)
                self.layer1.sublayer = brainstate.nn.Linear(3, 3)

        model = Model()
        level1_nodes = brainstate.graph.nodes(model, allowed_hierarchy=(1, 1))
        paths = [path for path, _ in level1_nodes.items()]

        self.assertIn(('layer1',), paths)
        self.assertNotIn(('layer1', 'sublayer'), paths)


# ---------------------------------------------------------------------------
# Static tests
# ---------------------------------------------------------------------------

class TestStatic(unittest.TestCase):
    def test_static_basic(self):
        from brainstate.graph._operation import Static

        value = {'key': 'value'}
        static = Static(value)

        self.assertEqual(static.value, value)
        self.assertIs(static.value, value)

    def test_static_is_pytree_leaf(self):
        from brainstate.graph._operation import Static

        static = Static({'key': 'value'})
        leaves, _ = jax.tree_util.tree_flatten(static)
        self.assertEqual(len(leaves), 0)

        tree = {'a': 1, 'b': static, 'c': [2, 3]}
        leaves, _ = jax.tree_util.tree_flatten(tree)
        self.assertNotIn(static, leaves)

    def test_static_equality_and_hash(self):
        from brainstate.graph._operation import Static

        static1 = Static(42)
        static2 = Static(42)
        static3 = Static(43)

        self.assertEqual(static1, static2)
        self.assertNotEqual(static1, static3)
        self.assertEqual(hash(static1), hash(static2))
        self.assertNotEqual(hash(static1), hash(static3))


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------

class TestErrorHandling(unittest.TestCase):
    def test_flatten_with_invalid_ref_index(self):
        model = brainstate.nn.Linear(2, 3)

        with self.assertRaises(TypeError):
            brainstate.graph.flatten(model, ref_index={})

    def test_unflatten_with_invalid_graphdef(self):
        state = brainstate.util.NestedDict({})

        with self.assertRaises(TypeError):
            brainstate.graph.unflatten("not_a_graphdef", state)

    def test_pop_states_without_filters(self):
        model = brainstate.nn.Linear(2, 3)

        with self.assertRaises(ValueError) as context:
            brainstate.graph.pop_states(model)
        self.assertIn('Expected at least one filter', str(context.exception))

    def test_update_states_immutable_node(self):
        node = (1, 2, brainstate.ParamState(3))
        state = brainstate.util.NestedDict({0: brainstate.TreefyState(int, 10)})

        with self.assertRaises(ValueError):
            brainstate.graph.update_states(node, state)

    def test_get_node_impl_with_state(self):
        from brainstate.graph._operation import _get_node_impl

        state = brainstate.ParamState(1)

        with self.assertRaises(ValueError) as context:
            _get_node_impl(state)
        self.assertIn('State is not a node', str(context.exception))

    def test_split_with_non_exhaustive_filters(self):
        from brainstate.graph._operation import _split_flatted

        flatted = [(('a',), 1), (('b',), 2)]
        filters = (lambda path, value: value == 1,)

        with self.assertRaises(ValueError) as context:
            _split_flatted(flatted, filters)
        self.assertIn('Non-exhaustive filters', str(context.exception))

    def test_invalid_filter_order(self):
        from brainstate.graph._operation import _filters_to_predicates

        filters = (..., lambda p, v: True)

        with self.assertRaises(ValueError) as context:
            _filters_to_predicates(filters)
        self.assertIn('can only be used as the last filters', str(context.exception))


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestIntegration(unittest.TestCase):
    def test_complex_graph_operations(self):
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
                self.layer2.shared_ref = self.shared
                self.nested = {
                    'a': brainstate.nn.Linear(4, 5),
                    'b': [brainstate.nn.Linear(5, 6), self.shared],
                }

        model = ComplexModel()
        graphdef, state = brainstate.graph.treefy_split(model)
        reconstructed = brainstate.graph.treefy_merge(graphdef, state)

        self.assertIs(reconstructed.shared, reconstructed.layer2.shared_ref)
        self.assertIs(reconstructed.shared, reconstructed.nested['b'][1])

        new_state = jax.tree.map(lambda x: x * 2, state)
        brainstate.graph.update_states(model, new_state)

        self.assertTrue(jnp.allclose(
            model.shared.weight.value,
            jnp.ones((2, 2)) * 2,
        ))

    def test_recursive_structure(self):
        class RecursiveModule(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = brainstate.ParamState(1)
                self.child = None

        parent = RecursiveModule()
        child = RecursiveModule()
        parent.child = child
        child.child = parent

        graphdef, state = brainstate.graph.treefy_split(parent)
        reconstructed = brainstate.graph.treefy_merge(graphdef, state)

        self.assertIsNotNone(reconstructed.child)
        self.assertIs(reconstructed.child.child, reconstructed)


# ---------------------------------------------------------------------------
# Threading tests
# ---------------------------------------------------------------------------

class SimpleModule(brainstate.nn.Module):
    pass


class TestThreading(parameterized.TestCase):
    @parameterized.parameters((SimpleModule,),)
    def test_threading(self, module_fn: Callable[[], brainstate.nn.Module]):
        x = module_fn()

        class MyThread(Thread):
            def run(self) -> None:
                brainstate.graph.treefy_split(x)

        thread = MyThread()
        thread.start()
        thread.join()


if __name__ == '__main__':
    absltest.main()
