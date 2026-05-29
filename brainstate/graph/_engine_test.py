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

"""Unit tests for the flat-IR graph engine (``_reftrack``/``_graphdef``/
``_walk``/``_flatten``/``_operations``)."""

import dataclasses
import unittest

import jax
import jax.numpy as jnp

import brainstate
import brainstate.util
from brainstate.graph._reftrack import RefMap
from brainstate.graph._graphdef import (
    PytreeType, NodeEdge, StateEdge, StateLeafEdge, PytreeEdge, StaticEdge,
    NodeSpec, GraphDef,
)


# ---------------------------------------------------------------------------
# A registrable graph node backed by a plain dict (used across the tests).
# ---------------------------------------------------------------------------

class _Dict:
    def __init__(self):
        self.data = {}


_DICT_REGISTERED = False


def _register_dict():
    global _DICT_REGISTERED
    if _DICT_REGISTERED:
        return
    from brainstate.graph._walk import register_graph_node_type
    register_graph_node_type(
        _Dict,
        flatten=lambda n: (sorted(n.data.items()), (_Dict,)),
        set_key=lambda n, k, v: n.data.__setitem__(k, v),
        pop_key=lambda n, k: n.data.pop(k),
        create_empty=lambda meta: _Dict(),
        clear=lambda n: n.data.clear(),
    )
    _DICT_REGISTERED = True


class TestRefMap(unittest.TestCase):
    def test_identity_keyed(self):
        a, b = [1, 2, 3], [1, 2, 3]  # equal but distinct objects
        m = RefMap()
        m[a] = 'a'
        m[b] = 'b'
        self.assertEqual(len(m), 2)          # not collapsed by ==
        self.assertEqual(m[a], 'a')
        self.assertEqual(m[b], 'b')
        self.assertIn(a, m)
        del m[a]
        self.assertNotIn(a, m)
        self.assertEqual(len(m), 1)

    def test_init_and_iter(self):
        o1, o2 = object(), object()
        m = RefMap([(o1, 1), (o2, 2)])
        self.assertEqual({m[o1], m[o2]}, {1, 2})
        self.assertEqual(sorted(m[k] for k in m), [1, 2])
        self.assertEqual(str(m), repr(m))


class TestEdgesAndSpec(unittest.TestCase):
    def test_edges_are_frozen_and_hashable(self):
        for e in [NodeEdge(1), StateEdge(2, ('a',), int), StateLeafEdge(('b',)),
                  PytreeEdge(None, ()), StaticEdge(7)]:
            hash(e)                       # hashable
            with self.assertRaises(dataclasses.FrozenInstanceError):
                e.__setattr__('index', 9)

    def test_nodespec_hash_and_eq(self):
        s1 = NodeSpec(int, 0, (int,), (('x', StaticEdge(1)),))
        s2 = NodeSpec(int, 0, (int,), (('x', StaticEdge(1)),))
        s3 = NodeSpec(int, 0, (int,), (('x', StaticEdge(2)),))
        self.assertEqual(s1, s2)
        self.assertEqual(hash(s1), hash(s2))
        self.assertNotEqual(s1, s3)


class TestGraphDef(unittest.TestCase):
    def _gdef(self, val):
        spec = NodeSpec(int, 0, (int,), (('x', StaticEdge(val)),))
        return GraphDef(NodeEdge(0), (spec,))

    def test_hash_eq_cached(self):
        g1, g2, g3 = self._gdef(1), self._gdef(1), self._gdef(2)
        self.assertEqual(g1, g2)
        self.assertEqual(hash(g1), hash(g2))
        self.assertNotEqual(g1, g3)
        self.assertEqual(len({g1, g2, g3}), 2)
        self.assertEqual(g1.__hash__(), g1.__hash__())

    def test_type_and_index_shims(self):
        g = self._gdef(1)
        self.assertEqual(g.type, int)
        self.assertEqual(g.index, 0)

    def test_pytree_root_type(self):
        g = GraphDef(PytreeEdge(None, ()), ())
        self.assertIs(g.type, PytreeType)
        self.assertEqual(g.index, 0)

    def test_jax_static_roundtrip(self):
        g = self._gdef(1)
        leaves, treedef = jax.tree_util.tree_flatten(g)
        self.assertEqual(leaves, [])               # static: no leaves
        g2 = jax.tree_util.tree_unflatten(treedef, [])
        self.assertEqual(g, g2)

    def test_repr_is_string(self):
        self.assertIsInstance(repr(self._gdef(1)), str)


class TestWalkRegistry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _register_dict()

    def test_predicates(self):
        from brainstate.graph._walk import (
            _is_node, _is_graph_node, _is_pytree_node, _is_state_leaf, _is_node_leaf,
        )
        self.assertTrue(_is_graph_node(_Dict()))
        self.assertTrue(_is_node([1, 2]))
        self.assertTrue(_is_pytree_node({'a': 1}))
        self.assertFalse(_is_node(1))
        self.assertFalse(_is_pytree_node(jnp.array([1, 2])))
        st = brainstate.ParamState(1)
        self.assertTrue(_is_node_leaf(st))
        self.assertTrue(_is_state_leaf(st.to_state_ref()))
        self.assertFalse(_is_state_leaf(st))

    def test_get_node_impl(self):
        from brainstate.graph._walk import (
            _get_node_impl, get_node_impl_for_type, GraphNodeImpl, PyTreeNodeImpl,
        )
        self.assertIsInstance(_get_node_impl(_Dict()), GraphNodeImpl)
        self.assertIsInstance(_get_node_impl([1]), PyTreeNodeImpl)
        self.assertIs(get_node_impl_for_type(PytreeType).type, PytreeType)
        with self.assertRaisesRegex(ValueError, 'State is not a node'):
            _get_node_impl(brainstate.ParamState(1))

    def test_static_is_pytree_leaf(self):
        from brainstate.graph._walk import Static
        s = Static({'k': 'v'})
        self.assertEqual(jax.tree_util.tree_leaves(s), [])
        self.assertEqual(s, Static({'k': 'v'}))


class TestWalkKernel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _register_dict()

    def _model(self):
        root = _Dict()
        a = _Dict()
        a.data['w'] = brainstate.ParamState(jnp.ones(2))
        root.data['a'] = a
        root.data['shared'] = a               # shared graph node
        root.data['n'] = 5                    # static leaf
        return root, a

    def test_iter_node_dedups_shared(self):
        from brainstate.graph._walk import iter_node
        root, a = self._model()
        nodes = list(iter_node(root))
        objs = [n for _, n in nodes]
        self.assertIn(root, objs)
        self.assertEqual(sum(1 for o in objs if o is a), 1)   # visited once

    def test_iter_leaf_yields_leaves(self):
        from brainstate.graph._walk import iter_leaf
        root, a = self._model()
        leaves = list(iter_leaf(root))
        vals = [v for _, v in leaves]
        self.assertTrue(any(isinstance(v, brainstate.ParamState) for v in vals))
        self.assertIn(5, vals)

    def test_hierarchy_bounds(self):
        from brainstate.graph._walk import iter_node
        root, a = self._model()
        top = list(iter_node(root, allowed_hierarchy=(0, 0)))
        self.assertEqual([n for _, n in top], [root])         # depth 0 only


class TestFlatten(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _register_dict()

    def test_pytree_root_and_refcount(self):
        from brainstate.graph._flatten import flatten
        a = {'a': 1, 'b': brainstate.ParamState(2)}
        g = [a, 3, a, brainstate.ParamState(4)]            # shared dict + states
        refmap = RefMap()
        gdef, states = flatten(g, ref_index=refmap, treefy_state=True)
        self.assertIsInstance(gdef, GraphDef)
        self.assertIsInstance(states, brainstate.util.NestedDict)
        self.assertEqual(len(refmap), 2)                  # 2 unique states, 0 graph nodes
        self.assertIn(a['b'], refmap)
        self.assertIn(g[3], refmap)
        self.assertIsInstance(gdef.root, PytreeEdge)       # list root is a pytree

    def test_treefy_state_flag(self):
        from brainstate.graph._flatten import flatten
        g = [brainstate.ParamState(1)]
        _, s_true = flatten(g, treefy_state=True)
        _, s_false = flatten(g, treefy_state=False)
        self.assertIsInstance(s_true[0], brainstate.TreefyState)
        self.assertIsInstance(s_false[0], brainstate.State)

    def test_shared_graph_node_backref(self):
        from brainstate.graph._flatten import flatten
        a = _Dict()
        a.data['w'] = brainstate.ParamState(1)
        root = _Dict()
        root.data['x'] = a
        root.data['y'] = a
        gdef, _ = flatten(root)
        root_spec = [s for s in gdef.node_specs if s.index == gdef.index][0]
        edges = dict(root_spec.fields)
        self.assertIsInstance(edges['x'], NodeEdge)
        self.assertIsInstance(edges['y'], NodeEdge)
        self.assertEqual(edges['x'].index, edges['y'].index)   # same node

    def test_invalid_ref_index_type(self):
        from brainstate.graph._flatten import flatten
        with self.assertRaises(TypeError):
            flatten(_Dict(), ref_index={})

    def test_unhashable_static_deferred(self):
        from brainstate.graph._flatten import flatten
        n = _Dict()
        n.data['bad'] = jnp.ones(3)   # bare array attr -> unhashable static
        gdef, _ = flatten(n)          # flatten succeeds (deferred, like legacy engine)
        with self.assertRaisesRegex(TypeError, 'not hashable'):
            hash(gdef)                # hash-check fires only when hashed


class TestUnflatten(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _register_dict()

    def test_roundtrip_pytree_and_states(self):
        from brainstate.graph._flatten import flatten, unflatten
        a = {'a': 1, 'b': brainstate.ParamState(2)}
        g = [a, 3, a, brainstate.ParamState(4)]
        gdef, states = flatten(g)
        g2 = unflatten(gdef, states)
        self.assertIsNot(g2[0], g2[2])                       # pytrees re-expanded
        self.assertIs(g2[0]['b'], g2[2]['b'])                # shared state dedup'd
        self.assertEqual(g2[0]['b'].value, 2)

    def test_roundtrip_shared_graph_node(self):
        from brainstate.graph._flatten import flatten, unflatten
        a = _Dict()
        a.data['w'] = brainstate.ParamState(1)
        root = _Dict()
        root.data['x'] = a
        root.data['y'] = a
        gdef, states = flatten(root)
        r2 = unflatten(gdef, states)
        self.assertIs(r2.data['x'], r2.data['y'])            # shared node round-trips

    def test_roundtrip_cycle(self):
        from brainstate.graph._flatten import flatten, unflatten
        p = _Dict()
        c = _Dict()
        p.data['child'] = c
        c.data['parent'] = p                                 # cycle
        gdef, states = flatten(p)
        p2 = unflatten(gdef, states)
        self.assertIs(p2.data['child'].data['parent'], p2)

    def test_root_pytree_holds_defining_state_referenced_by_node(self):
        from brainstate.graph._flatten import flatten, unflatten
        s = brainstate.ParamState(7)
        n = _Dict()
        n.data['ref'] = s
        g = [s, n]                                           # state defined at root pytree
        gdef, states = flatten(g)
        g2 = unflatten(gdef, states)
        self.assertIs(g2[0], g2[1].data['ref'])              # Pass-0 makes this work

    def test_invalid_graphdef(self):
        from brainstate.graph._flatten import unflatten
        with self.assertRaises(TypeError):
            unflatten("nope", brainstate.util.NestedDict({}))


class TestOps1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _register_dict()

    def _model(self):
        n = _Dict()
        n.data['w'] = brainstate.ParamState(jnp.ones((2, 2)))
        n.data['c'] = brainstate.ShortTermState(jnp.zeros(2))
        return n

    def test_split_merge_roundtrip(self):
        from brainstate.graph._operations import treefy_split, treefy_merge
        n = self._model()
        gdef, params, others = treefy_split(n, brainstate.ParamState, ...)
        n2 = treefy_merge(gdef, params, others)
        self.assertIsInstance(n2.data['w'], brainstate.ParamState)
        self.assertTrue(jnp.allclose(n2.data['w'].value, jnp.ones((2, 2))))

    def test_treefy_states_and_graphdef(self):
        from brainstate.graph._operations import treefy_states, graphdef
        from brainstate.graph._flatten import flatten
        n = self._model()
        st = treefy_states(n)
        self.assertIn('w', st)
        self.assertEqual(graphdef(n), flatten(n)[0])

    def test_clone(self):
        from brainstate.graph._operations import clone
        n = self._model()
        c = clone(n)
        self.assertIsNot(c, n)
        self.assertIsNot(c.data['w'], n.data['w'])
        self.assertTrue(jnp.allclose(c.data['w'].value, n.data['w'].value))


class TestOps2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _register_dict()

    def _model(self):
        n = _Dict()
        n.data['w'] = brainstate.ParamState(1)
        n.data['c'] = brainstate.ShortTermState(2)
        child = _Dict()
        child.data['p'] = brainstate.ParamState(3)
        n.data['child'] = child
        return n

    def test_states_all_and_filtered(self):
        from brainstate.graph._operations import states
        n = self._model()
        allst = states(n)
        self.assertIsInstance(allst, brainstate.util.FlattedDict)
        params, others = states(n, brainstate.ParamState, ...)
        self.assertTrue(all(isinstance(v, brainstate.ParamState) for v in params.values()))

    def test_nodes_all_and_hierarchy(self):
        from brainstate.graph._operations import nodes
        n = self._model()
        alln = nodes(n)
        self.assertIn((), [p for p in alln])               # root present
        lvl1 = nodes(n, allowed_hierarchy=(1, 1))
        self.assertIn(('child',), [p for p in lvl1])


class TestOps3(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _register_dict()

    def test_pop_states_by_type(self):
        from brainstate.graph._operations import pop_states
        n = _Dict()
        n.data['w'] = brainstate.ParamState(1)
        n.data['c'] = brainstate.ShortTermState(2)
        popped = pop_states(n, brainstate.ShortTermState)
        self.assertNotIn('c', n.data)
        self.assertIn('w', n.data)
        self.assertEqual(len(popped.to_flat()), 1)

    def test_pop_states_no_filter_raises(self):
        from brainstate.graph._operations import pop_states
        with self.assertRaisesRegex(ValueError, 'at least one filter'):
            pop_states(_Dict())

    def test_update_states(self):
        from brainstate.graph._operations import treefy_states, update_states
        n = _Dict()
        n.data['w'] = brainstate.ParamState(jnp.ones(2))
        st = treefy_states(n)
        st = jax.tree.map(lambda x: x * 2, st)
        update_states(n, st)
        self.assertTrue(jnp.allclose(n.data['w'].value, jnp.ones(2) * 2))


if __name__ == '__main__':
    unittest.main()
