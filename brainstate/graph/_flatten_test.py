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

"""Tests for the low-level ``flatten`` / ``unflatten`` encode-decode routines.

Covers the encoder's State / bare-``TreefyState`` / static-attribute edges, the
non-node-root error, and the decoder's ``index_ref_cache`` "update in place"
path (state refresh, shell reuse, and type-mismatch detection).
"""

import unittest

import jax.numpy as jnp

import brainstate
from brainstate.graph import flatten, unflatten, GraphDef
from brainstate.graph import _flatten
from brainstate._state import TreefyState


class _Cell(brainstate.graph.Node):
    """A graph node with a parameter state and a static attribute."""

    def __init__(self, n=2, tag="cell"):
        self.w = brainstate.ParamState(jnp.ones((n,)))
        self.tag = tag  # static (inline) attribute


class TestFlattenEncode(unittest.TestCase):
    """Encoder edge cases for ``flatten``."""

    def test_basic_flatten_returns_graphdef_and_nesteddict(self):
        """``flatten`` returns a ``GraphDef`` plus a ``NestedDict`` state."""
        gd, state = flatten(_Cell())
        self.assertIsInstance(gd, GraphDef)
        self.assertIsInstance(state, brainstate.util.NestedDict)

    def test_flatten_non_node_root_raises(self):
        """Flattening a bare ``State`` (not a node/pytree) raises ``ValueError``."""
        with self.assertRaises(ValueError):
            flatten(brainstate.ParamState(jnp.ones((2,))))

    def test_flatten_bad_ref_index_type_raises(self):
        """A non-``RefMap`` ``ref_index`` raises ``TypeError``."""
        with self.assertRaises(TypeError):
            flatten(_Cell(), ref_index={})

    def test_static_attribute_roundtrips(self):
        """A non-state static attribute is preserved verbatim through a round-trip."""
        gd, state = flatten(_Cell(tag="hello"))
        rebuilt = unflatten(gd, state)
        self.assertEqual(rebuilt.tag, "hello")

    def test_bare_treefystate_leaf_roundtrips(self):
        """A bare ``TreefyState`` attribute is encoded as a leaf and restored."""
        node = _Cell()
        node.ref = brainstate.ParamState(jnp.arange(3.0)).to_state_ref()
        self.assertIsInstance(node.ref, TreefyState)
        gd, state = flatten(node)
        rebuilt = unflatten(gd, state)
        self.assertIsInstance(rebuilt.ref, TreefyState)
        self.assertTrue(bool(jnp.allclose(rebuilt.ref.value, jnp.arange(3.0))))


class TestUnflattenDecode(unittest.TestCase):
    """Decoder behaviour for ``unflatten``."""

    def test_unflatten_bad_graphdef_raises(self):
        """A non-``GraphDef`` first argument raises ``TypeError``."""
        _, state = flatten(_Cell())
        with self.assertRaises(TypeError):
            unflatten("not-a-graphdef", state)

    def test_non_treefy_state_roundtrips(self):
        """``treefy_state=False`` keeps raw ``State`` objects in the mapping."""
        node = _Cell()
        gd, state = flatten(node, treefy_state=False)
        rebuilt = unflatten(gd, state)
        self.assertTrue(bool(jnp.allclose(rebuilt.w.value, jnp.ones((2,)))))


class TestIndexRefCacheUpdate(unittest.TestCase):
    """The ``index_ref_cache`` "update existing objects in place" path."""

    def _first_build(self):
        """Build a node once and return (graphdef, populated index_ref)."""
        node = _Cell()
        gd, state = flatten(node)
        index_ref = {}
        rebuilt = unflatten(gd, state, index_ref=index_ref)
        return gd, rebuilt, index_ref

    def test_cache_reuses_shell_and_updates_state(self):
        """A second unflatten with a cache reuses the node and updates its state."""
        gd, rebuilt, cache = self._first_build()
        # New values, identical structure -> matching global indices.
        node2 = _Cell()
        node2.w.value = jnp.full((2,), 9.0)
        gd2, state2 = flatten(node2)
        out = unflatten(gd2, state2, index_ref={}, index_ref_cache=cache)
        self.assertIs(out, rebuilt)  # node shell reused from the cache
        self.assertTrue(bool(jnp.allclose(out.w.value, 9.0)))  # state updated in place

    def test_cache_update_non_treefy_state(self):
        """The cache update path also handles raw (non-treefy) ``State`` values."""
        gd, rebuilt, cache = self._first_build()
        node2 = _Cell()
        node2.w.value = jnp.full((2,), 5.0)
        gd2, state2 = flatten(node2, treefy_state=False)
        out = unflatten(gd2, state2, index_ref={}, index_ref_cache=cache)
        self.assertIs(out, rebuilt)
        self.assertTrue(bool(jnp.allclose(out.w.value, 5.0)))

    def test_cache_type_mismatch_raises(self):
        """A cached object of the wrong type at a node index raises ``ValueError``."""
        gd, _, cache = self._first_build()
        node_index = gd.node_specs[0].index
        gd2, state2 = flatten(_Cell())
        bad_cache = {node_index: object()}
        with self.assertRaises(ValueError):
            unflatten(gd2, state2, index_ref={}, index_ref_cache=bad_cache)


class TestSharedStateAndMissing(unittest.TestCase):
    """Shared-state deduplication and missing-state error handling."""

    def test_shared_state_single_index(self):
        """The same ``State`` referenced twice collapses to one index."""

        class Twin(brainstate.graph.Node):
            def __init__(self, st):
                self.a = st
                self.b = st  # same object as ``a``

        shared = brainstate.ParamState(jnp.ones((2,)))
        gd, state = flatten(Twin(shared))
        rebuilt = unflatten(gd, state)
        self.assertIs(rebuilt.a, rebuilt.b)

    def test_missing_state_path_raises(self):
        """A ``StateEdge`` whose path is absent (no cache) raises ``ValueError``."""
        gd, _ = flatten(_Cell())
        empty = brainstate.util.NestedDict()
        with self.assertRaises(ValueError):
            unflatten(gd, empty)


class TestIndexRefCacheEdgeCases(unittest.TestCase):
    """Less-common branches of the ``index_ref_cache`` decode path."""

    @staticmethod
    def _state_index(index_ref):
        from brainstate._state import State
        return next(i for i, v in index_ref.items() if isinstance(v, State))

    def test_cache_restores_unwritten_state(self):
        """An unmodified (non-written) cached ``State`` is refreshed via restore."""
        node = _Cell()
        gd, _ = flatten(node)
        ref = {}
        rebuilt = unflatten(gd, flatten(node)[1], index_ref=ref)
        # A freshly-built node whose value was never reassigned -> restore path.
        fresh = _Cell()
        gd2, state2 = flatten(fresh, treefy_state=False)
        out = unflatten(gd2, state2, index_ref={}, index_ref_cache=ref)
        self.assertIs(out, rebuilt)
        self.assertTrue(bool(jnp.allclose(out.w.value, jnp.ones((2,)))))

    def test_cache_non_state_at_state_index_raises(self):
        """A cached non-``State`` object at a state index raises ``ValueError``."""
        node = _Cell()
        gd, state = flatten(node)
        ref = {}
        unflatten(gd, flatten(node)[1], index_ref=ref)
        bad_cache = {self._state_index(ref): object()}
        with self.assertRaises(ValueError):
            unflatten(gd, state, index_ref={}, index_ref_cache=bad_cache)

    def test_cache_supplies_state_absent_from_mapping(self):
        """A state missing from the mapping but present in the cache is reused."""
        node = _Cell()
        gd, _ = flatten(node)
        ref = {}
        unflatten(gd, flatten(node)[1], index_ref=ref)
        empty = brainstate.util.NestedDict()
        out = unflatten(gd, empty, index_ref={}, index_ref_cache=ref)
        self.assertTrue(bool(jnp.allclose(out.w.value, jnp.ones((2,)))))


class TestStateLeafEdgeMissingPath(unittest.TestCase):
    """A ``StateLeafEdge`` whose path is absent from the mapping must raise a
    friendly ``ValueError`` (mirroring ``StateEdge``), not a bare ``KeyError``.

    ``StateLeafEdge`` is a public, exported IR type, so a hand-built or
    deserialized ``GraphDef`` can reference a path missing from the supplied
    state mapping. The decoder used a bare ``flat_states[path]`` lookup there.
    """

    def test_missing_stateleafedge_path_raises_valueerror(self):
        from brainstate.graph import NodeSpec, NodeEdge, StateLeafEdge
        from brainstate.util import NestedDict

        # Borrow a valid node spec, then point its single field at a missing path.
        template, _ = flatten(_Cell())
        spec0 = template.node_specs[0]
        spec = NodeSpec(
            spec0.type, spec0.index, spec0.metadata,
            (('w', StateLeafEdge(('definitely_missing',))),),
        )
        gd = GraphDef(NodeEdge(spec0.index), (spec,))

        with self.assertRaises(ValueError) as ctx:
            unflatten(gd, NestedDict.from_flat({}))
        msg = str(ctx.exception)
        self.assertIn('definitely_missing', msg)
        self.assertIn('state mapping', msg)


class TestFormatPath(unittest.TestCase):
    """The ``_format_path`` helper used in error messages."""

    def test_empty_path_is_root(self):
        """An empty path renders as ``<root>``."""
        self.assertEqual(_flatten._format_path(()), "<root>")

    def test_nested_path_is_dotted(self):
        """A multi-part path renders dotted."""
        self.assertEqual(_flatten._format_path(("a", "b", "c")), "a.b.c")


if __name__ == "__main__":
    unittest.main()
