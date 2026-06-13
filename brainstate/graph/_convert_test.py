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

"""Tests for ``brainstate.graph`` graph<->pytree conversion utilities.

Exercises :func:`graph_to_tree` / :func:`tree_to_graph` round-trips, the
:class:`NodeStates` pytree wrapper, shared-reference aliasing (consistent,
inconsistent, and disabled), ``map_non_graph_nodes``, and integration with a
JAX transform.
"""

import unittest

import jax
import jax.numpy as jnp

import brainstate
from brainstate.graph import graph_to_tree, tree_to_graph, NodeStates
from brainstate.graph import _convert


class _Linear(brainstate.graph.Node):
    """A tiny graph node holding two parameter states."""

    def __init__(self, din, dout):
        self.w = brainstate.ParamState(jnp.ones((din, dout)))
        self.b = brainstate.ParamState(jnp.zeros((dout,)))


class TestGraphToTreeRoundtrip(unittest.TestCase):
    """``graph_to_tree`` then ``tree_to_graph`` reconstructs node values."""

    def test_basic_roundtrip(self):
        """A single node round-trips by parameter value and type."""
        node = _Linear(2, 3)
        tree, found_states = graph_to_tree(node)
        rebuilt = tree_to_graph(tree)
        self.assertIsInstance(rebuilt, _Linear)
        self.assertTrue(bool(jnp.allclose(rebuilt.w.value, node.w.value)))
        self.assertTrue(bool(jnp.allclose(rebuilt.b.value, node.b.value)))

    def test_find_states_returned(self):
        """``graph_to_tree`` returns a mapping of discovered states."""
        node = _Linear(2, 3)
        _, found_states = graph_to_tree(node)
        # At least the two ParamStates should be discoverable.
        self.assertGreaterEqual(len(jax.tree.leaves(found_states)), 2)

    def test_tree_is_pure_pytree(self):
        """``graph_to_tree`` output flattens to array leaves via ``jax.tree``."""
        node = _Linear(2, 3)
        tree, _ = graph_to_tree(node)
        leaves = jax.tree.leaves(tree)
        self.assertGreaterEqual(len(leaves), 2)
        self.assertTrue(all(hasattr(x, "shape") for x in leaves))

    def test_nodestates_is_pytree(self):
        """The packed tree re-flattens/unflattens to the same structure."""
        node = _Linear(2, 3)
        tree, _ = graph_to_tree(node)
        flat, treedef = jax.tree.flatten(tree)
        rebuilt_tree = jax.tree.unflatten(treedef, flat)
        self.assertEqual(jax.tree.structure(tree), jax.tree.structure(rebuilt_tree))

    def test_shared_state_aliasing_preserved(self):
        """A state shared in two slots stays a single object after round-trip."""
        shared = brainstate.ParamState(jnp.ones((2,)))
        g = {"a": shared, "b": shared}
        tree, _ = graph_to_tree(g)
        rebuilt = tree_to_graph(tree)
        self.assertIs(rebuilt["a"], rebuilt["b"])

    def test_map_non_graph_nodes_custom_fns(self):
        """``map_non_graph_nodes=True`` routes plain leaves through custom fns.

        The default ``split_fn`` only understands graph nodes, so exercising the
        non-graph-leaf branch requires a custom ``split_fn`` / ``merge_fn`` pair.
        """
        g = {"x": jnp.zeros((3,))}
        tree, _ = graph_to_tree(
            g, map_non_graph_nodes=True,
            split_fn=lambda ctx, path, prefix, leaf: leaf + 1.0,
        )
        self.assertTrue(bool(jnp.allclose(tree["x"], jnp.ones((3,)))))
        back = tree_to_graph(
            tree, map_non_graph_nodes=True,
            merge_fn=lambda ctx, path, prefix, leaf: leaf - 1.0,
        )
        self.assertTrue(bool(jnp.allclose(back["x"], jnp.zeros((3,)))))

    def test_roundtrip_usable_in_jit(self):
        """The pure pytree from ``graph_to_tree`` flows through ``jit``."""
        node = _Linear(2, 3)
        tree, _ = graph_to_tree(node)

        @brainstate.transform.jit
        def double(t):
            return jax.tree.map(lambda x: x * 2, t)

        out = double(tree)
        self.assertEqual(jax.tree.structure(out), jax.tree.structure(tree))


class TestNodeStates(unittest.TestCase):
    """The :class:`NodeStates` pytree wrapper and its constructors/properties."""

    def test_from_split_exposes_graphdef_and_state(self):
        """``from_split`` packs a graphdef plus exactly one state mapping."""
        node = _Linear(2, 3)
        graphdef, state = brainstate.graph.treefy_split(node)
        ns = NodeStates.from_split(graphdef, state)
        self.assertIs(ns.graphdef, graphdef)
        self.assertIs(ns.state, state)
        self.assertEqual(len(ns.states), 1)

    def test_graphdef_property_raises_without_graphdef(self):
        """Accessing ``graphdef`` when none was packed raises ``ValueError``."""
        node = _Linear(2, 3)
        _, state = brainstate.graph.treefy_split(node)
        ns = NodeStates.from_states(state)
        with self.assertRaises(ValueError):
            _ = ns.graphdef

    def test_state_property_requires_exactly_one(self):
        """``state`` raises when there is not exactly one state mapping."""
        node = _Linear(2, 3)
        _, state = brainstate.graph.treefy_split(node)
        ns = NodeStates.from_states(state, state)
        with self.assertRaises(ValueError):
            _ = ns.state

    def test_from_prefixes(self):
        """``from_prefixes`` stores the given prefixes as the state tuple."""
        ns = NodeStates.from_prefixes([1, 2, 3], metadata="m")
        self.assertEqual(ns.states, (1, 2, 3))
        self.assertEqual(ns.metadata, "m")


class TestAliasingChecks(unittest.TestCase):
    """``check_consistent_aliasing`` accepts consistent and rejects conflicting refs."""

    def test_consistent_shared_node_ok(self):
        """A consistently-prefixed shared node does not raise."""
        shared = _Linear(2, 3)
        graph_to_tree([shared, shared], prefix=0)  # should not raise

    def test_inconsistent_shared_node_raises(self):
        """A shared node with conflicting prefixes raises ``ValueError``."""
        shared = _Linear(2, 3)
        with self.assertRaises(ValueError):
            graph_to_tree([shared, shared], prefix=[0, 1])

    def test_check_aliasing_can_be_disabled(self):
        """``check_aliasing=False`` skips the consistency check."""
        shared = _Linear(2, 3)
        tree, _ = graph_to_tree([shared, shared], prefix=[0, 1], check_aliasing=False)
        self.assertIsNotNone(tree)

    def test_inconsistent_stateless_shared_node_raises(self):
        """A *state-free* shared node with conflicting prefixes still raises.

        Regression (L7): ``check_consistent_aliasing`` only iterated
        ``iter_leaf`` (which yields leaf States, never graph nodes), so the
        graph-node aliasing branch was dead. A stateless shared node therefore
        recorded no prefixes and inconsistent aliasing went undetected. The
        added ``iter_node`` pass must catch it.
        """

        class Empty(brainstate.graph.Node):
            """A graph node carrying no ``State`` leaves at all."""

            def __init__(self):
                self.name = 'x'

        shared = Empty()
        with self.assertRaises(ValueError):
            graph_to_tree([shared, shared], prefix=[0, 1])

    def test_consistent_stateless_shared_node_ok(self):
        """A consistently-prefixed *state-free* shared node does not raise."""

        class Empty(brainstate.graph.Node):
            def __init__(self):
                self.name = 'x'

        shared = Empty()
        graph_to_tree([shared, shared], prefix=0)  # should not raise

    def test_nested_graph_node_walked_and_recorded(self):
        """The ``iter_node`` pass descends into *nested* graph nodes too.

        ``check_consistent_aliasing`` walks every graph node reachable from a
        leaf (the leaf node and each nested graph-node child). A single node that
        holds an inner graph node must split without raising; the walk visits both
        the outer and the inner node.
        """

        class Inner(brainstate.graph.Node):
            def __init__(self):
                self.w = brainstate.ParamState(jnp.ones((2,)))

        class Outer(brainstate.graph.Node):
            def __init__(self, inner):
                self.inner = inner  # a nested graph node

        tree, _ = graph_to_tree(Outer(Inner()), prefix=0)
        self.assertIsNotNone(tree)

    def test_shared_nested_graph_node_inconsistent_prefix_raises(self):
        """A *nested* graph node shared across slots with conflicting prefixes
        is detected by the per-node aliasing walk and raises ``ValueError``.

        The inner node is reachable only by descending into each outer node, so
        catching the conflict proves the walk records prefixes for nested nodes,
        not just the top-level leaf node.
        """

        class Inner(brainstate.graph.Node):
            def __init__(self):
                self.w = brainstate.ParamState(jnp.ones((2,)))

        class Outer(brainstate.graph.Node):
            def __init__(self, inner):
                self.inner = inner

        inner = Inner()
        with self.assertRaises(ValueError):
            graph_to_tree([Outer(inner), Outer(inner)], prefix=[0, 1])


class TestConvertHelpers(unittest.TestCase):
    """Internal helpers used by the conversion routines."""

    def test_broadcast_prefix_replicates_over_leaves(self):
        """A scalar prefix is broadcast to one entry per full-tree leaf."""
        out = _convert.broadcast_prefix(7, {"a": 1, "b": 2, "c": 3})
        self.assertEqual(out, [7, 7, 7])

    def test_get_rand_state_returns_random_state_class(self):
        """``_get_rand_state`` lazily imports and caches ``RandomState``."""
        cls = _convert._get_rand_state()
        from brainstate.random import RandomState
        self.assertIs(cls, RandomState)
        # Second call returns the cached class.
        self.assertIs(_convert._get_rand_state(), RandomState)


class TestLeafPrefixParityValidation(unittest.TestCase):
    """The leaf/prefix length-parity guard must hold even under ``python -O``.

    The check was historically an ``assert``, which is stripped when Python runs
    optimized; a length mismatch would then silently truncate-``zip`` leaves and
    prefixes (a wrong result) instead of raising. These tests force a mismatch
    and require the *explicit* parity ``ValueError`` (identified by its message),
    so they fail if the guard reverts to a stripped ``assert``.
    """

    def test_graph_to_tree_mismatched_prefixes_raise(self):
        node = _Linear(2, 3)
        original = _convert.broadcast_prefix
        # Force a prefix list one element longer than the flattened leaves so the
        # explicit guard (not a downstream pytree error) is the one that fires.
        _convert.broadcast_prefix = lambda *a, **k: original(*a, **k) + [None]
        try:
            with self.assertRaisesRegex(ValueError, 'Mismatched number of leaves'):
                graph_to_tree(node)
        finally:
            _convert.broadcast_prefix = original

    def test_tree_to_graph_mismatched_prefixes_raise(self):
        node = _Linear(2, 3)
        tree, _ = graph_to_tree(node)
        original = _convert.broadcast_prefix
        _convert.broadcast_prefix = lambda *a, **k: original(*a, **k) + [None]
        try:
            with self.assertRaisesRegex(ValueError, 'Mismatched number of leaves'):
                tree_to_graph(tree)
        finally:
            _convert.broadcast_prefix = original


class TestGraphToTreeStates(unittest.TestCase):
    def test_find_states_and_sharing(self):
        import jax.numpy as jnp
        import brainstate as bs
        from brainstate import graph

        class Box(graph.Node):
            def __init__(self, **kw):
                for k, v in kw.items():
                    setattr(self, k, v)

        shared = bs.ParamState(jnp.ones((3, 3)))
        a, b = Box(w=shared), Box(w=shared)
        from brainstate.util import FlattedDict
        tree, find = graph.graph_to_tree([a, b])
        self.assertIsInstance(find, FlattedDict)
        # shared State surfaces in find_states (deduped)
        self.assertTrue(any(v is shared for v in find.values()))
        # round-trip restores sharing
        back = graph.tree_to_graph(tree)
        self.assertIs(back[0].w, back[1].w)


if __name__ == "__main__":
    unittest.main()
