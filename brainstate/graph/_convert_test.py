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


if __name__ == "__main__":
    unittest.main()
