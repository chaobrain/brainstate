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

"""Property-based / fuzz tests for the flat-IR graph engine.

These tests build *randomized* graphs — with shared sub-nodes, shared states,
and reference cycles — and assert the structural invariants the engine
promises:

* **Determinism** — flattening the same graph twice yields the exact same
  :class:`GraphDef` and the same state values.
* **Value preservation** — every state value survives a flatten/unflatten
  round-trip unchanged.
* **Aliasing preservation** — objects that are shared (or cyclic) in the input
  remain shared in the output.
* **split/merge fidelity** — ``treefy_split`` followed by ``treefy_merge``
  reconstructs an equivalent graph for arbitrary filter partitions.

Round-tripped values are read back with the structural walker
(:func:`iter_leaf`) rather than by re-flattening. Re-flattening a *freshly
materialized* graph is unsupported by ``brainstate`` itself — ``State`` objects
rebuilt from a :class:`TreefyState` intentionally drop their trace metadata —
so the walker is the correct, engine-agnostic way to inspect a rebuilt graph.

Randomness uses :mod:`brainstate.random` with fixed seeds so every run is
fully deterministic and reproducible.
"""

import unittest

import jax
import jax.numpy as jnp

import brainstate
from brainstate.graph._flatten import flatten, unflatten
from brainstate.graph._operations import treefy_split, treefy_merge
from brainstate.graph._graphdef import GraphDef
from brainstate.graph._walk import iter_leaf

# Seeds driving the fuzz iterations. Fixed for determinism.
_SEEDS = (0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144)


# ---------------------------------------------------------------------------
# A registrable graph node backed by a plain dict.
# ---------------------------------------------------------------------------

class _Dict:
    """A minimal mutable graph node used to assemble random graphs."""

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


def _make_state(rng, kind):
    """Create a fresh state of the given kind carrying a random array."""
    value = rng.rand(2)
    if kind == 0:
        return brainstate.ParamState(value)
    return brainstate.ShortTermState(value)


def _random_graph(rng, max_depth=3):
    """Build a random graph and report the aliases it intentionally created.

    Returns
    -------
    root : _Dict
        The root graph node.
    aliases : list of tuple
        Each entry is ``(path_a, path_b)`` — two key-paths that must resolve to
        the *same* object (a shared node/state, or a cycle when ``path_a`` is an
        ancestor of ``path_b``).
    """
    created_nodes = []   # (path, node) for every _Dict made so far
    created_states = []  # (path, state) for every State made so far
    aliases = []

    def build(depth, path):
        node = _Dict()
        created_nodes.append((path, node))
        n_children = int(rng.randint(1, 4))
        for i in range(n_children):
            key = f'k{i}'
            cpath = path + (key,)
            choice = int(rng.randint(0, 6))
            if depth >= max_depth or choice == 0:
                state = _make_state(rng, int(rng.randint(0, 2)))
                node.data[key] = state
                created_states.append((cpath, state))
            elif choice == 1:
                node.data[key] = int(rng.randint(0, 100))           # static int
            elif choice == 2:
                node.data[key] = build(depth + 1, cpath)            # nested node
            elif choice == 3:
                # an inline pytree holding a fresh state and a static
                node.data[key] = [_make_state(rng, 0), int(rng.randint(0, 9))]
            elif choice == 4:
                # share an existing graph node (may form a cycle)
                opath, onode = created_nodes[int(rng.randint(0, len(created_nodes)))]
                node.data[key] = onode
                aliases.append((opath, cpath))
            elif choice == 5 and created_states:
                # share an existing state
                opath, ostate = created_states[int(rng.randint(0, len(created_states)))]
                node.data[key] = ostate
                aliases.append((opath, cpath))
            else:
                node.data[key] = int(rng.randint(0, 100))           # static fallback
        return node

    return build(0, ()), aliases


def _resolve(root, path):
    """Follow a key-path from ``root`` through nodes/pytrees to the leaf object."""
    cur = root
    for k in path:
        if isinstance(cur, _Dict):
            cur = cur.data[k]
        elif isinstance(cur, dict):
            cur = cur[k]
        elif isinstance(cur, (list, tuple)):
            cur = cur[k]
        else:
            raise KeyError(f'cannot descend into {type(cur).__name__} at {path!r}')
    return cur


def _value_leaves(state_value):
    return jax.tree_util.tree_leaves(state_value)


class _PropertyTestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _register_dict()

    def _rng(self, seed):
        return brainstate.random.RandomState(seed)

    def _walk_state_values(self, root):
        """Map every state path to its value by walking the live graph.

        Uses the structural walker so it works on *materialized* graphs too
        (it never calls ``State.to_state_ref``).
        """
        return {
            path: leaf.value
            for path, leaf in iter_leaf(root)
            if isinstance(leaf, brainstate.State)
        }

    def _assert_values_equal(self, before, after):
        self.assertEqual(set(before), set(after))
        for path in before:
            lb = _value_leaves(before[path])
            la = _value_leaves(after[path])
            self.assertEqual(len(lb), len(la))
            for x, y in zip(lb, la):
                self.assertTrue(jnp.allclose(x, y), msg=f'value mismatch at {path!r}')


class TestFlattenDeterminism(_PropertyTestBase):
    def test_flatten_is_deterministic(self):
        for seed in _SEEDS:
            with self.subTest(seed=seed):
                root, _ = _random_graph(self._rng(seed))
                gdef_a, states_a = flatten(root)
                gdef_b, states_b = flatten(root)

                self.assertIsInstance(gdef_a, GraphDef)
                # Same definition: indices, statics, types, ordering.
                self.assertEqual(gdef_a, gdef_b)
                self.assertEqual(hash(gdef_a), hash(gdef_b))
                # Same state mapping, value-for-value.
                self._assert_values_equal(
                    {p: ts.value for p, ts in states_a.to_flat().items()},
                    {p: ts.value for p, ts in states_b.to_flat().items()},
                )


class TestValuePreservation(_PropertyTestBase):
    def test_state_values_survive_round_trip(self):
        for seed in _SEEDS:
            with self.subTest(seed=seed):
                root, _ = _random_graph(self._rng(seed))
                before = self._walk_state_values(root)

                gdef, states = flatten(root)
                root2 = unflatten(gdef, states)

                after = self._walk_state_values(root2)
                self._assert_values_equal(before, after)


class TestAliasingPreservation(_PropertyTestBase):
    def test_sharing_and_cycles_survive_round_trip(self):
        for seed in _SEEDS:
            with self.subTest(seed=seed):
                root, aliases = _random_graph(self._rng(seed))
                # Sanity: the aliases really are aliases in the original graph.
                for pa, pb in aliases:
                    self.assertIs(_resolve(root, pa), _resolve(root, pb))

                gdef, states = flatten(root)
                root2 = unflatten(gdef, states)

                for pa, pb in aliases:
                    self.assertIs(
                        _resolve(root2, pa), _resolve(root2, pb),
                        msg=f'alias {pa!r} <-> {pb!r} not preserved (seed={seed})',
                    )


class TestSplitMergeFidelity(_PropertyTestBase):
    def _split_merge(self, root, *filters):
        gdef, *parts = treefy_split(root, *filters)
        return treefy_merge(gdef, *parts)

    def test_split_merge_reconstructs_graph(self):
        for seed in _SEEDS:
            with self.subTest(seed=seed):
                root, aliases = _random_graph(self._rng(seed))
                before = self._walk_state_values(root)

                # Single combined state group.
                merged_all = self._split_merge(root)
                self._assert_values_equal(before, self._walk_state_values(merged_all))

                # Partition by state type; ``...`` sweeps the remainder.
                merged_split = self._split_merge(root, brainstate.ParamState, ...)
                self._assert_values_equal(before, self._walk_state_values(merged_split))

                # Aliasing must survive the split/merge as well.
                for pa, pb in aliases:
                    self.assertIs(_resolve(merged_split, pa), _resolve(merged_split, pb))


if __name__ == '__main__':
    unittest.main()
