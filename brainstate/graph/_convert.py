# The file is adapted from the Flax library (https://github.com/google/flax).
# The credit should go to the Flax authors.
#
# Copyright 2024 The Flax Authors
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

from collections.abc import Callable, Iterable
from typing import Any, TypeVar

import jax

from brainstate._state import State
from brainstate.typing import Missing, PyTree, SeedOrKey, PathParts
from brainstate.util import PyTreeNode, field
from ._context import SplitContext, MergeContext, split_context, merge_context
from ._node import Node as GraphNode
from ._operation import (
    RefMap,
    iter_leaf as iter_graph,
    _is_graph_node,
    GraphDef,
    GraphStateMapping,
    states,
)

__all__ = [
    'graph_to_tree', 'tree_to_graph', 'NodeStates'
]

Node = TypeVar('Node')
Leaf = TypeVar('Leaf')
KeyEntry = TypeVar('KeyEntry')
KeyPath = tuple[KeyEntry, ...]
Prefix = Any
RandomState = None


def _get_rand_state() -> type:
    global RandomState
    if RandomState is None:
        from ..random import RandomState as RS
        RandomState = RS
    return RandomState


def check_consistent_aliasing(
    node: tuple[Any, ...],
    prefix: tuple[Any, ...],
    /,
    *,
    node_prefixes: RefMap[Any, list[tuple[PathParts, Any]]] | None = None,
) -> None:
    """Check that shared nodes have consistent prefixes across all paths."""
    node_prefixes = RefMap() if node_prefixes is None else node_prefixes

    for path, value in iter_graph(node):
        if _is_graph_node(value) or isinstance(value, State):
            if isinstance(value, GraphNode):
                value.check_valid_context(
                    lambda: f'Trying to extract graph node from different trace level, got {value!r}'
                )
            if isinstance(value, State):
                value.check_valid_trace(
                    lambda: f'Trying to extract graph node from different trace level, got {value!r}'
                )
            if value in node_prefixes:
                node_prefixes[value].append((path, prefix))
            else:
                node_prefixes[value] = [(path, prefix)]

    node_msgs = []
    for node, paths_prefixes in node_prefixes.items():
        unique_prefixes = {prefix for _, prefix in paths_prefixes}
        if len(unique_prefixes) > 1:
            path_prefix_repr = '\n'.join([
                f'  {"/".join(map(str, path)) if path else "<root>"}: {prefix}'
                for path, prefix in paths_prefixes
            ])
            node_msgs.append(f'Node: {type(node)}\n{path_prefix_repr}')

    if node_msgs:
        raise ValueError(
            'Inconsistent aliasing detected. The following nodes have different prefixes:\n'
            + '\n'.join(node_msgs)
        )


def broadcast_prefix(
    prefix_tree: Any,
    full_tree: Any,
    prefix_is_leaf: Callable[[Any], bool] | None = None,
    tree_is_leaf: Callable[[Any], bool] | None = None,
) -> list[Any]:
    """Broadcast a prefix tree to match the leaves of a full tree."""
    result = []
    num_leaves = lambda t: jax.tree_util.tree_structure(t, is_leaf=tree_is_leaf).num_leaves
    add_leaves = lambda x, subtree: result.extend([x] * num_leaves(subtree))
    jax.tree.map(add_leaves, prefix_tree, full_tree, is_leaf=prefix_is_leaf)
    return result


class NodeStates(PyTreeNode):
    """A JAX pytree wrapper that carries both a GraphDef and one or more state mappings.

    Used by ``graph_to_tree`` / ``tree_to_graph`` to represent graph nodes as
    pure pytrees so that JAX transforms (vmap, jit, etc.) can operate on them.
    """

    _graphdef: GraphDef[Any] | None
    states: tuple[GraphStateMapping, ...]
    metadata: Any = field(pytree_node=False)

    @property
    def graphdef(self) -> GraphDef[Any]:
        if self._graphdef is None:
            raise ValueError('No graphdef available')
        return self._graphdef

    @property
    def state(self) -> GraphStateMapping:
        if len(self.states) != 1:
            raise ValueError(f'Expected exactly one GraphDefState, got {len(self.states)}')
        return self.states[0]

    @classmethod
    def from_split(
        cls,
        graphdef: GraphDef[Any],
        state: GraphStateMapping,
        /,
        *states: GraphStateMapping,
        metadata: Any = None,
    ) -> NodeStates:
        return cls(_graphdef=graphdef, states=(state, *states), metadata=metadata)

    @classmethod
    def from_states(cls, state: GraphStateMapping, *states: GraphStateMapping) -> NodeStates:
        return cls(_graphdef=None, states=(state, *states), metadata=None)

    @classmethod
    def from_prefixes(cls, prefixes: Iterable[Any], /, *, metadata: Any = None) -> NodeStates:
        return cls(_graphdef=None, states=tuple(prefixes), metadata=metadata)


def _default_split_fn(ctx: SplitContext, path: KeyPath, prefix: Prefix, leaf: Leaf) -> NodeStates:
    return NodeStates.from_split(*ctx.treefy_split(leaf))


def graph_to_tree(
    may_have_graph_nodes,
    /,
    *,
    prefix: Any = Missing,
    split_fn: Callable[[SplitContext, KeyPath, Prefix, Leaf], Any] = _default_split_fn,
    map_non_graph_nodes: bool = False,
    check_aliasing: bool = True,
) -> tuple[PyTree, dict[KeyPath, SeedOrKey]]:
    """Convert a pytree that may contain graph nodes into a pure pytree of NodeStates."""
    leaf_prefixes = broadcast_prefix(prefix, may_have_graph_nodes, prefix_is_leaf=lambda x: x is None)
    leaf_keys, treedef = jax.tree_util.tree_flatten_with_path(may_have_graph_nodes)

    assert len(leaf_keys) == len(leaf_prefixes)

    with split_context() as (ctx, index_ref):
        leaves_out = []
        node_prefixes: RefMap[Any, list[tuple[PathParts, Any]]] = RefMap()
        for (keypath, leaf), leaf_prefix in zip(leaf_keys, leaf_prefixes):
            if _is_graph_node(leaf):
                if check_aliasing:
                    check_consistent_aliasing(leaf, leaf_prefix, node_prefixes=node_prefixes)
                leaves_out.append(split_fn(ctx, keypath, leaf_prefix, leaf))
            else:
                if map_non_graph_nodes:
                    leaf = split_fn(ctx, keypath, leaf_prefix, leaf)
                leaves_out.append(leaf)

    # Build a dict mirroring RefMap's content via the public API, then extract
    # State objects from it.  We must not access the private ._mapping attribute.
    public_map = {id(k): (k, v) for k, v in index_ref.items()}
    find_states = states(public_map)
    pytree_out = jax.tree.unflatten(treedef, leaves_out)
    return pytree_out, find_states


def _is_tree_node(x: Any) -> bool:
    return isinstance(x, NodeStates)


def _merge_tree_node(ctx: MergeContext, path: KeyPath, prefix: Prefix, leaf: Leaf) -> Any:
    if not isinstance(leaf, NodeStates):
        raise ValueError(f'Expected TreeNode, got {type(leaf)} at path {path}')
    return ctx.treefy_merge(leaf.graphdef, *leaf.states)


def tree_to_graph(
    tree: Any,
    /,
    *,
    prefix: Any = Missing,
    merge_fn: Callable[[MergeContext, KeyPath, Prefix, Leaf], Any] = _merge_tree_node,
    is_node_leaf: Callable[[Leaf], bool] = _is_tree_node,
    is_leaf: Callable[[Leaf], bool] = _is_tree_node,
    map_non_graph_nodes: bool = False,
) -> Any:
    """Convert a pytree of NodeStates back into graph nodes."""
    _prefix_is_leaf = lambda x: x is None or is_leaf(x)
    leaf_prefixes = broadcast_prefix(prefix, tree, prefix_is_leaf=_prefix_is_leaf, tree_is_leaf=is_leaf)
    leaf_keys, treedef = jax.tree_util.tree_flatten_with_path(tree, is_leaf=is_leaf)
    assert len(leaf_keys) == len(leaf_prefixes), "Mismatched number of keys and prefixes"

    with merge_context() as (ctx, index_ref):
        leaves_out = []
        for (keypath, leaf), leaf_prefix in zip(leaf_keys, leaf_prefixes):
            if is_node_leaf(leaf):
                leaves_out.append(merge_fn(ctx, keypath, leaf_prefix, leaf))
            else:
                if map_non_graph_nodes:
                    leaf = merge_fn(ctx, keypath, leaf_prefix, leaf)
                leaves_out.append(leaf)

    return jax.tree.unflatten(treedef, leaves_out)
