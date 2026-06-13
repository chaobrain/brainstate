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

"""Convert between graph nodes and pure pytrees of :class:`NodeStates`.

:func:`graph_to_tree` replaces graph nodes in a pytree with :class:`NodeStates`
wrappers (a ``GraphDef`` + state mappings) so JAX transforms can operate on
them as pure pytrees; :func:`tree_to_graph` rebuilds the graph nodes. Shared
references are kept consistent across the conversion via the split/merge
contexts.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, TypeVar

import jax

from brainstate._state import State
from brainstate.typing import Missing, PyTree, SeedOrKey, PathParts
from brainstate.util import PyTreeNode, field, NestedDict as GraphStateMapping
from ._context import SplitContext, MergeContext, split_context, merge_context
from ._graphdef import GraphDef
from ._node import Node as GraphNode
from ._operations import states
from ._reftrack import RefMap
from ._walk import iter_leaf as iter_graph, _is_graph_node

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
        if isinstance(value, State):
            value.check_valid_trace(
                lambda: f'Trying to extract graph node from different trace level, got {value!r}'
            )
            if value in node_prefixes:
                node_prefixes[value].append((path, prefix))
            else:
                node_prefixes[value] = [(path, prefix)]

    from ._walk import iter_node  # already exported from _walk
    for path, value in iter_node(node):
        if _is_graph_node(value):
            if isinstance(value, GraphNode):
                value.check_valid_context(
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
    """Convert a pytree that may contain graph nodes into a pure pytree.

    Every graph node embedded in the input is replaced (by ``split_fn``, which
    defaults to wrapping it in a :class:`NodeStates`) so the result is a plain
    pytree safe to pass through JAX transformations. Aliasing across the split
    nodes is collected so it can be restored later by :func:`tree_to_graph`.

    Parameters
    ----------
    may_have_graph_nodes : Any
        A pytree whose leaves may include graph nodes (``Node`` subclasses and
        registered graph-node types). Passed positionally.
    prefix : Any, optional
        A prefix pytree broadcast over ``may_have_graph_nodes`` and handed to
        ``split_fn`` per leaf. Defaults to ``Missing`` (no prefix).
    split_fn : callable, optional
        Called as ``split_fn(ctx, keypath, prefix, leaf)`` for each graph node
        (and, when ``map_non_graph_nodes`` is true, for every other leaf). The
        default wraps the node's split result in a :class:`NodeStates`.
    map_non_graph_nodes : bool, optional
        When ``True``, also run ``split_fn`` on non-graph-node leaves. Defaults
        to ``False``.
    check_aliasing : bool, optional
        When ``True`` (default), verify that shared/aliased graph nodes are
        consistent with ``prefix`` and raise if not.

    Returns
    -------
    pytree_out : PyTree
        The input pytree with graph nodes replaced by ``split_fn`` outputs.
    find_states : dict
        A mapping from key-path to the :class:`State` objects discovered while
        splitting, used to relink shared state on the way back.

    See Also
    --------
    tree_to_graph : The inverse conversion.
    """
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
    """Convert a pytree of :class:`NodeStates` back into graph nodes.

    The inverse of :func:`graph_to_tree`: each leaf recognised as a packed node
    (by ``is_node_leaf``) is merged back into a live graph node via ``merge_fn``,
    restoring the original sharing within a single :func:`merge_context`.

    Parameters
    ----------
    tree : Any
        A pytree whose node leaves are :class:`NodeStates` (or whatever
        ``is_node_leaf`` recognises). Passed positionally.
    prefix : Any, optional
        A prefix pytree broadcast over ``tree`` and handed to ``merge_fn`` per
        leaf. Defaults to ``Missing`` (no prefix).
    merge_fn : callable, optional
        Called as ``merge_fn(ctx, keypath, prefix, leaf)`` for each node leaf.
        The default merges ``leaf.graphdef`` with ``leaf.states``.
    is_node_leaf : callable, optional
        Predicate selecting which leaves are merged back into graph nodes.
        Defaults to "is a :class:`NodeStates`".
    is_leaf : callable, optional
        Predicate marking where ``jax.tree`` flattening stops. Defaults to
        "is a :class:`NodeStates`".
    map_non_graph_nodes : bool, optional
        When ``True``, also run ``merge_fn`` on non-node leaves. Defaults to
        ``False``.

    Returns
    -------
    Any
        The reconstructed pytree with graph nodes restored in place.

    See Also
    --------
    graph_to_tree : The inverse conversion.
    """
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
