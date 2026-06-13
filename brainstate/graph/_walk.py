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

"""The traversal kernel and node-type registry.

This module owns the vocabulary the rest of the engine builds on:

* the **node-impl protocol** (:class:`GraphNodeImpl` / :class:`PyTreeNodeImpl`)
  and the registry behind :func:`register_graph_node_type`;
* the ``_is_*`` predicates that classify a value as a graph node, a pytree, a
  ``State``, or a ``TreefyState`` leaf;
* :class:`Static` (a pytree-static wrapper) and the one-level pytree
  flatten/unflatten helpers;
* the single depth-first traversal kernel that backs :func:`iter_leaf` and
  :func:`iter_node` (and, via ``_operations``, ``states`` / ``nodes``).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterator, Sequence
from typing import Any, Generic, TypeVar

import jax
import numpy as np
from typing_extensions import TypeGuard

from brainstate._state import State, TreefyState
from brainstate._utils import set_module_as
from brainstate.typing import Key, PathParts
from ._graphdef import PytreeType

__all__ = ['register_graph_node_type', 'Static', 'iter_leaf', 'iter_node']

MAX_INT = np.iinfo(np.int32).max

N = TypeVar('N')
Leaf = TypeVar('Leaf')
AuxData = TypeVar('AuxData')
A = TypeVar('A')


# ---------------------------------------------------------------------------
# Predicates
# ---------------------------------------------------------------------------

def _is_state_leaf(x: Any) -> TypeGuard[TreefyState]:
    """Return whether ``x`` is a bare ``TreefyState`` leaf."""
    return isinstance(x, TreefyState)


def _is_node_leaf(x: Any) -> TypeGuard[State]:
    """Return whether ``x`` is a ``State`` leaf."""
    return isinstance(x, State)


def _is_pytree_node(x: Any) -> bool:
    """Return whether ``x`` is a (non-leaf) JAX pytree container."""
    return classify(x) == PYTREE


def _is_graph_node(x: Any) -> bool:
    """Return whether ``type(x)`` is a registered mutable graph-node type."""
    return classify(x) == GRAPH_NODE


def _is_node(x: Any) -> bool:
    """Return whether ``x`` is a container the engine descends into."""
    k = classify(x)
    return k == GRAPH_NODE or k == PYTREE


def _is_node_type(x: type) -> bool:
    """Return whether ``x`` is a registered node type or :class:`PytreeType`."""
    return x in _node_impl_for_type or x is PytreeType


# ---------------------------------------------------------------------------
# Node-impl protocol + registry
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class NodeImplBase(Generic[N, Leaf, AuxData]):
    """Base descriptor for a node type: its type and one-level flatten function."""
    type: type
    flatten: Callable[[N], tuple[Sequence[tuple[Key, Leaf]], AuxData]]

    def node_dict(self, node: N) -> dict[Key, Leaf]:
        """Return the node's one-level children as an ordered ``{key: value}`` dict."""
        nodes, _ = self.flatten(node)
        nodes = tuple(nodes)
        result = dict(nodes)
        if len(result) != len(nodes):
            raise ValueError(
                f"Duplicate child keys returned by the flatten function for "
                f"{type(node).__name__}; each child key must be unique."
            )
        return result


@dataclasses.dataclass(frozen=True)
class GraphNodeImpl(NodeImplBase[N, Leaf, AuxData]):
    """Node implementation for registered mutable graph nodes."""
    set_key: Callable[[N, Key, Leaf], None]
    pop_key: Callable[[N, Key], Leaf]
    create_empty: Callable[[AuxData], N]
    clear: Callable[[N], None]

    def init(self, node: N, items: tuple) -> None:
        """Populate an empty ``node`` from ``(key, value)`` items via ``set_key``."""
        for key, value in items:
            self.set_key(node, key, value)


@dataclasses.dataclass(frozen=True)
class PyTreeNodeImpl(NodeImplBase[N, Leaf, AuxData]):
    """Node implementation for JAX pytree containers (immutable structure)."""
    unflatten: Callable[[tuple, AuxData], N]


NodeImpl = GraphNodeImpl | PyTreeNodeImpl
_node_impl_for_type: dict[type, NodeImpl] = {}


# ---------------------------------------------------------------------------
# Value classification (type-keyed, cached)
# ---------------------------------------------------------------------------
# Kinds, ordered to match the historical encoder/kernel dispatch exactly:
# a value is a graph node, else a pytree container, else a State, else a bare
# TreefyState leaf, else an inline static. The order matters: TreefyState IS a
# registered jax pytree, so it must resolve to PYTREE (not STATE_LEAF) to keep
# the legacy behavior of encoding a bare TreefyState attribute as a PytreeEdge.
GRAPH_NODE, PYTREE, STATE, STATE_LEAF, STATIC = range(5)

_KIND_CACHE: dict[type, int] = {}


def _compute_kind(t: type, x: Any) -> int:
    if t in _node_impl_for_type:
        return GRAPH_NODE
    if not jax.tree_util.all_leaves((x,)):  # x (the instance) is needed only for the pytree check; all other branches are type-only
        return PYTREE
    if issubclass(t, State):
        return STATE
    if issubclass(t, TreefyState):
        return STATE_LEAF
    return STATIC


def classify(x: Any) -> int:
    """Classify ``x`` into one of GRAPH_NODE/PYTREE/STATE/STATE_LEAF/STATIC.

    The result depends only on ``type(x)`` for the registered/container types
    the engine encounters, so it is memoized per type. The cache is cleared by
    :func:`register_graph_node_type`.
    """
    t = type(x)
    k = _KIND_CACHE.get(t)
    if k is None:
        k = _compute_kind(t, x)
        _KIND_CACHE[t] = k
    return k


def _clear_classification_cache() -> None:
    """Drop all memoized classifications (internal; tests / dynamic registration)."""
    _KIND_CACHE.clear()


@set_module_as('brainstate.graph')
def register_graph_node_type(
    type: type,
    flatten: Callable[[N], tuple[Sequence[tuple[Key, Leaf]], AuxData]],
    set_key: Callable[[N, Key, Leaf], None],
    pop_key: Callable[[N, Key], Leaf],
    create_empty: Callable[[AuxData], N],
    clear: Callable[[N], None],
) -> None:
    """Register a custom mutable graph-node type with the graph engine.

    Parameters
    ----------
    type : type
        The Python type to register.
    flatten : callable
        ``node -> (items, metadata)`` where ``items`` is an ordered sequence of
        ``(key, value)`` pairs and ``metadata`` is hashable auxiliary data.
    set_key : callable
        ``(node, key, value) -> None`` setting one child in place.
    pop_key : callable
        ``(node, key) -> value`` removing and returning one child.
    create_empty : callable
        ``metadata -> node`` building an empty shell of the type.
    clear : callable
        ``node -> None`` removing all children in place.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> class MyList:
        ...     def __init__(self):
        ...         self.items = {}
        >>> brainstate.graph.register_graph_node_type(
        ...     MyList,
        ...     flatten=lambda n: (sorted(n.items.items()), (MyList,)),
        ...     set_key=lambda n, k, v: n.items.__setitem__(k, v),
        ...     pop_key=lambda n, k: n.items.pop(k),
        ...     create_empty=lambda meta: MyList(),
        ...     clear=lambda n: n.items.clear(),
        ... )
    """
    _node_impl_for_type[type] = GraphNodeImpl(
        type=type, flatten=flatten, set_key=set_key,
        pop_key=pop_key, create_empty=create_empty, clear=clear,
    )
    _KIND_CACHE.clear()


def _get_node_impl(x: Any) -> NodeImpl:
    """Return the node-impl for value ``x`` (graph node or pytree)."""
    if isinstance(x, State):
        raise ValueError(f'State is not a node: {x}')
    node_type = type(x)
    if node_type not in _node_impl_for_type:
        if _is_pytree_node(x):
            return PYTREE_NODE_IMPL
        raise ValueError(f'Unknown node type: {x}')
    return _node_impl_for_type[node_type]


def get_node_impl_for_type(x: type) -> NodeImpl:
    """Return the node-impl registered for type ``x`` (or the pytree impl)."""
    if x is PytreeType:
        return PYTREE_NODE_IMPL
    try:
        return _node_impl_for_type[x]
    except KeyError:
        raise ValueError(
            f"Unknown graph node type: {x!r}. Register it with "
            f"brainstate.graph.register_graph_node_type before flatten/unflatten."
        ) from None


# ---------------------------------------------------------------------------
# Static wrapper + pytree node-impl
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Static(Generic[A]):
    """An empty pytree node that treats its inner value as static.

    Wrap a value in ``Static`` to carry it through graph flattening as static
    metadata (no leaves). ``value`` must define ``__eq__`` and ``__hash__``.

    Parameters
    ----------
    value : Any
        The static payload.
    """
    value: A


jax.tree_util.register_static(Static)


def _key_path_to_key(key: Any) -> Key:
    """Convert a JAX key-path entry into a plain mapping key."""
    if isinstance(key, jax.tree_util.SequenceKey):
        return key.idx
    elif isinstance(key, (jax.tree_util.DictKey, jax.tree_util.FlattenedIndexKey)):
        if not isinstance(key.key, Key):
            raise ValueError(
                f'Invalid key: {key.key!r}. A pytree key must be hashable '
                f'(its type does not satisfy the Key protocol).'
            )
        return key.key
    elif isinstance(key, jax.tree_util.GetAttrKey):
        return key.name
    return str(key)


def _flatten_pytree(pytree: Any):
    """Flatten one level of a pytree into ``((key, value)...)`` + treedef."""
    leaves, treedef = jax.tree_util.tree_flatten_with_path(
        pytree, is_leaf=lambda x: x is not pytree
    )
    nodes = tuple((_key_path_to_key(path[0]), value) for path, value in leaves)
    return nodes, treedef


def _unflatten_pytree(nodes: tuple, treedef) -> Any:
    """Rebuild a pytree container from ``(key, value)`` items + treedef."""
    return treedef.unflatten(value for _, value in nodes)


PYTREE_NODE_IMPL = PyTreeNodeImpl(
    type=PytreeType, flatten=_flatten_pytree, unflatten=_unflatten_pytree
)


# ---------------------------------------------------------------------------
# The single traversal kernel
# ---------------------------------------------------------------------------

def _iter_graph(
    node: Any, *, allowed_hierarchy: tuple[int, int], want: str,
    dedup_leaves: bool = True,
) -> Iterator[tuple[PathParts, Any]]:
    """Shared depth-first traversal backing the iteration family.

    Visits each container (graph node and pytree) once by identity. The depth
    ``level`` increments only when descending into a graph node, so pytree
    nesting does not consume hierarchy depth. ``want`` selects what is yielded:
    ``'leaf'`` yields non-container values (post-children), ``'node'`` yields
    graph nodes (post-children).
    """
    lo, hi = allowed_hierarchy

    def _iter(node_, visited, path_, level_):
        if level_ > hi:
            return
        kind = classify(node_)
        if kind == GRAPH_NODE or kind == PYTREE:
            if id(node_) in visited:
                return
            visited.add(id(node_))
            if kind == GRAPH_NODE:
                impl = _node_impl_for_type[type(node_)]
            else:
                impl = PYTREE_NODE_IMPL
            items, _ = impl.flatten(node_)
            for key, value in items:
                child_is_node = classify(value) == GRAPH_NODE
                yield from _iter(
                    value, visited, (*path_, key),
                    level_ + 1 if child_is_node else level_,
                )
            if want == 'node' and kind == GRAPH_NODE and level_ >= lo:
                yield path_, node_
        else:
            if want == 'leaf' and level_ >= lo:
                # State leaves dedup by identity (first pre-order path wins).
                # Reusing `visited` is safe: containers and State leaves are
                # disjoint object sets, so their ids never collide.
                if dedup_leaves and (kind == STATE or kind == STATE_LEAF):
                    if id(node_) in visited:
                        return
                    visited.add(id(node_))
                yield path_, node_

    yield from _iter(node, set(), (), 0)


@set_module_as('brainstate.graph')
def iter_leaf(
    node: Any, allowed_hierarchy: tuple[int, int] = (0, MAX_INT)
) -> Iterator[tuple[PathParts, Any]]:
    """Iterate ``(path, value)`` over every leaf in the graph.

    Repeated containers are visited only once (by identity). State leaves are
    likewise deduplicated by identity: if the same State object is reachable
    via multiple paths, only its first pre-order occurrence is yielded.

    Parameters
    ----------
    node : Any
        The root graph node or pytree.
    allowed_hierarchy : tuple of (int, int), optional
        ``(lo, hi)`` graph-node depth bounds; leaves outside the range are
        skipped. Depth increments only when descending into a graph node.

    Yields
    ------
    tuple of (PathParts, Any)
        The path to, and value of, each leaf.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> model = brainstate.nn.Linear(2, 3)
        >>> leaves = list(brainstate.graph.iter_leaf(model))
        >>> all(len(p) >= 1 for p, _ in leaves)
        True
    """
    yield from _iter_graph(node, allowed_hierarchy=allowed_hierarchy, want='leaf')


@set_module_as('brainstate.graph')
def iter_node(
    node: Any, allowed_hierarchy: tuple[int, int] = (0, MAX_INT)
) -> Iterator[tuple[PathParts, Any]]:
    """Iterate ``(path, graph_node)`` over every graph node in the graph.

    Repeated nodes are visited only once (by identity).

    Parameters
    ----------
    node : Any
        The root graph node or pytree.
    allowed_hierarchy : tuple of (int, int), optional
        ``(lo, hi)`` graph-node depth bounds.

    Yields
    ------
    tuple of (PathParts, Any)
        The path to, and the object of, each graph node.
    """
    yield from _iter_graph(node, allowed_hierarchy=allowed_hierarchy, want='node')
