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

"""Encode (:func:`flatten`) and decode (:func:`unflatten`) routines.

:func:`flatten` walks an object graph once in pre-order, hoisting every graph
node and ``State`` into a single global index space and embedding pytrees
inline, producing a :class:`~brainstate.graph.GraphDef` plus a flat state
mapping. :func:`unflatten` rebuilds the graph in three linear passes
(materialize states, create node shells, fill), so reconstruction over graph
nodes never recurses and cycles/sharing resolve regardless of fill order.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

from brainstate._state import State, TreefyState
from brainstate._utils import set_module_as
from brainstate.typing import PathParts
from brainstate.util import NestedDict
from ._graphdef import (
    GraphDef, NodeSpec, NodeEdge, StateEdge, StateLeafEdge, PytreeEdge, StaticEdge, Edge,
)
from ._reftrack import RefMap
from ._walk import (
    _is_node, _is_graph_node, _is_state_leaf, _get_node_impl,
    get_node_impl_for_type, PYTREE_NODE_IMPL,
)

__all__ = ['flatten', 'unflatten']

_MISSING = object()


def _format_path(path: PathParts) -> str:
    """Render a path tuple as a dotted human-readable string."""
    if not path:
        return '<root>'
    return '.'.join(str(p) for p in path)


# ---------------------------------------------------------------------------
# Encode
# ---------------------------------------------------------------------------

class _Encoder:
    """Single-pass pre-order encoder producing NodeSpecs + a flat state map."""

    def __init__(self, ref_index: RefMap, treefy_state: bool):
        self.ref_index = ref_index
        self.treefy_state = treefy_state
        self.specs: list[NodeSpec] = []
        self.mapping: dict[PathParts, Any] = {}

    def edge(self, path: PathParts, value: Any) -> Edge:
        if _is_node(value):
            if _is_graph_node(value):
                if value in self.ref_index:
                    return NodeEdge(self.ref_index[value])
                index = len(self.ref_index)
                self.ref_index[value] = index
                impl = _get_node_impl(value)
                items, metadata = impl.flatten(value)
                fields = tuple((k, self.edge((*path, k), v)) for k, v in items)
                self.specs.append(NodeSpec(impl.type, index, metadata, fields))
                return NodeEdge(index)
            impl = _get_node_impl(value)               # pytree (re-expanded)
            items, metadata = impl.flatten(value)
            fields = tuple((k, self.edge((*path, k), v)) for k, v in items)
            return PytreeEdge(metadata, fields)

        if isinstance(value, State):
            if value in self.ref_index:
                return StateEdge(self.ref_index[value], None, type(value))
            index = len(self.ref_index)
            self.ref_index[value] = index
            self.mapping[path] = value.to_state_ref() if self.treefy_state else value
            return StateEdge(index, path, type(value))

        if _is_state_leaf(value):                      # bare TreefyState
            self.mapping[path] = value
            return StateLeafEdge(path)

        # Any other value is an inline static field. It is kept as-is (matching
        # the legacy engine); hashability is enforced lazily by ``GraphDef`` so
        # that graphs with unhashable static attributes still split/merge.
        return StaticEdge(value)


@set_module_as('brainstate.graph')
def flatten(
    node: Any,
    /,
    ref_index: Optional[RefMap] = None,
    treefy_state: bool = True,
) -> Tuple[GraphDef, NestedDict]:
    """Flatten a graph ``node`` into a ``(GraphDef, NestedDict)`` pair.

    Graph nodes and ``State`` objects are hoisted into a single global index
    space (deduplicated by identity, so shared references and cycles collapse
    to one index); pytree containers are embedded inline and re-expanded per
    occurrence.

    Parameters
    ----------
    node : Any
        The root graph node or pytree to flatten.
    ref_index : RefMap, optional
        An identity map ``object -> index`` to accumulate into. When supplied
        (e.g. by ``split_context``), indices are global across multiple calls
        so cross-object references share indices. A fresh map is used if
        ``None``.
    treefy_state : bool, default True
        If ``True`` the state mapping holds ``TreefyState`` values; if
        ``False`` it holds the raw ``State`` objects.

    Returns
    -------
    tuple of (GraphDef, NestedDict)
        The static structure and the dynamic state mapping.

    Raises
    ------
    TypeError
        If ``ref_index`` is not a :class:`~brainstate.graph.RefMap`, or a
        static attribute value is not hashable.
    ValueError
        If ``node`` is not a graph node or pytree (e.g. a bare ``State``).

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> model = brainstate.nn.Linear(2, 3)
        >>> graphdef, states = brainstate.graph.flatten(model)
        >>> isinstance(states, brainstate.util.NestedDict)
        True
    """
    if ref_index is None:
        ref_index = RefMap()
    elif not isinstance(ref_index, RefMap):
        raise TypeError(f"ref_index must be a RefMap, got: {type(ref_index)}")
    if not _is_node(node):
        raise ValueError(
            f"Cannot flatten a non-node root of type {type(node).__name__}: {node!r}"
        )
    enc = _Encoder(ref_index, treefy_state)
    root = enc.edge((), node)
    node_specs = tuple(sorted(enc.specs, key=lambda s: s.index))
    return GraphDef(root, node_specs), NestedDict.from_flat(enc.mapping)


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------

def _materialize_state(value, index, index_ref_cache):
    """Build or refresh a ``State`` for a defining ``StateEdge``."""
    if index_ref_cache is not None and index in index_ref_cache:
        variable = index_ref_cache[index]
        if not isinstance(variable, State):
            raise ValueError(f"Cached object at index {index} is not a State.")
        if isinstance(value, TreefyState):
            variable.update_from_ref(value)
        elif isinstance(value, State):
            if value._been_writen:
                variable.value = value.value
            else:
                variable.restore_value(value.value)
        else:
            raise ValueError(f"Expected a State/TreefyState, got {type(value)}.")
        return variable
    if isinstance(value, TreefyState):
        return value.to_state()
    if isinstance(value, State):
        return value
    raise ValueError(f"Expected a State/TreefyState, got {type(value)}.")


def _iter_edges(graphdef: GraphDef):
    """Yield every edge in the IR (root + all fields, descending PytreeEdges)."""
    stack = [graphdef.root]
    for spec in graphdef.node_specs:
        stack.extend(e for _, e in spec.fields)
    while stack:
        e = stack.pop()
        yield e
        if isinstance(e, PytreeEdge):
            stack.extend(c for _, c in e.fields)


@set_module_as('brainstate.graph')
def unflatten(
    graphdef: GraphDef,
    state_mapping: NestedDict,
    /,
    *,
    index_ref: Optional[dict] = None,
    index_ref_cache: Optional[dict] = None,
) -> Any:
    """Reconstruct a graph node from a ``GraphDef`` and a state mapping.

    Parameters
    ----------
    graphdef : GraphDef
        The static structure produced by :func:`flatten`.
    state_mapping : NestedDict
        The dynamic state mapping (or a merge of filtered mappings).
    index_ref : dict, optional
        A ``global index -> rebuilt object`` table to accumulate into. Shared
        across a ``merge_context`` so references resolve across calls.
    index_ref_cache : dict, optional
        A cache of pre-existing objects keyed by global index; when supplied,
        matching nodes/states are reused and updated in place (the "update"
        path) instead of created fresh.

    Returns
    -------
    Any
        The reconstructed root object.

    Raises
    ------
    TypeError
        If ``graphdef`` is not a :class:`~brainstate.graph.GraphDef`.
    ValueError
        If a referenced state path is missing from the mapping.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> model = brainstate.nn.Linear(2, 3)
        >>> graphdef, states = brainstate.graph.flatten(model)
        >>> rebuilt = brainstate.graph.unflatten(graphdef, states)
        >>> isinstance(rebuilt, brainstate.nn.Linear)
        True
    """
    if not isinstance(graphdef, GraphDef):
        raise TypeError(f"graphdef must be a GraphDef, got: {type(graphdef)}")
    if index_ref is None:
        index_ref = {}
    flat_states = (state_mapping.to_flat() if hasattr(state_mapping, 'to_flat')
                   else dict(state_mapping))

    # Pass 0 — materialize States into index_ref (handles all sharing/back-refs).
    for e in _iter_edges(graphdef):
        if isinstance(e, StateEdge) and e.path is not None and e.index not in index_ref:
            value = flat_states.get(e.path, _MISSING)
            if value is not _MISSING:
                index_ref[e.index] = _materialize_state(value, e.index, index_ref_cache)
            elif index_ref_cache is not None and e.index in index_ref_cache:
                index_ref[e.index] = index_ref_cache[e.index]

    # Pass 1 — create/reuse graph-node shells.
    for spec in graphdef.node_specs:
        impl = get_node_impl_for_type(spec.type)
        if index_ref_cache is not None and spec.index in index_ref_cache:
            shell = index_ref_cache[spec.index]
            if type(shell) is not spec.type:
                raise ValueError(
                    f"Expected a node of type {spec.type} for index {spec.index}, "
                    f"but cache held {type(shell)}."
                )
            impl.clear(shell)
        else:
            shell = impl.create_empty(spec.metadata)
        index_ref[spec.index] = shell

    def resolve(edge):
        if isinstance(edge, StaticEdge):
            return edge.value
        if isinstance(edge, NodeEdge):
            return index_ref[edge.index]
        if isinstance(edge, StateEdge):
            if edge.index in index_ref:
                return index_ref[edge.index]
            raise ValueError(
                f"Expected key {_format_path(edge.path) if edge.path else edge.path!r} "
                f"in the state mapping while rebuilding the graph."
            )
        if isinstance(edge, StateLeafEdge):
            value = flat_states.get(edge.path, _MISSING)
            if value is _MISSING:
                raise ValueError(
                    f"Expected key {_format_path(edge.path)} "
                    f"in the state mapping while rebuilding the graph."
                )
            return value.to_state() if isinstance(value, TreefyState) else value
        if isinstance(edge, PytreeEdge):
            items = tuple((k, resolve(c)) for k, c in edge.fields)
            return PYTREE_NODE_IMPL.unflatten(items, edge.metadata)
        raise TypeError(f"Unknown edge type: {type(edge)}")

    # Pass 2 — fill shells.
    for spec in graphdef.node_specs:
        impl = get_node_impl_for_type(spec.type)
        items = tuple((k, resolve(e)) for k, e in spec.fields)
        impl.init(index_ref[spec.index], items)

    return resolve(graphdef.root)
