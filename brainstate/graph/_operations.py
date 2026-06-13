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

"""User-facing graph operations layered over the encode/decode/kernel core.

These are thin compositions of :func:`~brainstate.graph.flatten`,
:func:`~brainstate.graph.unflatten`, and the iteration kernel
(:func:`~brainstate.graph.iter_leaf` / :func:`~brainstate.graph.iter_node`):
splitting/merging (:func:`treefy_split` / :func:`treefy_merge`), state and node
introspection (:func:`states` / :func:`nodes` / :func:`treefy_states`), in-place
update (:func:`update_states`), removal (:func:`pop_states`), deep copy
(:func:`clone`), and structure extraction (:func:`graphdef`).
"""

from __future__ import annotations

from typing import Any, Tuple, Union

from brainstate._state import State, TreefyState
from brainstate._utils import set_module_as
from brainstate.typing import Filter, Predicate
from brainstate.util import NestedDict, FlattedDict
from brainstate.util.filter import to_predicate
from ._flatten import flatten, unflatten
from ._graphdef import GraphDef
from ._walk import (
    iter_leaf, iter_node, _is_node, _is_graph_node, _is_state_leaf,
    _is_node_leaf, _get_node_impl, MAX_INT,
)

__all__ = [
    'treefy_split', 'treefy_merge', 'treefy_states', 'update_states',
    'pop_states', 'states', 'nodes', 'graphdef', 'clone',
]


# ---------------------------------------------------------------------------
# Filter helpers
# ---------------------------------------------------------------------------

def _filters_to_predicates(filters: tuple[Filter, ...]) -> tuple[Predicate, ...]:
    for i, filter_ in enumerate(filters):
        if filter_ in (..., True) and i != len(filters) - 1:
            remaining_filters = filters[i + 1:]
            if not all(f in (..., True) for f in remaining_filters):
                raise ValueError(
                    f'`...` or `True` can only be used as the last filters, '
                    f'got {filter_!r} at index {i}.'
                )
    return tuple(map(to_predicate, filters))


def _split_flatted(flatted, filters: tuple[Filter, ...]):
    predicates = _filters_to_predicates(filters)
    flat_states = tuple([] for _ in predicates)
    for path, value in flatted:
        for i, predicate in enumerate(predicates):
            if predicate(path, value):
                flat_states[i].append((path, value))
                break
        else:
            raise ValueError(
                f'Non-exhaustive filters, got a non-empty remainder: {path} -> {value}.'
                '\nUse `...` to match all remaining elements.'
            )
    return flat_states


def _split_state(state: NestedDict, filters: tuple[Filter, ...]):
    if not filters:
        return (state,)
    states_ = state.split(*filters)
    if isinstance(states_, NestedDict):
        return (states_,)
    return states_


# ---------------------------------------------------------------------------
# Split / merge
# ---------------------------------------------------------------------------

@set_module_as('brainstate.graph')
def treefy_split(node: Any, *filters: Filter) -> Tuple[GraphDef, Any]:
    """Split a graph node into a ``GraphDef`` and one or more state mappings.

    Parameters
    ----------
    node : Any
        The graph node to split.
    *filters : Filter
        Optional predicates/types to partition the state. With no filters the
        single full state mapping is returned; with filters the state is split
        accordingly (use ``...`` as the last filter to capture the rest).

    Returns
    -------
    tuple
        ``(graphdef, state)`` with no filters, or
        ``(graphdef, state1, state2, ...)`` with filters.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> model = brainstate.nn.Linear(2, 3)
        >>> graphdef, params = brainstate.graph.treefy_split(model)
        >>> rebuilt = brainstate.graph.treefy_merge(graphdef, params)
        >>> isinstance(rebuilt, brainstate.nn.Linear)
        True
    """
    graphdef, state_tree = flatten(node)
    states_ = tuple(_split_state(state_tree, filters))
    return graphdef, *states_


@set_module_as('brainstate.graph')
def treefy_merge(graphdef: GraphDef, *state_mappings: NestedDict) -> Any:
    """Reconstruct a node from a ``GraphDef`` and one or more state mappings.

    Parameters
    ----------
    graphdef : GraphDef
        The static structure.
    *state_mappings : NestedDict
        One or more state mappings (e.g. the partitions from
        :func:`treefy_split`); they are merged before reconstruction.

    Returns
    -------
    Any
        The reconstructed node.
    """
    state_mapping = NestedDict.merge(*state_mappings)
    return unflatten(graphdef, state_mapping)


# ---------------------------------------------------------------------------
# State / node introspection
# ---------------------------------------------------------------------------

@set_module_as('brainstate.graph')
def treefy_states(node: Any, *filters: Filter) -> Union[NestedDict, Tuple[NestedDict, ...]]:
    """Return the treefy state mapping of ``node``, optionally filtered.

    Parameters
    ----------
    node : Any
        The graph node.
    *filters : Filter
        Optional filters; when given the mapping is filtered/split.

    Returns
    -------
    NestedDict or tuple of NestedDict
        The treefy state mapping (or partitions when multiple filters given).
    """
    _, state_mapping = flatten(node)
    if not filters:
        return state_mapping
    return state_mapping.filter(*filters)


def _states_generator(node, allowed_hierarchy):
    for path, value in iter_leaf(node, allowed_hierarchy=allowed_hierarchy):
        if isinstance(value, State):
            yield path, value


@set_module_as('brainstate.graph')
def states(
    node: Any,
    *filters: Filter,
    allowed_hierarchy: tuple[int, int] = (0, MAX_INT),
) -> Union[FlattedDict, Tuple[FlattedDict, ...]]:
    """Collect ``State`` objects from a graph node as ``FlattedDict``(s).

    Parameters
    ----------
    node : Any
        The graph node.
    *filters : Filter
        Optional predicates/types to partition the collected states.
    allowed_hierarchy : tuple of (int, int), optional
        ``(lo, hi)`` graph-node depth bounds.

    Returns
    -------
    FlattedDict or tuple of FlattedDict
        All states (single filter or none), or one mapping per filter.
    """
    num_filters = len(filters)
    if num_filters == 0:
        filters = (..., ...)
    else:
        filters = (*filters, ...)
    gen = _states_generator(node, allowed_hierarchy=allowed_hierarchy)
    flat_states = _split_flatted(gen, (*filters, ...))
    state_maps = tuple(FlattedDict(flat) for flat in flat_states)
    return state_maps[0] if num_filters < 2 else state_maps[:num_filters]


@set_module_as('brainstate.graph')
def nodes(
    node: Any,
    *filters: Filter,
    allowed_hierarchy: tuple[int, int] = (0, MAX_INT),
) -> Union[FlattedDict, Tuple[FlattedDict, ...]]:
    """Collect graph nodes as ``FlattedDict``(s), optionally filtered.

    Parameters
    ----------
    node : Any
        The graph node.
    *filters : Filter
        Optional predicates/types to partition the collected nodes.
    allowed_hierarchy : tuple of (int, int), optional
        ``(lo, hi)`` graph-node depth bounds.

    Returns
    -------
    FlattedDict or tuple of FlattedDict
        All nodes (single filter or none), or one mapping per filter.
    """
    num_filters = len(filters)
    if num_filters == 0:
        filters = (..., ...)
    else:
        filters = (*filters, ...)
    gen = iter_node(node, allowed_hierarchy=allowed_hierarchy)
    flat_nodes = _split_flatted(gen, (*filters, ...))
    node_maps = tuple(FlattedDict(flat) for flat in flat_nodes)
    return node_maps[0] if num_filters < 2 else node_maps[:num_filters]


@set_module_as('brainstate.graph')
def graphdef(node: Any) -> GraphDef:
    """Return the ``GraphDef`` of ``node``.

    Parameters
    ----------
    node : Any
        The graph node.

    Returns
    -------
    GraphDef
        The static structure of ``node``.
    """
    return flatten(node)[0]


@set_module_as('brainstate.graph')
def clone(node: Any) -> Any:
    """Deep-copy ``node`` via split/merge (shared references preserved).

    Parameters
    ----------
    node : Any
        The graph node to copy.

    Returns
    -------
    Any
        A structurally identical copy with fresh ``State`` objects.
    """
    graphdef_, state = treefy_split(node)
    return treefy_merge(graphdef_, state)


# ---------------------------------------------------------------------------
# Removal / update (mutating recursions)
# ---------------------------------------------------------------------------

def _graph_pop(node, id_to_index, path_parts, flatted_state_dicts, predicates) -> None:
    if not _is_node(node):
        raise RuntimeError(f'Unsupported type: {type(node)}, this is a bug.')
    if id(node) in id_to_index:
        return
    id_to_index[id(node)] = len(id_to_index)
    impl = _get_node_impl(node)
    for name, value in impl.node_dict(node).items():
        if _is_node(value):
            _graph_pop(value, id_to_index, (*path_parts, name), flatted_state_dicts, predicates)
            continue
        if not _is_node_leaf(value):
            continue
        if id(value) in id_to_index:
            # This ``State`` was already matched and popped via another
            # (shared/tied) reference. Detach this alias too so the popped state
            # leaves no dangling reference behind, but do not record it twice
            # (it is deduplicated by identity). States are only ever entered into
            # ``id_to_index`` when popped, so membership here means "popped".
            if _is_graph_node(node):
                impl.pop_key(node, name)
            continue
        node_path = (*path_parts, name)
        for state_dict, predicate in zip(flatted_state_dicts, predicates):
            if predicate(node_path, value):
                if not _is_graph_node(node):
                    raise ValueError(
                        f'Cannot pop key {name!r} from node of type {type(node).__name__}'
                    )
                id_to_index[id(value)] = len(id_to_index)
                impl.pop_key(node, name)
                state_dict[node_path] = value
                break


@set_module_as('brainstate.graph')
def pop_states(node: Any, *filters: Filter) -> Union[NestedDict, Tuple[NestedDict, ...]]:
    """Remove and return ``State``s matching the filters (deduped by identity).

    Parameters
    ----------
    node : Any
        The graph node to mutate.
    *filters : Filter
        One or more predicates/types selecting which states to remove. At least
        one filter is required.

    Returns
    -------
    NestedDict or tuple of NestedDict
        The removed states, one mapping per filter.

    Raises
    ------
    ValueError
        If no filter is given, or a matched state lives on an immutable node.
    """
    if len(filters) == 0:
        raise ValueError('Expected at least one filter')
    predicates = tuple(to_predicate(f) for f in filters)
    flatted_state_dicts = tuple({} for _ in predicates)
    _graph_pop(node, {}, (), flatted_state_dicts, predicates)
    result = tuple(NestedDict.from_flat(fd) for fd in flatted_state_dicts)
    return result[0] if len(result) == 1 else result


def _graph_update_dynamic(node, state) -> None:
    if not _is_node(node):
        raise RuntimeError(f'Unsupported type: {type(node)}')
    impl = _get_node_impl(node)
    node_dict = impl.node_dict(node)
    for key, value in state.items():
        if key not in node_dict:
            if not _is_graph_node(node):
                raise ValueError(
                    f'Cannot set key {key!r} on immutable node of type {type(node).__name__}'
                )
            if isinstance(value, TreefyState):
                value = value.to_state()
            impl.set_key(node, key, value)
            continue
        current_value = node_dict[key]
        if _is_node(current_value):
            if _is_state_leaf(value):
                raise ValueError(f'Expected a subgraph for {key!r}, but got: {value!r}')
            _graph_update_dynamic(current_value, value)
        elif isinstance(value, State):
            if not isinstance(current_value, State):
                raise ValueError(
                    f'Trying to update a non-State attribute {key!r} with a State: {value!r}'
                )
            current_value.update_from_ref(value.to_state_ref())
        elif isinstance(value, TreefyState):
            if not isinstance(current_value, State):
                raise ValueError(
                    f'Trying to update a non-State attribute {key!r} with a State: {value!r}'
                )
            current_value.update_from_ref(value)
        elif _is_state_leaf(value):
            if not _is_graph_node(node):
                raise ValueError(
                    f'Cannot set key {key!r} on immutable node of type {type(node).__name__}'
                )
            impl.set_key(node, key, value)
        else:
            raise ValueError(f'Unsupported update type: {type(value)} for key {key!r}')


@set_module_as('brainstate.graph')
def update_states(node, state_dict, /, *state_dicts) -> None:
    """Update ``node`` in place from one or more state mappings.

    Parameters
    ----------
    node : Any
        The graph node to update in place.
    state_dict : NestedDict or FlattedDict
        The state mapping to apply.
    *state_dicts : NestedDict or FlattedDict
        Additional mappings; merged with ``state_dict`` before applying.
    """
    if state_dicts:
        state_dict = NestedDict.merge(state_dict, *state_dicts)
    _graph_update_dynamic(node, state_dict.to_dict())
