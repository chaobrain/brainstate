# The file is adapted from the Flax library (https://github.com/google/flax).
# The credit should go to the Flax authors.
#
# Copyright 2024 The Flax Authors.
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

import dataclasses
from collections.abc import Callable, Iterable, Iterator, Mapping, MutableMapping, Sequence
from typing import Any, Generic, TypeVar

import jax
import numpy as np
from typing_extensions import TypeGuard, Unpack

from brainstate._state import State, TreefyState
from brainstate._utils import set_module_as
from brainstate.typing import PathParts, Filter, Predicate, Key
from brainstate.util import (
    PrettyRepr,
    PrettyType,
    PrettyAttr,
    PrettyMapping,
    MappingReprMixin,
    NestedDict,
    FlattedDict,
    PrettyDict,
)
from brainstate.util.filter import to_predicate
from brainstate.util.struct import FrozenDict

__all__ = [
    'register_graph_node_type',

    # state management
    'pop_states',
    'nodes',
    'states',
    'treefy_states',
    'update_states',

    # graph node operations
    'flatten',
    'unflatten',
    'treefy_split',
    'treefy_merge',
    'iter_leaf',
    'iter_node',
    'clone',
    'graphdef',

    # data structures
    'RefMap',
    'GraphDef',
    'NodeDef',
    'NodeRef',
]

MAX_INT = np.iinfo(np.int32).max

A = TypeVar('A')
B = TypeVar('B')
F = TypeVar('F', bound=Callable)
HA = TypeVar('HA')
HB = TypeVar('HB')
N = TypeVar('N')       # generic node type (distinct from the Node base class)
Leaf = TypeVar('Leaf')
AuxData = TypeVar('AuxData')

Index = int
StateLeaf = TreefyState[Any]
NodeLeaf = State[Any]
GraphStateMapping = NestedDict


def _is_state_leaf(x: Any) -> TypeGuard[StateLeaf]:
    return isinstance(x, TreefyState)


def _is_node_leaf(x: Any) -> TypeGuard[NodeLeaf]:
    return isinstance(x, State)


class RefMap(MutableMapping[A, B], MappingReprMixin[A, B]):
    """A mapping that uses object identity (id) as the hash key."""

    __module__ = 'brainstate.graph'

    def __init__(self, mapping: Mapping[A, B] | Iterable[tuple[A, B]] = ()) -> None:
        self._mapping: dict[int, tuple[A, B]] = {}
        self.update(mapping)

    def __getitem__(self, key: A) -> B:
        return self._mapping[id(key)][1]

    def __contains__(self, key: Any) -> bool:
        return id(key) in self._mapping

    def __setitem__(self, key: A, value: B) -> None:
        self._mapping[id(key)] = (key, value)

    def __delitem__(self, key: A) -> None:
        del self._mapping[id(key)]

    def __iter__(self) -> Iterator[A]:
        return (key for key, _ in self._mapping.values())

    def __len__(self) -> int:
        return len(self._mapping)

    def __str__(self) -> str:
        return repr(self)


@dataclasses.dataclass(frozen=True)
class NodeImplBase(Generic[N, Leaf, AuxData]):
    """Base descriptor for a node type, holding its type and flatten function."""

    type: type
    flatten: Callable[[N], tuple[Sequence[tuple[Key, Leaf]], AuxData]]

    def node_dict(self, node: N) -> dict[Key, Leaf]:
        nodes, _ = self.flatten(node)
        return dict(nodes)


@dataclasses.dataclass(frozen=True)
class GraphNodeImpl(NodeImplBase[N, Leaf, AuxData]):
    """Node implementation for registered graph nodes (supports mutation and identity tracking)."""

    set_key: Callable[[N, Key, Leaf], None]
    pop_key: Callable[[N, Key], Leaf]
    create_empty: Callable[[AuxData], N]
    clear: Callable[[N], None]

    def init(self, node: N, items: tuple[tuple[Key, Leaf], ...]) -> None:
        for key, value in items:
            self.set_key(node, key, value)


@dataclasses.dataclass(frozen=True)
class PyTreeNodeImpl(NodeImplBase[N, Leaf, AuxData]):
    """Node implementation for JAX pytree nodes (lists, dicts, tuples — immutable structure)."""

    unflatten: Callable[[tuple[tuple[Key, Leaf], ...], AuxData], N]


NodeImpl = GraphNodeImpl[N, Leaf, AuxData] | PyTreeNodeImpl[N, Leaf, AuxData]

_node_impl_for_type: dict[type, NodeImpl] = {}


def register_graph_node_type(
    type: type,
    flatten: Callable[[N], tuple[Sequence[tuple[Key, Leaf]], AuxData]],
    set_key: Callable[[N, Key, Leaf], None],
    pop_key: Callable[[N, Key], Leaf],
    create_empty: Callable[[AuxData], N],
    clear: Callable[[N], None],
) -> None:
    """Register a custom graph node type with the graph system."""
    _node_impl_for_type[type] = GraphNodeImpl(
        type=type,
        flatten=flatten,
        set_key=set_key,
        pop_key=pop_key,
        create_empty=create_empty,
        clear=clear,
    )


def _is_node(x: Any) -> bool:
    return _is_graph_node(x) or _is_pytree_node(x)


def _is_pytree_node(x: Any) -> bool:
    return not jax.tree_util.all_leaves((x,))


def _is_graph_node(x: Any) -> bool:
    return type(x) in _node_impl_for_type


def _is_node_type(x: type[Any]) -> bool:
    return x in _node_impl_for_type or x is PytreeType


def _get_node_impl(x: Any) -> NodeImpl:
    if isinstance(x, State):
        raise ValueError(f'State is not a node: {x}')
    node_type = type(x)
    if node_type not in _node_impl_for_type:
        if _is_pytree_node(x):
            return PYTREE_NODE_IMPL
        raise ValueError(f'Unknown node type: {x}')
    return _node_impl_for_type[node_type]


def get_node_impl_for_type(x: type[Any]) -> NodeImpl:
    if x is PytreeType:
        return PYTREE_NODE_IMPL
    return _node_impl_for_type[x]


class HashableMapping(Mapping[HA, HB]):
    """An immutable, hashable mapping."""

    def __init__(self, mapping: Mapping[HA, HB] | Iterable[tuple[HA, HB]]) -> None:
        self._mapping = dict(mapping)

    def __contains__(self, key: object) -> bool:
        return key in self._mapping

    def __getitem__(self, key: HA) -> HB:
        return self._mapping[key]

    def __iter__(self) -> Iterator[HA]:
        return iter(self._mapping)

    def __len__(self) -> int:
        return len(self._mapping)

    def __hash__(self) -> int:
        return hash(tuple(sorted(self._mapping.items())))

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, HashableMapping) and self._mapping == other._mapping

    def __repr__(self) -> str:
        return repr(self._mapping)


class GraphDef(Generic[N]):
    """Base class representing the static graph structure of a node."""

    type: type[N]
    index: int


@dataclasses.dataclass(frozen=True, repr=False)
class NodeDef(GraphDef[N], PrettyRepr):
    """Graph structure of a node, containing all static information for reconstruction."""

    type: type[N]
    index: int
    attributes: tuple[Key, ...]
    subgraphs: HashableMapping[Key, NodeDef[Any] | NodeRef[Any]]
    static_fields: HashableMapping
    leaves: HashableMapping[Key, NodeRef[Any] | None]
    metadata: Any
    index_mapping: FrozenDict[Index, Index] | None

    @classmethod
    def create(
        cls,
        type: type[N],
        index: int,
        attributes: tuple[Key, ...],
        subgraphs: Iterable[tuple[Key, NodeDef[Any] | NodeRef[Any]]],
        static_fields: Iterable[tuple],
        leaves: Iterable[tuple[Key, NodeRef[Any] | None]],
        metadata: Any,
        index_mapping: Mapping[Index, Index] | None,
    ) -> NodeDef[N]:
        return cls(
            type=type,
            index=index,
            attributes=attributes,
            subgraphs=HashableMapping(subgraphs),
            static_fields=HashableMapping(static_fields),
            leaves=HashableMapping(leaves),
            metadata=metadata,
            index_mapping=FrozenDict(index_mapping) if index_mapping is not None else None,
        )

    def __pretty_repr__(self):
        yield PrettyType(type=type(self))
        yield PrettyAttr('type', self.type.__name__)
        yield PrettyAttr('index', self.index)
        yield PrettyAttr('attributes', self.attributes)
        yield PrettyAttr('subgraphs', PrettyMapping(self.subgraphs))
        yield PrettyAttr('static_fields', PrettyMapping(self.static_fields))
        yield PrettyAttr('leaves', PrettyMapping(self.leaves))
        yield PrettyAttr('metadata', self.metadata)
        yield PrettyAttr('index_mapping', PrettyMapping(self.index_mapping) if self.index_mapping is not None else None)


jax.tree_util.register_static(NodeDef)


@dataclasses.dataclass(frozen=True, repr=False)
class NodeRef(GraphDef[N], PrettyRepr):
    """A reference to an already-seen node in the graph (used for shared/circular refs)."""

    type: type[N]
    index: int

    def __pretty_repr__(self):
        yield PrettyType(type=type(self))
        yield PrettyAttr('type', self.type.__name__)
        yield PrettyAttr('index', self.index)


jax.tree_util.register_static(NodeRef)


# --------------------------------------------------------
# Graph flatten / unflatten
# --------------------------------------------------------


def _graph_flatten(
    path: PathParts,
    ref_index: RefMap[Any, Index],
    flatted_state_mapping: dict[PathParts, StateLeaf],
    node: Any,
    treefy_state: bool = False,
) -> NodeDef[Any] | NodeRef[Any]:
    """Recursively flatten a graph node into a NodeDef/NodeRef tree."""
    if not _is_node(node):
        raise RuntimeError(f'Unsupported type: {type(node)}, this is a bug.')

    if node in ref_index:
        return NodeRef(type(node), ref_index[node])

    node_impl = _get_node_impl(node)

    if isinstance(node_impl, GraphNodeImpl):
        index = len(ref_index)
        ref_index[node] = index
    else:
        index = -1

    subgraphs: list[tuple[Key, NodeDef[Any] | NodeRef[Any]]] = []
    static_fields: list[tuple] = []
    leaves: list[tuple[Key, NodeRef[Any] | None]] = []

    values, metadata = node_impl.flatten(node)
    for key, value in values:
        if _is_node(value):
            nodedef = _graph_flatten((*path, key), ref_index, flatted_state_mapping, value, treefy_state)
            subgraphs.append((key, nodedef))
        elif isinstance(value, State):
            if value in ref_index:
                leaves.append((key, NodeRef(type(value), ref_index[value])))
            else:
                flatted_state_mapping[(*path, key)] = (value.to_state_ref() if treefy_state else value)
                variable_index = ref_index[value] = len(ref_index)
                leaves.append((key, NodeRef(type(value), variable_index)))
        elif _is_state_leaf(value):
            flatted_state_mapping[(*path, key)] = value
            leaves.append((key, None))
        else:
            static_fields.append((key, value))

    return NodeDef.create(
        type=node_impl.type,
        index=index,
        attributes=tuple(key for key, _ in values),
        subgraphs=subgraphs,
        static_fields=static_fields,
        leaves=leaves,
        metadata=metadata,
        index_mapping=None,
    )


@set_module_as('brainstate.graph')
def flatten(
    node: Any,
    /,
    ref_index: RefMap[Any, Index] | None = None,
    treefy_state: bool = True,
) -> tuple[GraphDef[Any], NestedDict]:
    """Flatten a graph node into a (graph_def, state_mapping) pair."""
    ref_index = RefMap() if ref_index is None else ref_index
    if not isinstance(ref_index, RefMap):
        raise TypeError(f"ref_index must be a RefMap, got: {type(ref_index)}")
    flatted_state_mapping: dict[PathParts, StateLeaf] = {}
    graph_def = _graph_flatten((), ref_index, flatted_state_mapping, node, treefy_state)
    return graph_def, NestedDict.from_flat(flatted_state_mapping)


def _get_children(
    graph_def: NodeDef[Any],
    state_mapping: Mapping,
    index_ref: dict[Index, Any],
    index_ref_cache: dict[Index, Any] | None,
) -> dict[Key, StateLeaf | Any]:
    children: dict[Key, StateLeaf | Any] = {}

    if unknown_keys := set(state_mapping) - set(graph_def.attributes):
        raise ValueError(f'Unknown keys: {unknown_keys}')

    for key in graph_def.attributes:
        if key not in state_mapping:
            if key in graph_def.static_fields:
                children[key] = graph_def.static_fields[key]

            elif key in graph_def.subgraphs:
                subgraphdef = graph_def.subgraphs[key]
                if isinstance(subgraphdef, NodeRef):
                    children[key] = index_ref[subgraphdef.index]
                else:
                    children[key] = _graph_unflatten(subgraphdef, {}, index_ref, index_ref_cache)

            elif key in graph_def.leaves:
                noderef = graph_def.leaves[key]
                if (noderef is not None) and (noderef.index in index_ref):
                    children[key] = index_ref[noderef.index]
                else:
                    raise ValueError(
                        f'Expected key {key!r} in state while building node of type '
                        f'{graph_def.type.__name__}.'
                    )

            else:
                raise RuntimeError(f'Unknown static field: {key!r}')

        else:
            value = state_mapping[key]
            if isinstance(value, PrettyDict):
                value = dict(value)

            if key in graph_def.static_fields:
                raise ValueError(f'Got state for static field {key!r}, this is not supported.')

            if key in graph_def.subgraphs:
                if isinstance(value, (TreefyState, State)):
                    raise ValueError(
                        f'Expected value of type {graph_def.subgraphs[key]} '
                        f'for {key!r}, but got {value!r}'
                    )
                if not isinstance(value, dict):
                    raise TypeError(f'Expected a dict for {key!r}, but got {type(value)}.')

                subgraphdef = graph_def.subgraphs[key]
                if isinstance(subgraphdef, NodeRef):
                    children[key] = index_ref[subgraphdef.index]
                else:
                    children[key] = _graph_unflatten(subgraphdef, value, index_ref, index_ref_cache)

            elif key in graph_def.leaves:
                if not isinstance(value, (TreefyState, State)):
                    raise ValueError(f'Expected a leaf for {key!r}, but got {value!r}')

                noderef = graph_def.leaves[key]
                if noderef is None:
                    if isinstance(value, TreefyState):
                        value = value.to_state()
                    children[key] = value

                elif noderef.index in index_ref:
                    children[key] = index_ref[noderef.index]

                else:
                    if not isinstance(value, (TreefyState, State)):
                        raise ValueError(
                            f'Expected a State type for {key!r}, but got {type(value)}.'
                        )

                    if index_ref_cache is not None and noderef.index in index_ref_cache:
                        variable = index_ref_cache[noderef.index]
                        if not isinstance(variable, State):
                            raise ValueError(f'Expected a State type for {key!r}, but got {type(variable)}.')
                        if isinstance(value, TreefyState):
                            variable.update_from_ref(value)
                        elif isinstance(value, State):
                            if value._been_writen:
                                variable.value = value.value
                            else:
                                variable.restore_value(value.value)
                        else:
                            raise ValueError(f'Expected a State type for {key!r}, but got {type(value)}.')
                    else:
                        if isinstance(value, TreefyState):
                            variable = value.to_state()
                        elif isinstance(value, State):
                            variable = value
                        else:
                            raise ValueError(f'Expected a State type for {key!r}, but got {type(value)}.')
                    children[key] = variable
                    index_ref[noderef.index] = variable

            else:
                raise RuntimeError(f'Unknown key: {key!r}, this is a bug.')

    return children


def _graph_unflatten(
    graph_def: NodeDef[Any] | NodeRef[Any],
    state_mapping: Mapping[Key, StateLeaf | Mapping],
    index_ref: dict[Index, Any],
    index_ref_cache: dict[Index, Any] | None,
) -> Any:
    """Recursively unflatten a graph_def + state_mapping into a node."""
    if isinstance(graph_def, NodeRef):
        return index_ref[graph_def.index]

    if not isinstance(graph_def, NodeDef):
        raise TypeError(f"graph_def must be a NodeDef, got: {type(graph_def)}")

    if not _is_node_type(graph_def.type):
        raise RuntimeError(f'Unsupported type: {graph_def.type}, this is a bug.')

    if graph_def.index in index_ref:
        raise RuntimeError(f'GraphDef index {graph_def.index} already used.')

    node_impl = get_node_impl_for_type(graph_def.type)

    if isinstance(node_impl, GraphNodeImpl):
        if (index_ref_cache is not None) and (graph_def.index in index_ref_cache):
            node = index_ref_cache[graph_def.index]
            if type(node) != graph_def.type:
                raise ValueError(
                    f'Expected a node of type {graph_def.type} for index '
                    f'{graph_def.index}, but got a node of type {type(node)}.'
                )
            node_impl.clear(node)
        else:
            node = node_impl.create_empty(graph_def.metadata)

        index_ref[graph_def.index] = node
        children = _get_children(graph_def, state_mapping, index_ref, index_ref_cache)
        node_impl.init(node, tuple(children.items()))
    else:
        children = _get_children(graph_def, state_mapping, index_ref, index_ref_cache)
        node = node_impl.unflatten(tuple(children.items()), graph_def.metadata)

    return node


@set_module_as('brainstate.graph')
def unflatten(
    graph_def: GraphDef[Any],
    state_mapping: NestedDict,
    /,
    *,
    index_ref: dict[Index, Any] | None = None,
    index_ref_cache: dict[Index, Any] | None = None,
) -> Any:
    """Unflatten a graph_def + state_mapping back into a node."""
    index_ref = {} if index_ref is None else index_ref
    if not isinstance(graph_def, (NodeDef, NodeRef)):
        raise TypeError(f"graph_def must be a NodeDef or NodeRef, got: {type(graph_def)}")
    return _graph_unflatten(graph_def, state_mapping.to_dict(), index_ref, index_ref_cache)


# --------------------------------------------------------
# State extraction / manipulation
# --------------------------------------------------------


def _graph_pop(
    node: Any,
    id_to_index: dict[int, Index],
    path_parts: PathParts,
    flatted_state_dicts: tuple[FlattedDict[PathParts, StateLeaf], ...],
    predicates: tuple[Predicate, ...],
) -> None:
    if not _is_node(node):
        raise RuntimeError(f'Unsupported type: {type(node)}, this is a bug.')

    if id(node) in id_to_index:
        return

    id_to_index[id(node)] = len(id_to_index)
    node_impl = _get_node_impl(node)
    node_dict = node_impl.node_dict(node)

    for name, value in node_dict.items():
        if _is_node(value):
            _graph_pop(
                node=value,
                id_to_index=id_to_index,
                path_parts=(*path_parts, name),
                flatted_state_dicts=flatted_state_dicts,
                predicates=predicates,
            )
            continue
        elif not _is_node_leaf(value) or id(value) in id_to_index:
            continue

        node_path = (*path_parts, name)
        node_impl = _get_node_impl(node)
        for state_dicts, predicate in zip(flatted_state_dicts, predicates):
            if predicate(node_path, value):
                if isinstance(node_impl, PyTreeNodeImpl):
                    raise ValueError(f'Cannot pop key {name!r} from node of type {type(node).__name__}')
                id_to_index[id(value)] = len(id_to_index)
                node_impl.pop_key(node, name)
                state_dicts[node_path] = value
                break


@set_module_as('brainstate.graph')
def pop_states(
    node: Any, *filters: Any
) -> NestedDict | tuple[NestedDict, ...]:
    """Pop one or more State types from the graph node, removing them from the node."""
    if len(filters) == 0:
        raise ValueError('Expected at least one filter')

    id_to_index: dict[int, Index] = {}
    predicates = tuple(to_predicate(filter) for filter in filters)
    flatted_state_dicts: tuple[FlattedDict[PathParts, StateLeaf], ...] = tuple({} for _ in predicates)
    _graph_pop(
        node=node,
        id_to_index=id_to_index,
        path_parts=(),
        flatted_state_dicts=flatted_state_dicts,
        predicates=predicates,
    )
    result = tuple(NestedDict.from_flat(flat_state) for flat_state in flatted_state_dicts)
    return result[0] if len(result) == 1 else result


def _split_state(
    state: GraphStateMapping,
    filters: tuple[Filter, ...],
) -> tuple[GraphStateMapping, Unpack[tuple[GraphStateMapping, ...]]]:
    if not filters:
        return (state,)
    states = state.split(*filters)
    if isinstance(states, NestedDict):
        return (states,)
    return states  # type: ignore[return-value]


@set_module_as('brainstate.graph')
def treefy_split(node: A, *filters: Filter):
    """Split a graph node into a GraphDef and one or more state NestedDicts.

    If no filters are given, returns (graphdef, state).
    With filters, returns (graphdef, state1, state2, ...) split by filter predicates.
    """
    graphdef, state_tree = flatten(node)
    states = tuple(_split_state(state_tree, filters))
    return graphdef, *states


@set_module_as('brainstate.graph')
def treefy_merge(graphdef: GraphDef[A], *state_mappings) -> A:
    """Reconstruct a node from its GraphDef and one or more state NestedDicts."""
    state_mapping = GraphStateMapping.merge(*state_mappings)
    return unflatten(graphdef, state_mapping)


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


def _split_flatted(
    flatted: Iterable[tuple[PathParts, Any]],
    filters: tuple[Filter, ...],
) -> tuple[list[tuple[PathParts, Any]], ...]:
    predicates = _filters_to_predicates(filters)
    flat_states: tuple[list[tuple[PathParts, Any]], ...] = tuple([] for _ in predicates)

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


@set_module_as('brainstate.graph')
def nodes(
    node, *filters: Filter, allowed_hierarchy: tuple[int, int] = (0, MAX_INT)
) -> FlattedDict | tuple[FlattedDict, ...]:
    """Return all graph nodes, optionally filtered and limited by hierarchy depth."""
    num_filters = len(filters)
    if num_filters == 0:
        filters = (..., ...)
    else:
        filters = (*filters, ...)

    nodes_iterable = iter_node(node, allowed_hierarchy=allowed_hierarchy)
    flat_nodes = _split_flatted(nodes_iterable, (*filters, ...))
    node_maps = tuple(FlattedDict(flat_node) for flat_node in flat_nodes)
    if num_filters < 2:
        return node_maps[0]
    return node_maps[:num_filters]


def _states_generator(node, allowed_hierarchy) -> Iterable[tuple[PathParts, State]]:
    for path, value in iter_leaf(node, allowed_hierarchy=allowed_hierarchy):
        if isinstance(value, State):
            yield path, value


@set_module_as('brainstate.graph')
def states(
    node, *filters: Filter, allowed_hierarchy: tuple[int, int] = (0, MAX_INT)
) -> FlattedDict | tuple[FlattedDict, ...]:
    """Return all State objects from a graph node, optionally filtered."""
    num_filters = len(filters)
    if num_filters == 0:
        filters = (..., ...)
    else:
        filters = (*filters, ...)

    states_iterable = _states_generator(node, allowed_hierarchy=allowed_hierarchy)
    flat_states = _split_flatted(states_iterable, (*filters, ...))
    state_maps = tuple(FlattedDict(flat_state) for flat_state in flat_states)
    if num_filters < 2:
        return state_maps[0]
    return state_maps[:num_filters]


@set_module_as('brainstate.graph')
def treefy_states(node, *filters) -> NestedDict | tuple[NestedDict, ...]:
    """Return the treefy state mapping of a graph node, optionally filtered."""
    _, state_mapping = flatten(node)
    if len(filters) == 0:
        return state_mapping
    return state_mapping.filter(*filters)


def _graph_update_dynamic(node: Any, state: Mapping) -> None:
    if not _is_node(node):
        raise RuntimeError(f'Unsupported type: {type(node)}')

    node_impl = _get_node_impl(node)
    node_dict = node_impl.node_dict(node)
    for key, value in state.items():
        if key not in node_dict:
            if isinstance(node_impl, PyTreeNodeImpl):
                raise ValueError(
                    f'Cannot set key {key!r} on immutable node of type {type(node).__name__}'
                )
            if isinstance(value, State):
                value = value.to_state_ref()
            node_impl.set_key(node, key, value)
            continue

        current_value = node_dict[key]

        if _is_node(current_value):
            if _is_state_leaf(value):
                raise ValueError(f'Expected a subgraph for {key!r}, but got: {value!r}')
            _graph_update_dynamic(current_value, value)
        elif isinstance(value, TreefyState):
            if not isinstance(current_value, State):
                raise ValueError(
                    f'Trying to update a non-State attribute {key!r} with a State: {value!r}'
                )
            current_value.update_from_ref(value)
        elif _is_state_leaf(value):
            if isinstance(node_impl, PyTreeNodeImpl):
                raise ValueError(
                    f'Cannot set key {key!r} on immutable node of type {type(node).__name__}'
                )
            node_impl.set_key(node, key, value)
        else:
            raise ValueError(f'Unsupported update type: {type(value)} for key {key!r}')


def update_states(
    node: Any,
    state_dict: NestedDict | FlattedDict,
    /,
    *state_dicts: NestedDict | FlattedDict,
) -> None:
    """Update the graph node in-place with the given state dict(s)."""
    if state_dicts:
        state_dict = NestedDict.merge(state_dict, *state_dicts)
    _graph_update_dynamic(node, state_dict.to_dict())


@set_module_as('brainstate.graph')
def graphdef(node: Any) -> GraphDef[Any]:
    """Return the GraphDef of the given graph node."""
    gdef, _ = flatten(node)
    return gdef


@set_module_as('brainstate.graph')
def clone(node: A) -> A:
    """Create a deep copy of the given graph node."""
    gdef, state = treefy_split(node)
    return treefy_merge(gdef, state)


@set_module_as('brainstate.graph')
def iter_leaf(
    node: Any, allowed_hierarchy: tuple[int, int] = (0, MAX_INT)
) -> Iterator[tuple[PathParts, Any]]:
    """Iterate over all leaf values in the graph node (non-node values).

    Repeated nodes are visited only once. Yields (path, value) pairs.
    """

    def _iter(node_: Any, visited_: set[int], path_: PathParts, level_: int):
        if level_ > allowed_hierarchy[1]:
            return
        if _is_node(node_):
            if id(node_) in visited_:
                return
            visited_.add(id(node_))
            node_dict = _get_node_impl(node_).node_dict(node_)
            for key, value in node_dict.items():
                yield from _iter(
                    value, visited_, (*path_, key),
                    level_ + 1 if _is_graph_node(value) else level_,
                )
        else:
            if level_ >= allowed_hierarchy[0]:
                yield path_, node_

    yield from _iter(node, set(), (), 0)


@set_module_as('brainstate.graph')
def iter_node(
    node: Any, allowed_hierarchy: tuple[int, int] = (0, MAX_INT)
) -> Iterator[tuple[PathParts, Any]]:
    """Iterate over all graph nodes within the given node.

    Repeated nodes are visited only once. Yields (path, node) pairs.
    """

    def _iter(node_: Any, visited_: set[int], path_: PathParts, level_: int):
        if level_ > allowed_hierarchy[1]:
            return
        if _is_node(node_):
            if id(node_) in visited_:
                return
            visited_.add(id(node_))
            node_dict = _get_node_impl(node_).node_dict(node_)
            for key, value in node_dict.items():
                yield from _iter(
                    value, visited_, (*path_, key),
                    level_ + 1 if _is_graph_node(value) else level_,
                )
            if _is_graph_node(node_) and level_ >= allowed_hierarchy[0]:
                yield path_, node_

    yield from _iter(node, set(), (), 0)


# --------------------------------------------------------
# Static wrapper and Pytree support
# --------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Static(Generic[A]):
    """An empty pytree node that treats its inner value as static.

    ``value`` must define ``__eq__`` and ``__hash__``.
    """

    value: A


jax.tree_util.register_static(Static)


class PytreeType:
    ...


def _key_path_to_key(key: Any) -> Key:
    if isinstance(key, jax.tree_util.SequenceKey):
        return key.idx
    elif isinstance(key, (jax.tree_util.DictKey, jax.tree_util.FlattenedIndexKey)):
        if not isinstance(key.key, Key):
            raise ValueError(
                f'Invalid key: {key.key}. May be due to its type not being hashable or comparable.'
            )
        return key.key
    elif isinstance(key, jax.tree_util.GetAttrKey):
        return key.name
    return str(key)


def _flatten_pytree(pytree: Any) -> tuple[tuple[tuple, ...], jax.tree_util.PyTreeDef]:
    leaves, treedef = jax.tree_util.tree_flatten_with_path(pytree, is_leaf=lambda x: x is not pytree)
    nodes = tuple((_key_path_to_key(path[0]), value) for path, value in leaves)
    return nodes, treedef


def _unflatten_pytree(
    nodes: tuple[tuple, ...],
    treedef: jax.tree_util.PyTreeDef,
) -> Any:
    return treedef.unflatten(value for _, value in nodes)


PYTREE_NODE_IMPL = PyTreeNodeImpl(type=PytreeType, flatten=_flatten_pytree, unflatten=_unflatten_pytree)
