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
from typing import (
    Any, Callable, Generic, Iterable, Iterator, Mapping, MutableMapping,
    Sequence, Type, TypeVar, Union, Hashable, Tuple, Dict, Optional
)

import jax
import numpy as np
from typing_extensions import TypeGuard, Unpack

from brainstate._state import State, TreefyState
from brainstate._utils import set_module_as
from brainstate.typing import PathParts, Filter, Predicate, Key
from brainstate.util._pretty_pytree import NestedDict, FlattedDict, PrettyDict
from brainstate.util._pretty_repr import PrettyRepr, PrettyType, PrettyAttr, PrettyMapping, MappingReprMixin
from brainstate.util.filter import to_predicate
from brainstate.util.struct import FrozenDict

__all__ = [
    'register_graph_node_type',

    # state management in the given graph or node
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

    # others
    'RefMap',
    'GraphDef',
    'NodeDef',
    'NodeRef',
]

MAX_INT = np.iinfo(np.int32).max

A = TypeVar('A')
B = TypeVar('B')
C = TypeVar('C')
F = TypeVar('F', bound=Callable)

HA = TypeVar('HA', bound=Hashable)
HB = TypeVar('HB', bound=Hashable)

Index = int
Names = Sequence[int]
Node = TypeVar('Node')
Leaf = TypeVar('Leaf')
AuxData = TypeVar('AuxData')

StateLeaf = TreefyState[Any]
NodeLeaf = State[Any]
GraphStateMapping = NestedDict


# --------------------------------------------------------

def _is_state_leaf(x: Any) -> TypeGuard[StateLeaf]:
    return isinstance(x, TreefyState)


def _is_node_leaf(x: Any) -> TypeGuard[NodeLeaf]:
    return isinstance(x, State)


class RefMap(MutableMapping[A, B], MappingReprMixin[A, B]):
    """
    A mapping that uses object id as the hash for the keys.

    This mapping is useful when we want to keep track of objects
    that are being referenced by other objects.

    Parameters
    ----------
    mapping : Mapping[A, B] or Iterable[Tuple[A, B]], optional
        A mapping or iterable of key-value pairs.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> obj1 = object()
        >>> obj2 = object()
        >>> ref_map = brainstate.graph.RefMap()
        >>> ref_map[obj1] = 'value1'
        >>> ref_map[obj2] = 'value2'
        >>> print(obj1 in ref_map)
        True
        >>> print(ref_map[obj1])
        value1

    """
    __module__ = 'brainstate.graph'

    def __init__(self, mapping: Union[Mapping[A, B], Iterable[Tuple[A, B]]] = ()) -> None:
        self._mapping: Dict[int, Tuple[A, B]] = {}
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
class NodeImplBase(Generic[Node, Leaf, AuxData]):
    type: type
    flatten: Callable[[Node], tuple[Sequence[tuple[Key, Leaf]], AuxData]]

    def node_dict(self, node: Node) -> dict[Key, Leaf]:
        nodes, _ = self.flatten(node)
        return dict(nodes)


@dataclasses.dataclass(frozen=True)
class GraphNodeImpl(NodeImplBase[Node, Leaf, AuxData]):
    set_key: Callable[[Node, Key, Leaf], None]
    pop_key: Callable[[Node, Key], Leaf]
    create_empty: Callable[[AuxData], Node]
    clear: Callable[[Node], None]

    def init(self, node: Node, items: Tuple[Tuple[Key, Leaf], ...]) -> None:
        for key, value in items:
            self.set_key(node, key, value)


@dataclasses.dataclass(frozen=True)
class PyTreeNodeImpl(NodeImplBase[Node, Leaf, AuxData]):
    unflatten: Callable[[tuple[tuple[Key, Leaf], ...], AuxData], Node]


NodeImpl = Union[GraphNodeImpl[Node, Leaf, AuxData], PyTreeNodeImpl[Node, Leaf, AuxData]]

# --------------------------------------------------------
# Graph Node implementation: start
# --------------------------------------------------------

_node_impl_for_type: dict[type, NodeImpl] = {}


def register_graph_node_type(
    type: type,
    flatten: Callable[[Node], tuple[Sequence[tuple[Key, Leaf]], AuxData]],
    set_key: Callable[[Node, Key, Leaf], None],
    pop_key: Callable[[Node, Key], Leaf],
    create_empty: Callable[[AuxData], Node],
    clear: Callable[[Node], None],
):
    """
    Register a graph node type.

    Parameters
    ----------
    type : type
        The type of the node.
    flatten : Callable[[Node], tuple[Sequence[tuple[Key, Leaf]], AuxData]]
        A function that flattens the node into a sequence of key-value pairs.
    set_key : Callable[[Node, Key, Leaf], None]
        A function that sets a key in the node.
    pop_key : Callable[[Node, Key], Leaf]
        A function that pops a key from the node.
    create_empty : Callable[[AuxData], Node]
        A function that creates an empty node.
    clear : Callable[[Node], None]
        A function that clears the node.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> # Custom node type implementation
        >>> class CustomNode:
        ...     def __init__(self):
        ...         self.data = {}
        ...
        >>> def flatten_custom(node):
        ...     return list(node.data.items()), None
        ...
        >>> def set_key_custom(node, key, value):
        ...     node.data[key] = value
        ...
        >>> def pop_key_custom(node, key):
        ...     return node.data.pop(key)
        ...
        >>> def create_empty_custom(metadata):
        ...     return CustomNode()
        ...
        >>> def clear_custom(node):
        ...     node.data.clear()
        ...
        >>> # Register the custom node type
        >>> brainstate.graph.register_graph_node_type(
        ...     CustomNode,
        ...     flatten_custom,
        ...     set_key_custom,
        ...     pop_key_custom,
        ...     create_empty_custom,
        ...     clear_custom
        ... )

    """
    _node_impl_for_type[type] = GraphNodeImpl(
        type=type,
        flatten=flatten,
        set_key=set_key,
        pop_key=pop_key,
        create_empty=create_empty,
        clear=clear,
    )


# --------------------------------------------------------
# Graph node implementation: end
# --------------------------------------------------------


def _is_node(x: Any) -> bool:
    return _is_graph_node(x) or _is_pytree_node(x)


def _is_pytree_node(x: Any) -> bool:
    return not jax.tree_util.all_leaves((x,))


def _is_graph_node(x: Any) -> bool:
    return type(x) in _node_impl_for_type


def _is_node_type(x: Type[Any]) -> bool:
    return x in _node_impl_for_type or x is PytreeType


def _get_node_impl(x: Any) -> NodeImpl:
    if isinstance(x, State):
        raise ValueError(f'State is not a node: {x}')

    node_type = type(x)
    if node_type not in _node_impl_for_type:
        if _is_pytree_node(x):
            return PYTREE_NODE_IMPL
        else:
            raise ValueError(f'Unknown node type: {x}')

    return _node_impl_for_type[node_type]


def get_node_impl_for_type(x: Type[Any]) -> NodeImpl:
    if x is PytreeType:
        return PYTREE_NODE_IMPL
    return _node_impl_for_type[x]


class HashableMapping(Mapping[HA, HB], Hashable):
    def __init__(self, mapping: Union[Mapping[HA, HB], Iterable[tuple[HA, HB]]]) -> None:
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


class GraphDef(Generic[Node]):
    """
    A base dataclass that denotes the graph structure of a :class:`Node`.

    It contains two main components:
    - type: The type of the node.
    - index: The index of the node in the graph.

    It has two concrete subclasses:

    - :class:`NodeRef`: A reference to a node in the graph.
    - :class:`NodeDef`: A dataclass that denotes the graph structure of a :class:`Node` or a :class:`State`.

    Attributes
    ----------
    type : Type[Node]
        The type of the node.
    index : int
        The index of the node in the graph.

    """
    type: Type[Node]
    index: int


@dataclasses.dataclass(frozen=True, repr=False)
class NodeDef(GraphDef[Node], PrettyRepr):
    """
    A dataclass that denotes the tree structure of a node, either :class:`Node` or :class:`State`.

    Attributes
    ----------
    type : Type[Node]
        Type of the node.
    index : int
        Index of the node in the graph.
    attributes : Tuple[Key, ...]
        Attributes for the node.
    subgraphs : HashableMapping[Key, NodeDef[Any] | NodeRef[Any]]
        Mapping of subgraph definitions.
    static_fields : HashableMapping
        Mapping of static fields.
    leaves : HashableMapping[Key, NodeRef[Any] | None]
        Mapping of leaf nodes.
    metadata : Hashable
        Metadata associated with the node.
    index_mapping : FrozenDict[Index, Index] | None
        Index mapping for node references.

    """

    type: Type[Node]  # type of the node
    index: int  # index of the node in the graph
    attributes: Tuple[Key, ...]  # attributes for the node
    subgraphs: HashableMapping[Key, NodeDef[Any] | NodeRef[Any]]
    static_fields: HashableMapping
    leaves: HashableMapping[Key, NodeRef[Any] | None]
    metadata: Hashable
    index_mapping: FrozenDict[Index, Index] | None

    @classmethod
    def create(
        cls,
        type: Type[Node],
        index: int,
        attributes: tuple[Key, ...],
        subgraphs: Iterable[tuple[Key, NodeDef[Any] | NodeRef[Any]]],
        static_fields: Iterable[tuple],
        leaves: Iterable[tuple[Key, NodeRef[Any] | None]],
        metadata: Hashable,
        index_mapping: Mapping[Index, Index] | None,
    ):
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
class NodeRef(GraphDef[Node], PrettyRepr):
    """
    A reference to a node in the graph.

    The node can be instances of :class:`Node` or :class:`State`.

    Attributes
    ----------
    type : Type[Node]
        The type of the node being referenced.
    index : int
        The index of the node in the graph.

    """
    type: Type[Node]
    index: int

    def __pretty_repr__(self):
        yield PrettyType(type=type(self))
        yield PrettyAttr('type', self.type.__name__)
        yield PrettyAttr('index', self.index)


jax.tree_util.register_static(NodeRef)


# --------------------------------------------------------
# Graph operations: start
# --------------------------------------------------------


def _graph_flatten(
    path: PathParts,
    ref_index: RefMap[Any, Index],
    flatted_state_mapping: Dict[PathParts, StateLeaf],
    node: Any,
    treefy_state: bool = False,
) -> Union[NodeDef[Any], NodeRef[Any]]:
    """
    Recursive helper for graph flatten.

    Parameters
    ----------
    path : PathParts
        The path to the node.
    ref_index : RefMap[Any, Index]
        A mapping from nodes to indexes.
    flatted_state_mapping : Dict[PathParts, StateLeaf]
        A mapping from paths to state leaves.
    node : Node
        The node to flatten.
    treefy_state : bool, optional
        Whether to convert states to TreefyState, by default False.

    Returns
    -------
    NodeDef or NodeRef
        A NodeDef or a NodeRef.

    """
    if not _is_node(node):
        raise RuntimeError(f'Unsupported type: {type(node)}, this is a bug.')

    # If the node is already in the cache, return a reference, otherwise
    # add it to the cache and continue with the flattening process.
    # This is done to avoid infinite recursion when there is a reference cycle.
    if node in ref_index:
        return NodeRef(type(node), ref_index[node])

    # Get the node implementation for the node type.
    # There are two types of node implementations: GraphNodeImpl and PyTreeNodeImpl.
    # - ``GraphNodeImpl`` is used for nodes that have a graph structure.
    # - ``PyTreeNodeImpl`` is used for nodes that have a tree structure.
    node_impl = _get_node_impl(node)

    # There are two types of nodes: Node and State.
    # Here we handle the Node case.
    if isinstance(node_impl, GraphNodeImpl):
        # add the node to the cache
        index = len(ref_index)
        ref_index[node] = index
    else:
        index = -1

    subgraphs: list[tuple[Key, Union[NodeDef[Any], NodeRef[Any]]]] = []
    static_fields: list[tuple] = []
    leaves: list[tuple[Key, Union[NodeRef[Any], None]]] = []

    # Flatten the node into a sequence of key-value pairs.
    values, metadata = node_impl.flatten(node)
    for key, value in values:
        if _is_node(value):
            # Recursively flatten the subgraph.
            nodedef = _graph_flatten((*path, key), ref_index, flatted_state_mapping, value, treefy_state)
            subgraphs.append((key, nodedef))
        elif isinstance(value, State):
            # If the variable is in the cache, add a reference to it.
            if value in ref_index:
                leaves.append((key, NodeRef(type(value), ref_index[value])))
            else:
                # If the variable is not in the cache, add it to the cache.
                # This is done to avoid multiple references to the same variable.
                flatted_state_mapping[(*path, key)] = (value.to_state_ref() if treefy_state else value)
                variable_index = ref_index[value] = len(ref_index)
                leaves.append((key, NodeRef(type(value), variable_index)))
        elif _is_state_leaf(value):
            # The instance of ``TreefyState`` is a leaf.
            flatted_state_mapping[(*path, key)] = value
            leaves.append((key, None))
        else:
            # if isinstance(value, (jax.Array, np.ndarray)):
            #   path_str = '/'.join(map(str, (*path, key)))
            #   raise ValueError(f'Arrays leaves are not supported, at {path_str!r}: {value}')

            # The value is a static field.
            static_fields.append((key, value))

    nodedef = NodeDef.create(
        type=node_impl.type,
        index=index,
        attributes=tuple(key for key, _ in values),
        subgraphs=subgraphs,
        static_fields=static_fields,
        leaves=leaves,
        metadata=metadata,
        index_mapping=None,
    )
    return nodedef


@set_module_as('brainstate.graph')
def flatten(
    node: Any,
    /,
    ref_index: Optional[RefMap[Any, Index]] = None,
    treefy_state: bool = True,
) -> Tuple[GraphDef[Any], NestedDict]:
    """
    Flattens a graph node into a (graph_def, state_mapping) pair.

    Parameters
    ----------
    node : Node
        A graph node.
    ref_index : RefMap[Any, Index], optional
        A mapping from nodes to indexes, defaults to None. If not provided, a new
        empty dictionary is created. This argument can be used to flatten a sequence of graph
        nodes that share references.
    treefy_state : bool, optional
        If True, the state mapping will be a NestedDict instead of a flat dictionary.
        Default is True.

    Returns
    -------
    tuple[GraphDef, NestedDict]
        A tuple containing the graph definition and state mapping.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> node = brainstate.graph.Node()
        >>> graph_def, state_mapping = brainstate.graph.flatten(node)
        >>> print(graph_def)
        >>> print(state_mapping)

    """
    ref_index = RefMap() if ref_index is None else ref_index
    assert isinstance(ref_index, RefMap), f"ref_index must be a RefMap. But we got: {ref_index}"
    flatted_state_mapping: dict[PathParts, StateLeaf] = {}
    graph_def = _graph_flatten((), ref_index, flatted_state_mapping, node, treefy_state)
    return graph_def, NestedDict.from_flat(flatted_state_mapping)


def _get_children(
    graph_def: NodeDef[Any],
    state_mapping: Mapping,
    index_ref: dict[Index, Any],
    index_ref_cache: Optional[dict[Index, Any]],
) -> dict[Key, Union[StateLeaf, Any]]:
    children: dict[Key, Union[StateLeaf, Any]] = {}

    # NOTE: we could allow adding new StateLeafs here
    # All state keys must be present in the graph definition (the object attributes)
    if unknown_keys := set(state_mapping) - set(graph_def.attributes):
        raise ValueError(f'Unknown keys: {unknown_keys}')

    # for every key in attributes there are 6 possible cases:
    #  - (2) the key can either be present in the state or not
    #  - (3) the key can be a subgraph, a leaf, or a static attribute
    for key in graph_def.attributes:
        if key not in state_mapping:  # static field
            # Support unflattening with missing keys for static fields and subgraphs
            # This allows partial state restoration and flexible graph reconstruction
            if key in graph_def.static_fields:
                children[key] = graph_def.static_fields[key]

            elif key in graph_def.subgraphs:
                # if the key is a subgraph we create an empty node
                subgraphdef = graph_def.subgraphs[key]
                if isinstance(subgraphdef, NodeRef):
                    # subgraph exists, take it from the cache
                    children[key] = index_ref[subgraphdef.index]

                else:
                    # create a node from an empty state, reasoning:
                    # * it is a node with no state
                    # * it is a node with state but only through references of already
                    #   created nodes
                    substate = {}
                    children[key] = _graph_unflatten(subgraphdef, substate, index_ref, index_ref_cache)

            elif key in graph_def.leaves:
                noderef = graph_def.leaves[key]
                if (noderef is not None) and (noderef.index in index_ref):
                    # variable exists, take it from the cache
                    children[key] = index_ref[noderef.index]

                else:
                    # key for a variable is missing, raise an error
                    raise ValueError(
                        f'Expected key {key!r} in state while building node of type '
                        f'{graph_def.type.__name__}.'
                    )

            else:
                raise RuntimeError(f'Unknown static field: {key!r}')

        else:  # state field
            value = state_mapping[key]
            if isinstance(value, PrettyDict):
                value = dict(value)

            if key in graph_def.static_fields:
                raise ValueError(f'Got state for static field {key!r}, this is not supported.')

            if key in graph_def.subgraphs:
                # if _is_state_leaf(value):
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
                # if not _is_state_leaf(value):
                if not isinstance(value, (TreefyState, State)):
                    raise ValueError(f'Expected a leaf for {key!r}, but got {value!r}')

                noderef = graph_def.leaves[key]
                if noderef is None:
                    # if the leaf is None, it means that the value was originally
                    # a non-TreefyState leaf, however we allow providing a
                    # TreefyState presumbly created by modifying the NestedDict
                    if isinstance(value, TreefyState):
                        value = value.to_state()
                    elif isinstance(value, State):
                        value = value
                    children[key] = value

                elif noderef.index in index_ref:
                    # add an existing variable
                    children[key] = index_ref[noderef.index]

                else:
                    # it is an unseen variable, create a new one
                    if not isinstance(value, (TreefyState, State)):
                        raise ValueError(
                            f'Expected a State type for {key!r}, but got {type(value)}.'
                        )

                    # when idxmap is present, check if the Varable exists there
                    # and update existing variables if it does
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
                    else:  # if it doesn't, create a new variable
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
    graph_def: Union[NodeDef[Any], NodeRef[Any]],
    state_mapping: Mapping[Key, Union[StateLeaf, Mapping]],
    index_ref: dict[Index, Any],
    index_ref_cache: Optional[dict[Index, Any]],
) -> Any:
    """
    Recursive helper for graph unflatten.

    Args:
      graph_def: A `GraphDef` instance or an index to a node in the cache.
      state_mapping: A state mapping from attribute names to variables or subgraphs.
      index_ref: A mapping from indexes to nodes that have been traversed.
                 If a node is already in the cache, it won't be traversed again.
      index_ref_cache: A mapping from indexes to existing nodes that can be reused.
                        When an reference is reused, ``GraphNodeImpl.clear`` is called to leave the
                        object in an empty state and then filled by the unflatten process, as a result
                        existing graph nodes are mutated to have the new content/topology
                        specified by the nodedef.

    Returns:
      A node instance.
    """

    # if the graph_def is a reference, this means that the node has already been created, so
    # we return the node from the cache
    if isinstance(graph_def, NodeRef):
        return index_ref[graph_def.index]
    else:
        assert isinstance(graph_def, NodeDef), f"graph_def must be a NodeDef. But we got: {graph_def}"

    # graph_def must be a registered node type
    if not _is_node_type(graph_def.type):
        raise RuntimeError(f'Unsupported type: {graph_def.type}, this is a bug.')

    # check if the index is already in the cache
    if graph_def.index in index_ref:
        raise RuntimeError(f'GraphDef index {graph_def.index} already used.')

    # get the node implementation for the node type
    node_impl = get_node_impl_for_type(graph_def.type)

    if isinstance(node_impl, GraphNodeImpl):
        # we create an empty node first and add it to the index
        # this avoids infinite recursion when there is a reference cycle

        if (index_ref_cache is not None) and (graph_def.index in index_ref_cache):
            # clear the node to leave it in an empty state
            node = index_ref_cache[graph_def.index]
            if type(node) != graph_def.type:
                raise ValueError(f'Expected a node of type {graph_def.type} for index '
                                 f'{graph_def.index}, but got a node of type {type(node)}.')
            node_impl.clear(node)
        else:
            # create an empty node
            node = node_impl.create_empty(graph_def.metadata)

        # add the node to the cache
        index_ref[graph_def.index] = node

        # get the children (the attributes) of the node
        children = _get_children(graph_def, state_mapping, index_ref, index_ref_cache)

        # initialize the node with the children
        node_impl.init(node, tuple(children.items()))

    else:
        # if the node type does not support the creation of an empty object it means
        # that it cannot reference itself, so we can create its children first

        # first, we create the children (attributes)
        children = _get_children(graph_def, state_mapping, index_ref, index_ref_cache)
        # then, we create the node
        node = node_impl.unflatten(tuple(children.items()), graph_def.metadata)

    return node


@set_module_as('brainstate.graph')
def unflatten(
    graph_def: GraphDef[Any],
    state_mapping: NestedDict,
    /,
    *,
    index_ref: Optional[dict[Index, Any]] = None,
    index_ref_cache: Optional[dict[Index, Any]] = None,
) -> Any:
    """
    Unflattens a graphdef into a node with the given state tree mapping.

    Parameters
    ----------
    graph_def : GraphDef
        A GraphDef instance.
    state_mapping : NestedDict
        A NestedDict instance containing the state mapping.
    index_ref : dict[Index, Any], optional
        A mapping from indexes to nodes references found during the graph
        traversal. If not provided, a new empty dictionary is created. This argument
        can be used to unflatten a sequence of (graphdef, state_mapping) pairs that
        share the same index space.
    index_ref_cache : dict[Index, Any], optional
        A mapping from indexes to existing nodes that can be reused. When a reference
        is reused, ``GraphNodeImpl.clear`` is called to leave the object in an empty
        state and then filled by the unflatten process. As a result, existing graph
        nodes are mutated to have the new content/topology specified by the graphdef.

    Returns
    -------
    Node
        The reconstructed node.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> class MyNode(brainstate.graph.Node):
        ...     def __init__(self):
        ...         self.a = brainstate.nn.Linear(2, 3)
        ...         self.b = brainstate.nn.Linear(3, 4)
        ...
        >>> # Flatten a node
        >>> node = MyNode()
        >>> graphdef, statetree = brainstate.graph.flatten(node)
        >>>
        >>> # Unflatten back to node
        >>> reconstructed_node = brainstate.graph.unflatten(graphdef, statetree)
        >>> assert isinstance(reconstructed_node, MyNode)
        >>> assert isinstance(reconstructed_node.a, brainstate.nn.Linear)
        >>> assert isinstance(reconstructed_node.b, brainstate.nn.Linear)
    """
    index_ref = {} if index_ref is None else index_ref
    assert isinstance(graph_def, (NodeDef, NodeRef)), f"graph_def must be a NodeDef or NodeRef. But we got: {graph_def}"
    node = _graph_unflatten(graph_def, state_mapping.to_dict(), index_ref, index_ref_cache)
    return node


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
        elif not _is_node_leaf(value):
            continue
        elif id(value) in id_to_index:
            continue

        node_path = (*path_parts, name)
        node_impl = _get_node_impl(node)
        for state_dicts, predicate in zip(flatted_state_dicts, predicates):
            if predicate(node_path, value):
                if isinstance(node_impl, PyTreeNodeImpl):
                    raise ValueError(f'Cannot pop key {name!r} from node of type {type(node).__name__}')
                id_to_index[id(value)] = len(id_to_index)
                node_impl.pop_key(node, name)
                # if isinstance(value, State):
                #   value = value.to_state_ref()
                state_dicts[node_path] = value  # type: ignore[index] # mypy is wrong here?
                break
        else:
            # NOTE: should we raise an error here?
            pass


@set_module_as('brainstate.graph')
def pop_states(
    node: Any,  *filters: Any
) -> Union[NestedDict, Tuple[NestedDict, ...]]:
    """
    Pop one or more :class:`State` types from the graph node.

    Parameters
    ----------
    node : Node
        A graph node object.
    *filters
        One or more :class:`State` objects to filter by.

    Returns
    -------
    NestedDict or tuple[NestedDict, ...]
        The popped :class:`NestedDict` containing the :class:`State`
        objects that were filtered for.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> import jax.numpy as jnp

        >>> class Model(brainstate.nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.a = brainstate.nn.Linear(2, 3)
        ...         self.b = brainpy.state.LIF([10, 2])

        >>> model = Model()
        >>> with brainstate.catch_new_states('new'):
        ...     brainstate.nn.init_all_states(model)

        >>> assert len(model.states()) == 2
        >>> model_states = brainstate.graph.pop_states(model, 'new')
        >>> model_states  # doctest: +SKIP
        NestedDict({
          'b': {
            'V': {
              'st': ShortTermState(
                value=Array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                       0., 0., 0.], dtype=float32),
                tag='new'
              )
            }
          }
        })
    """
    if len(filters) == 0:
        raise ValueError('Expected at least one filter')

    id_to_index: dict[int, Index] = {}
    path_parts: PathParts = ()
    predicates = tuple(to_predicate(filter) for filter in filters)
    flatted_state_dicts: tuple[FlattedDict[PathParts, StateLeaf], ...] = tuple({} for _ in predicates)
    _graph_pop(
        node=node,
        id_to_index=id_to_index,
        path_parts=path_parts,
        flatted_state_dicts=flatted_state_dicts,
        predicates=predicates,
    )
    states = tuple(NestedDict.from_flat(flat_state) for flat_state in flatted_state_dicts)

    if len(states) == 1:
        return states[0]
    else:
        return states


def _split_state(
    state: GraphStateMapping,
    filters: tuple[Filter, ...],
) -> tuple[GraphStateMapping, Unpack[tuple[GraphStateMapping, ...]]]:
    if not filters:
        return (state,)
    states = state.split(*filters)
    if isinstance(states, NestedDict):
        return (states,)
    assert len(states) > 0
    return states  # type: ignore[return-value]


@set_module_as('brainstate.graph')
def treefy_split(
    node: A, *filters: Filter
):
    """
    Split a graph node into a :class:`GraphDef` and one or more :class:`NestedDict`s.

    NestedDict is a ``Mapping`` from strings or integers to ``Variables``, Arrays or nested States.
    GraphDef contains all the static information needed to reconstruct a ``Module`` graph, it is
    analogous to JAX's ``PyTreeDef``. :func:`split` is used in conjunction with :func:`merge` to
    switch seamlessly between stateful and stateless representations of the graph.

    Parameters
    ----------
    node : A
        Graph node to split.
    *filters
        Optional filters to group the state into mutually exclusive substates.

    Returns
    -------
    tuple
        ``GraphDef`` and one or more ``States`` equal to the number of filters passed.
        If no filters are passed, a single ``NestedDict`` is returned.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> import jax, jax.numpy as jnp

        >>> class Foo(brainstate.graph.Node):
        ...     def __init__(self):
        ...         self.a = brainstate.nn.BatchNorm1d([10, 2])
        ...         self.b = brainstate.nn.Linear(2, 3)
        ...
        >>> node = Foo()
        >>> graphdef, params, others = brainstate.graph.treefy_split(
        ...     node, brainstate.ParamState, ...
        ... )
        >>> # params contains ParamState variables
        >>> # others contains all other state variables
    """
    graphdef, state_tree = flatten(node)
    states = tuple(_split_state(state_tree, filters))
    return graphdef, *states


@set_module_as('brainstate.graph')
def treefy_merge(graphdef: GraphDef[A], *state_mappings) -> A:
    """
    The inverse of :func:`split`.

    ``merge`` takes a :class:`GraphDef` and one or more :class:`NestedDict`'s and creates
    a new node with the same structure as the original node.

    Parameters
    ----------
    graphdef : GraphDef[A]
        A :class:`GraphDef` object.
    *state_mappings 
        Additional :class:`NestedDict` objects.

    Returns
    -------
    A
        The merged :class:`Module`.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> import jax, jax.numpy as jnp

        >>> class Foo(brainstate.graph.Node):
        ...     def __init__(self):
        ...         self.a = brainstate.nn.BatchNorm1d([10, 2])
        ...         self.b = brainstate.nn.Linear(2, 3)
        ...
        >>> node = Foo()
        >>> graphdef, params, others = brainstate.graph.treefy_split(
        ...     node, brainstate.ParamState, ...
        ... )
        >>> new_node = brainstate.graph.treefy_merge(graphdef, params, others)
        >>> assert isinstance(new_node, Foo)
        >>> assert isinstance(new_node.b, brainstate.nn.BatchNorm1d)
        >>> assert isinstance(new_node.a, brainstate.nn.Linear)
    """
    state_mapping = GraphStateMapping.merge(*state_mappings)
    node = unflatten(graphdef, state_mapping)
    return node


def _filters_to_predicates(filters: Tuple[Filter, ...]) -> Tuple[Predicate, ...]:
    for i, filter_ in enumerate(filters):
        if filter_ in (..., True) and i != len(filters) - 1:
            remaining_filters = filters[i + 1:]
            if not all(f in (..., True) for f in remaining_filters):
                raise ValueError('`...` or `True` can only be used as the last filters, '
                                 f'got {filter_} it at index {i}.')
    return tuple(map(to_predicate, filters))


def _split_flatted(
    flatted: Iterable[tuple[PathParts, Any]],
    filters: tuple[Filter, ...],
) -> tuple[list[tuple[PathParts, Any]], ...]:
    predicates = _filters_to_predicates(filters)

    # we have n + 1 states, where n is the number of predicates
    # the last state is for values that don't match any predicate
    flat_states: tuple[list[tuple[PathParts, Any]], ...] = tuple([] for _ in predicates)

    for path, value in flatted:
        for i, predicate in enumerate(predicates):
            if predicate(path, value):
                flat_states[i].append((path, value))
                break
        else:
            raise ValueError('Non-exhaustive filters, got a non-empty remainder: '
                             f'{path} -> {value}.'
                             '\nUse `...` to match all remaining elements.')

    return flat_states


@set_module_as('brainstate.graph')
def nodes(
    node, *filters: Filter, allowed_hierarchy: Tuple[int, int] = (0, MAX_INT)
):
    """
    Similar to :func:`split` but only returns the :class:`NestedDict`'s indicated by the filters.

    Parameters
    ----------
    node : Node
        The node to get nodes from.
    *filters
        Filters to apply to the nodes.
    allowed_hierarchy : tuple[int, int], optional
        The allowed hierarchy levels, by default (0, MAX_INT).

    Returns
    -------
    FlattedDict or tuple[FlattedDict, ...]
        The filtered nodes.

    """
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


def _states_generator(node, allowed_hierarchy) -> Iterable[Tuple[PathParts, State]]:
    for path, value in iter_leaf(node, allowed_hierarchy=allowed_hierarchy):
        if isinstance(value, State):
            yield path, value


@set_module_as('brainstate.graph')
def states(
    node, *filters: Filter, allowed_hierarchy: Tuple[int, int] = (0, MAX_INT)
) -> Union[FlattedDict, tuple[FlattedDict, ...]]:
    """
    Similar to :func:`split` but only returns the :class:`NestedDict`'s indicated by the filters.

    Parameters
    ----------
    node : Node
        The node to get states from.
    *filters
        Filters to apply to the states.
    allowed_hierarchy : tuple[int, int], optional
        The allowed hierarchy levels, by default (0, MAX_INT).

    Returns
    -------
    FlattedDict or tuple[FlattedDict, ...]
        The filtered states.

    """
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
def treefy_states(
    node, *filters,
):
    """
    Similar to :func:`split` but only returns the :class:`NestedDict`'s indicated by the filters.

    Parameters
    ----------
    node : Node
        A graph node object.
    *filters
        One or more :class:`State` objects to filter by.

    Returns
    -------
    NestedDict or tuple of NestedDict
        One or more :class:`NestedDict` mappings.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> class Model(brainstate.nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.l1 = brainstate.nn.Linear(2, 3)
        ...         self.l2 = brainstate.nn.Linear(3, 4)
        ...     def __call__(self, x):
        ...         return self.l2(self.l1(x))

        >>> model = Model()
        >>> # Get the learnable parameters
        >>> params = brainstate.graph.treefy_states(model, brainstate.ParamState)
        >>> # Get them separately
        >>> params, others = brainstate.graph.treefy_states(
        ...     model, brainstate.ParamState, brainstate.ShortTermState
        ... )
        >>> # Get all states together
        >>> states = brainstate.graph.treefy_states(model)
    """
    _, state_mapping = flatten(node)
    if len(filters) == 0:
        return state_mapping
    else:
        return state_mapping.filter(*filters)


def _graph_update_dynamic(node: Any, state: Mapping) -> None:
    if not _is_node(node):
        raise RuntimeError(f'Unsupported type: {type(node)}')

    node_impl = _get_node_impl(node)
    node_dict = node_impl.node_dict(node)
    for key, value in state.items():
        # case 1: new state is being added
        if key not in node_dict:
            if isinstance(node_impl, PyTreeNodeImpl):
                raise ValueError(f'Cannot set key {key!r} on immutable node of '
                                 f'type {type(node).__name__}')
            if isinstance(value, State):
                # TODO: here maybe error? we should check if the state already belongs to another node?
                value = value.to_state_ref()  # Convert to state reference for proper state management
            node_impl.set_key(node, key, value)
            continue

        # check values are of the same type
        current_value = node_dict[key]

        # case 2: subgraph is being updated
        if _is_node(current_value):
            if _is_state_leaf(value):
                raise ValueError(f'Expected a subgraph for {key!r}, but got: {value!r}')
            _graph_update_dynamic(current_value, value)
        elif isinstance(value, TreefyState):
            # case 3: state leaf is being updated
            if not isinstance(current_value, State):
                raise ValueError(f'Trying to update a non-State attribute {key!r} with a State: '
                                 f'{value!r}')
            current_value.update_from_ref(value)
        elif _is_state_leaf(value):
            # case 4: state field is being updated
            if isinstance(node_impl, PyTreeNodeImpl):
                raise ValueError(f'Cannot set key {key!r} on immutable node of '
                                 f'type {type(node).__name__}')
            node_impl.set_key(node, key, value)
        else:
            raise ValueError(f'Unsupported update type: {type(value)} for key {key!r}')


def update_states(
    node: Any,
    state_dict: Union[NestedDict, FlattedDict],
    /,
    *state_dicts: Union[NestedDict, FlattedDict]
) -> None:
    """
    Update the given graph node with a new :class:`NestedMapping` in-place.

    Parameters
    ----------
    node : Node
        A graph node to update.
    state_dict : NestedDict | FlattedDict
        A :class:`NestedMapping` object.
    *state_dicts : NestedDict | FlattedDict
        Additional :class:`NestedMapping` objects.

    """
    if state_dicts:
        state_dict = NestedDict.merge(state_dict, *state_dicts)
    _graph_update_dynamic(node, state_dict.to_dict())


@set_module_as('brainstate.graph')
def graphdef(node: Any) -> GraphDef[Any]:
    """
    Get the :class:`GraphDef` of the given graph node.

    Parameters
    ----------
    node : Any
        A graph node object.

    Returns
    -------
    GraphDef[Any]
        The :class:`GraphDef` of the :class:`Module` object.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate

        >>> model = brainstate.nn.Linear(2, 3)
        >>> graphdef, _ = brainstate.graph.treefy_split(model)
        >>> assert graphdef == brainstate.graph.graphdef(model)

    """
    graphdef, _ = flatten(node)
    return graphdef


@set_module_as('brainstate.graph')
def clone(node: A) -> A:
    """
    Create a deep copy of the given graph node.

    Parameters
    ----------
    node : Node
        A graph node object.

    Returns
    -------
    Node
        A deep copy of the :class:`Module` object.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> model = brainstate.nn.Linear(2, 3)
        >>> cloned_model = brainstate.graph.clone(model)
        >>> model.weight.value['bias'] += 1
        >>> assert (model.weight.value['bias'] != cloned_model.weight.value['bias']).all()

    """
    graphdef, state = treefy_split(node)
    return treefy_merge(graphdef, state)


@set_module_as('brainstate.graph')
def iter_leaf(
    node: Any, allowed_hierarchy: Tuple[int, int] = (0, MAX_INT)
) -> Iterator[tuple[PathParts, Any]]:
    """
    Iterates over all nested leaves in the given graph node, including the current node.

    ``iter_graph`` creates a generator that yields path and value pairs, where
    the path is a tuple of strings or integers representing the path to the value from the
    root. Repeated nodes are visited only once. Leaves include static values.

    Parameters
    ----------
    node : Any
        The node to iterate over.
    allowed_hierarchy : tuple[int, int], optional
        The allowed hierarchy levels, by default (0, MAX_INT).

    Yields
    ------
    Iterator[tuple[PathParts, Any]]
        Path and value pairs.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> import jax.numpy as jnp

        >>> class Linear(brainstate.nn.Module):
        ...     def __init__(self, din, dout):
        ...         super().__init__()
        ...         self.weight = brainstate.ParamState(brainstate.random.randn(din, dout))
        ...         self.bias = brainstate.ParamState(brainstate.random.randn(dout))
        ...         self.a = 1
        ...
        >>> module = Linear(3, 4)
        ...
        >>> for path, value in brainstate.graph.iter_leaf([module, module]):
        ...     print(path, type(value).__name__)
        ...
        (0, 'a') int
        (0, 'bias') ParamState
        (0, 'weight') ParamState

    """

    def _iter_graph_leaf(
        node_: Any,
        visited_: set[int],
        path_parts_: PathParts,
        level_: int,
    ) -> Iterator[tuple[PathParts, Any]]:
        if level_ > allowed_hierarchy[1]:
            return

        if _is_node(node_):
            if id(node_) in visited_:
                return
            visited_.add(id(node_))
            node_dict = _get_node_impl(node_).node_dict(node_)
            for key, value in node_dict.items():
                yield from _iter_graph_leaf(
                    value,
                    visited_,
                    (*path_parts_, key),
                    level_ + 1 if _is_graph_node(value) else level_
                )
        else:
            if level_ >= allowed_hierarchy[0]:
                yield path_parts_, node_

    visited: set[int] = set()
    path_parts: PathParts = ()
    level: int = 0
    yield from _iter_graph_leaf(node, visited, path_parts, level)


@set_module_as('brainstate.graph')
def iter_node(
    node: Any, allowed_hierarchy: Tuple[int, int] = (0, MAX_INT)
) -> Iterator[Tuple[PathParts, Any]]:
    """
    Iterates over all nested nodes of the given graph node, including the current node.

    ``iter_graph`` creates a generator that yields path and value pairs, where
    the path is a tuple of strings or integers representing the path to the value from the
    root. Repeated nodes are visited only once. Leaves include static values.

    Parameters
    ----------
    node : Any
        The node to iterate over.
    allowed_hierarchy : tuple[int, int], optional
        The allowed hierarchy levels, by default (0, MAX_INT).

    Yields
    ------
    Iterator[tuple[PathParts, Any]]
        Path and node pairs.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> import jax.numpy as jnp

        >>> class Model(brainstate.nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.a = brainstate.nn.Linear(1, 2)
        ...         self.b = brainstate.nn.Linear(2, 3)
        ...         self.c = [brainstate.nn.Linear(3, 4), brainstate.nn.Linear(4, 5)]
        ...         self.d = {'x': brainstate.nn.Linear(5, 6), 'y': brainstate.nn.Linear(6, 7)}
        ...         self.b.a = brainpy.state.LIF(2)
        ...
        >>> model = Model()
        ...
        >>> for path, node in brainstate.graph.iter_node([model, model]):
        ...     print(path, node.__class__.__name__)
        ...
        (0, 'a') Linear
        (0, 'b', 'a') LIF
        (0, 'b') Linear
        (0, 'c', 0) Linear
        (0, 'c', 1) Linear
        (0, 'd', 'x') Linear
        (0, 'd', 'y') Linear
        (0,) Model

    """

    def _iter_graph_node(
        node_: Any,
        visited_: set[int],
        path_parts_: PathParts,
        level_: int,
    ) -> Iterator[tuple[PathParts, Any]]:
        if level_ > allowed_hierarchy[1]:
            return

        if _is_node(node_):
            if id(node_) in visited_:
                return

            visited_.add(id(node_))
            node_dict = _get_node_impl(node_).node_dict(node_)
            for key, value in node_dict.items():
                yield from _iter_graph_node(value, visited_, (*path_parts_, key),
                                            level_ + 1 if _is_graph_node(value) else level_)

            if _is_graph_node(node_) and level_ >= allowed_hierarchy[0]:
                yield path_parts_, node_

    visited: set[int] = set()
    path_parts: PathParts = ()
    level: int = 0
    yield from _iter_graph_node(node, visited, path_parts, level)


# --------------------------------------------------------
# Graph operations: end
# --------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Static(Generic[A]):
    """
    An empty pytree node that treats its inner value as static.

    ``value`` must define ``__eq__`` and ``__hash__``.

    Attributes
    ----------
    value : A
        The static value to wrap.

    """

    value: A


jax.tree_util.register_static(Static)


# ---------------------------------------------------------
# Pytree
# ---------------------------------------------------------

class PytreeType:
    ...


def _key_path_to_key(key: Any) -> Key:
    if isinstance(key, jax.tree_util.SequenceKey):
        return key.idx
    elif isinstance(
        key, (jax.tree_util.DictKey, jax.tree_util.FlattenedIndexKey)
    ):
        if not isinstance(key.key, Key):
            raise ValueError(
                f'Invalid key: {key.key}. May be due to its type not being hashable or comparable.'
            )
        return key.key
    elif isinstance(key, jax.tree_util.GetAttrKey):
        return key.name
    else:
        return str(key)


def _flatten_pytree(pytree: Any) -> Tuple[Tuple[Tuple, ...], jax.tree_util.PyTreeDef]:
    leaves, treedef = jax.tree_util.tree_flatten_with_path(pytree, is_leaf=lambda x: x is not pytree)
    nodes = tuple((_key_path_to_key(path[0]), value) for path, value in leaves)
    return nodes, treedef


def _unflatten_pytree(
    nodes: tuple[tuple, ...],
    treedef: jax.tree_util.PyTreeDef
) -> Any:
    pytree = treedef.unflatten(value for _, value in nodes)
    return pytree


PYTREE_NODE_IMPL = PyTreeNodeImpl(type=PytreeType, flatten=_flatten_pytree, unflatten=_unflatten_pytree)
