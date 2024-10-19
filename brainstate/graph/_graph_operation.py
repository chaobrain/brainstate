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
from typing import (Any, Callable, Generic, Iterable, Iterator, Mapping, MutableMapping,
                    Sequence, Type, TypeVar, Union, Hashable, Tuple, Dict, Optional, overload)

import jax
import numpy as np
from typing_extensions import TypeGuard, Unpack

from brainstate._state import State, StateRef
from brainstate.typing import PathParts, Filter, Predicate, Key
from brainstate.util._caller import ApplyCaller, CallableProxy, DelayedAccessor
from brainstate.util._dict import FrozenDict
from brainstate.util._filter import to_predicate
from brainstate.util._mapping import FlattedStateMapping, NestedMapping, FlattedMapping
from brainstate.util._pretty_repr import PrettyRepr, PrettyType, PrettyAttr, PrettyMapping, MappingReprMixin

_max_int = np.iinfo(np.int32).max

__all__ = [
  # state management in the given graph or node
  'pop_states', 'nodes', 'states', 'state_refs', 'update_states',

  # graph node operations
  'flatten', 'unflatten', 'split', 'merge', 'iter_leaf', 'iter_node', 'clone', 'graphdef',
]

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

StateLeaf = StateRef[Any]
NodeLeaf = State[Any]
GraphStateMapping = NestedMapping[Key, StateLeaf]


# --------------------------------------------------------


def _is_state_leaf(x: Any) -> TypeGuard[StateLeaf]:
  return isinstance(x, StateRef)


def _is_node_leaf(x: Any) -> TypeGuard[NodeLeaf]:
  return isinstance(x, State)


class RefMap(MutableMapping[A, B], MappingReprMixin[A, B]):
  """
  A mapping that uses object id as the hash for the keys.

  This mapping is useful when we want to keep track of objects
  that are being referenced by other objects.

  Args:
    mapping: A mapping or iterable of key-value pairs.

  """

  def __init__(self, mapping: Mapping[A, B] | Iterable[Tuple[A, B]] = ()):
    self._mapping: Dict[int, Tuple[A, B]] = {}
    self.update(mapping)

  def __getitem__(self, key: A) -> B:
    return self._mapping[id(key)][1]

  def __contains__(self, key: Any) -> bool:
    return id(key) in self._mapping

  def __setitem__(self, key: A, value: B):
    self._mapping[id(key)] = (key, value)

  def __delitem__(self, key: A):
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

  def init(self, node: Node, items: Tuple[Tuple[Key, Leaf], ...]):
    for key, value in items:
      self.set_key(node, key, value)


@dataclasses.dataclass(frozen=True)
class PyTreeNodeImpl(NodeImplBase[Node, Leaf, AuxData]):
  unflatten: Callable[[tuple[tuple[Key, Leaf], ...], AuxData], Node]


NodeImpl = GraphNodeImpl[Node, Leaf, AuxData] | PyTreeNodeImpl[Node, Leaf, AuxData]

# --------------------------------------------------------
# Graph Node implementation: start
# --------------------------------------------------------

_node_impl_for_type: dict[type, NodeImpl[Any, Any, Any]] = {}


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

  Args:
    type: The type of the node.
    flatten: A function that flattens the node into a sequence of key-value pairs.
    set_key: A function that sets a key in the node.
    pop_key: A function that pops a key from the node.
    create_empty: A function that creates an empty node.
    clear: A function that clears the node
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


def _is_node_type(x: type[Any]) -> bool:
  return x in _node_impl_for_type or x is PytreeType


def _get_node_impl(x: Node) -> NodeImpl[Node, Any, Any]:
  if isinstance(x, State):
    raise ValueError(f'State is not a node: {x}')

  node_type = type(x)
  if node_type not in _node_impl_for_type:
    if _is_pytree_node(x):
      return PYTREE_NODE_IMPL
    else:
      raise ValueError(f'Unknown node type: {x}')

  return _node_impl_for_type[node_type]


def get_node_impl_for_type(x: type[Node]) -> NodeImpl[Node, Any, Any]:
  if x is PytreeType:
    return PYTREE_NODE_IMPL
  return _node_impl_for_type[x]


class HashableMapping(Mapping[HA, HB], Hashable):
  def __init__(self, mapping: Mapping[HA, HB] | Iterable[tuple[HA, HB]]):
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


class GraphDefinition(Generic[Node]):
  """
  A base dataclass that denotes the graph structure of a :class:`Node`.

  It contains two main components:
  - type: The type of the node.
  - index: The index of the node in the graph.

  It has two concrete subclasses:
  - :class:`NodeRef`: A reference to a node in the graph.
  - :class:`NodeDef`: A dataclass that denotes the graph structure of a :class:`Node` or a :class:`State`.

  """
  type: type[Node]
  index: int


@dataclasses.dataclass(frozen=True, repr=False)
class NodeRef(GraphDefinition[Node], PrettyRepr):
  """
  A reference to a node in the graph.

  The node can be instances of :class:`Node` or :class:`State`.
  """
  type: type[Node]
  index: int

  def __pretty_repr__(self):
    yield PrettyType(type=type(self))
    yield PrettyAttr('type', self.type.__name__)
    yield PrettyAttr('index', self.index)

  def __treescope_repr__(self, path, subtree_renderer):
    """
    Treescope repr for the object.
    """
    import treescope  # type: ignore[import-not-found,import-untyped]
    return treescope.repr_lib.render_object_constructor(
      object_type=type(self),
      attributes={'type': self.type, 'index': self.index},
      path=path,
      subtree_renderer=subtree_renderer,
    )


jax.tree_util.register_static(NodeRef)


@dataclasses.dataclass(frozen=True, repr=False)
class NodeDef(GraphDefinition[Node], PrettyRepr):
  """
  A dataclass that denotes the tree structure of a node, either :class:`Node` or :class:`State`.

  """

  type: Type[Node]  # type of the node
  index: int  # index of the node in the graph
  attributes: Tuple[Key, ...]  # attributes for the node
  subgraphs: HashableMapping[Key, NodeDef[Any] | NodeRef[Any]]
  static_fields: HashableMapping[Key, Any]
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
      static_fields: Iterable[tuple[Key, Any]],
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

  def __treescope_repr__(self, path, subtree_renderer):
    import treescope  # type: ignore[import-not-found,import-untyped]
    return treescope.repr_lib.render_object_constructor(
      object_type=type(self),
      attributes={'type': self.type,
                  'index': self.index,
                  'attributes': self.attributes,
                  'subgraphs': dict(self.subgraphs),
                  'static_fields': dict(self.static_fields),
                  'leaves': dict(self.leaves),
                  'metadata': self.metadata, },
      path=path,
      subtree_renderer=subtree_renderer,
    )

  def apply(
      self,
      state_map: GraphStateMapping,
      *state_maps: GraphStateMapping
  ) -> ApplyCaller[tuple[GraphDefinition[Node], GraphStateMapping]]:
    accessor = DelayedAccessor()

    def _apply(accessor: DelayedAccessor, *args, **kwargs) -> tuple[
      Any, tuple[GraphDefinition[Node], GraphStateMapping]]:
      module = merge(self, state_map, *state_maps)
      fn = accessor(module)
      out = fn(*args, **kwargs)
      return out, flatten(module)

    return CallableProxy(_apply, accessor)  # type: ignore


jax.tree_util.register_static(NodeDef)


# --------------------------------------------------------
# Graph operations: start
# --------------------------------------------------------


def _graph_flatten(
    path: PathParts,
    ref_index: RefMap[Any, Index],
    flatted_state_mapping: Dict[PathParts, StateLeaf],
    node: Node,
) -> NodeDef[Node] | NodeRef:
  """
  Recursive helper for graph flatten.

  Args:
    path: The path to the node.
    ref_index: A mapping from nodes to indexes.
    flatted_state_mapping: A mapping from paths to state leaves.
    node: The node to flatten.

  Returns:
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

  subgraphs: list[tuple[Key, NodeDef[Node] | NodeRef]] = []
  static_fields: list[tuple[Key, Any]] = []
  leaves: list[tuple[Key, NodeRef | None]] = []

  # Flatten the node into a sequence of key-value pairs.
  values, metadata = node_impl.flatten(node)
  for key, value in values:
    if _is_node(value):
      # Recursively flatten the subgraph.
      nodedef = _graph_flatten((*path, key), ref_index, flatted_state_mapping, value)
      subgraphs.append((key, nodedef))
    elif isinstance(value, State):
      # If the variable is in the cache, add a reference to it.
      if value in ref_index:
        leaves.append((key, NodeRef(type(value), ref_index[value])))
      else:
        # If the variable is not in the cache, add it to the cache.
        # This is done to avoid multiple references to the same variable.
        flatted_state_mapping[(*path, key)] = value.to_state_ref()
        variable_index = ref_index[value] = len(ref_index)
        leaves.append((key, NodeRef(type(value), variable_index)))
    elif _is_state_leaf(value):
      # The instance of ``StateRef`` is a leaf.
      flatted_state_mapping[(*path, key)] = value
      leaves.append((key, None))
    else:
      # if isinstance(value, (jax.Array, np.ndarray)):
      #   path_str = '/'.join(map(str, (*path, key)))
      #   raise ValueError(f'Arrays leaves are not supported, at {path_str!r}: {value}')

      # The value is a static field.
      static_fields.append((key, value))

  nodedef = NodeDef.create(type=node_impl.type,
                           index=index,
                           attributes=tuple(key for key, _ in values),
                           subgraphs=subgraphs,
                           static_fields=static_fields,
                           leaves=leaves,
                           metadata=metadata,
                           index_mapping=None, )
  return nodedef


def flatten(
    node: Node,
    /,
    ref_index: Optional[RefMap[Any, Index]] = None
) -> Tuple[GraphDefinition[Node], GraphStateMapping]:
  """
  Flattens a graph node into a (graph_def, state_mapping) pair.

  Example::

      >>> import brainstate as bst
      >>> node = bst.graph.Node()
      >>> graph_def, state_mapping = flatten(node)
      >>> print(graph_def)
      >>> print(state_mapping)

  Args:
    node: A graph node.
    ref_index: A mapping from nodes to indexes, defaults to None. If not provided, a new
               empty dictionary is created. This argument can be used to flatten a sequence of graph
               nodes that share references.
  """
  ref_index = RefMap() if ref_index is None else ref_index
  flatted_state_mapping: dict[PathParts, StateLeaf] = {}
  graph_def = _graph_flatten((), ref_index, flatted_state_mapping, node)
  return graph_def, GraphStateMapping.from_flat(flatted_state_mapping)


def _get_children(graph_def, state_mapping, index_ref, index_ref_cache):
  children: dict[Key, StateLeaf | Node] = {}

  # NOTE: we could allow adding new StateLeafs here
  # All state keys must be present in the graph definition (the object attributes)
  if unknown_keys := set(state_mapping) - set(graph_def.attributes):
    raise ValueError(f'Unknown keys: {unknown_keys}')

  # for every key in attributes there are 6 possible cases:
  #  - (2) the key can either be present in the state or not
  #  - (3) the key can be a subgraph, a leaf, or a static attribute
  for key in graph_def.attributes:
    if key not in state_mapping:  # static field
      # TODO(cgarcia): maybe we shouldn't support unflattening with missing keys?
      # if key is not present, create an empty types
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
          raise ValueError(f'Expected key {key!r} in state while building node of type '
                           f'{graph_def.type.__name__}.')

      else:
        raise RuntimeError(f'Unknown static field: {key!r}')

    else:  # state field
      value = state_mapping[key]
      if key in graph_def.static_fields:
        raise ValueError(f'Got state for static field {key!r}, this is not supported.')

      if key in graph_def.subgraphs:
        if _is_state_leaf(value):
          raise ValueError(f'Expected value of type {graph_def.subgraphs[key]} for {key!r}, but got {value!r}')
        if not isinstance(value, dict):
          raise TypeError(f'Expected a dict for {key!r}, but got {type(value)}.')

        subgraphdef = graph_def.subgraphs[key]
        if isinstance(subgraphdef, NodeRef):
          children[key] = index_ref[subgraphdef.index]
        else:
          children[key] = _graph_unflatten(subgraphdef, value, index_ref, index_ref_cache)

      elif key in graph_def.leaves:
        if not _is_state_leaf(value):
          raise ValueError(f'Expected a leaf for {key!r}, but got {value!r}')

        noderef = graph_def.leaves[key]
        if noderef is None:
          # if the leaf is None, it means that the value was originally
          # a non-StateRef leaf, however we allow providing a
          # StateRef presumbly created by modifying the NestedMapping
          if isinstance(value, StateRef):
            value = value.to_state()
          children[key] = value

        elif noderef.index in index_ref:
          # add an existing variable
          children[key] = index_ref[noderef.index]

        else:
          # it is an unseen variable, create a new one
          if not isinstance(value, StateRef):
            raise ValueError(f'Expected a State type for {key!r}, but got {type(value)}.')
          # when idxmap is present, check if the Varable exists there
          # and update existing variables if it does
          if index_ref_cache is not None and noderef.index in index_ref_cache:
            variable = index_ref_cache[noderef.index]
            if not isinstance(variable, State):
              raise ValueError(f'Expected a State type for {key!r}, but got {type(variable)}.')
            variable.update_from_ref(value)
          else:  # if it doesn't, create a new variable
            assert isinstance(value, StateRef)
            variable = value.to_state()
          children[key] = variable
          index_ref[noderef.index] = variable

      else:
        raise RuntimeError(f'Unknown key: {key!r}, this is a bug.')

  return children


def _graph_unflatten(
    graph_def: NodeDef[Node] | NodeRef[Node],
    state_mapping: Mapping[Key, StateLeaf | Mapping[Key, Any]],
    index_ref: dict[Index, Any],
    index_ref_cache: dict[Index, Any] | None,
) -> Node:
  """
  Recursive helper for graph unflatten.

  Args:
    graph_def: A `GraphDefinition` instance or an index to a node in the cache.
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
    raise RuntimeError(f'GraphDefinition index {graph_def.index} already used.')

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


def unflatten(
    graph_def: GraphDefinition[Node],
    state_mapping: GraphStateMapping,
    /,
    *,
    index_ref: dict[Index, Any] | None = None,
    index_ref_cache: dict[Index, Any] | None = None,
) -> Node:
  """
  Unflattens a graphdef into a node with the given state mapping.

  Args:
    graph_def: A GraphDefinition instance.
    state_mapping: A NestedMapping instance.
    index_ref: A mapping from indexes to nodes references found during the graph
               traversal, defaults to None. If not provided, a new empty dictionary is
               created. This argument can be used to unflatten a sequence of (graphdef, state_mapping)
               pairs that share the same index space.
    index_ref_cache: A mapping from indexes to existing nodes that can be reused.
                     When a reference is reused, ``GraphNodeImpl.clear`` is called to leave the
                     object in an empty state and then filled by the unflatten process, as a result
                     existing graph nodes are mutated to have the new content/topology
                     specified by the graphdef.
  """
  index_ref = {} if index_ref is None else index_ref
  assert isinstance(graph_def, (NodeDef, NodeRef)), f"graph_def must be a NodeDef or NodeRef. But we got: {graph_def}"
  node = _graph_unflatten(graph_def, state_mapping.raw_mapping, index_ref, index_ref_cache)
  return node


def graph_pop(
    node: Node,
    filters: Tuple[Filter, ...],
) -> Tuple[GraphStateMapping, ...]:
  """
  Pop one or more variables from the graph node.

  Args:
    node: A graph node.
    filters: One or more filters to filter the variables to pop.
  """
  id_to_index: dict[int, Index] = {}
  path_parts: PathParts = ()
  predicates = tuple(to_predicate(filter) for filter in filters)
  flatted_state_mappings: tuple[FlattedStateMapping[StateLeaf], ...] = tuple({} for _ in predicates)
  _graph_pop(node, id_to_index, path_parts, flatted_state_mappings, predicates)
  return tuple(GraphStateMapping.from_flat(flat_state) for flat_state in flatted_state_mappings)


def _graph_pop(
    node: Node,
    id_to_index: dict[int, Index],
    path_parts: PathParts,
    flatted_state_mappings: tuple[FlattedStateMapping[StateLeaf], ...],
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
        flatted_state_mappings=flatted_state_mappings,
        predicates=predicates,
      )
      continue
    elif not _is_node_leaf(value):
      continue
    elif id(value) in id_to_index:
      continue

    node_path = (*path_parts, name)
    node_impl = _get_node_impl(node)
    for state_mapping, predicate in zip(flatted_state_mappings, predicates):
      if predicate(node_path, value):
        if isinstance(node_impl, PyTreeNodeImpl):
          raise ValueError(f'Cannot pop key {name!r} from node of type {type(node).__name__}')
        id_to_index[id(value)] = len(id_to_index)
        node_impl.pop_key(node, name)
        if isinstance(value, State):
          value = value.to_state_ref()
        state_mapping[node_path] = value  # type: ignore[index] # mypy is wrong here?
        break
    else:
      # NOTE: should we raise an error here?
      pass


@overload
def pop_states(node, filter1: Filter, /) -> GraphStateMapping: ...


@overload
def pop_states(node, filter1: Filter, filter2: Filter, /, *filters: Filter) -> tuple[GraphStateMapping, ...]: ...


def pop_states(
    node: Node, *filters: Filter
) -> Union[GraphStateMapping, Tuple[GraphStateMapping, ...]]:
  """Pop one or more :class:`State` types from the graph node.

  Example usage::

    >>> from flax import nnx
    >>> import jax.numpy as jnp

    >>> class Model(nnx.Module):
    ...   def __init__(self, rngs):
    ...     self.linear1 = nnx.Linear(2, 3, rngs=rngs)
    ...     self.linear2 = nnx.Linear(3, 4, rngs=rngs)
    ...   def __call__(self, x):
    ...     x = self.linear1(x)
    ...     self.sow(nnx.Intermediate, 'i', x)
    ...     x = self.linear2(x)
    ...     return x

    >>> x = jnp.ones((1, 2))
    >>> model = Model(rngs=nnx.Rngs(0))
    >>> assert not hasattr(model, 'i')
    >>> y = model(x)
    >>> assert hasattr(model, 'i')

    >>> intermediates = nnx.pop_states(model, nnx.Intermediate)
    >>> assert intermediates['i'].value[0].shape == (1, 3)
    >>> assert not hasattr(model, 'i')

  Args:
    node: A graph node object.
    *filters: One or more :class:`State` objects to filter by.

  Returns:
    The popped :class:`NestedMapping` containing the :class:`State`
    objects that were filtered for.
  """
  if len(filters) == 0:
    raise ValueError('Expected at least one filter')

  id_to_index: dict[int, Index] = {}
  path_parts: PathParts = ()
  predicates = tuple(to_predicate(filter) for filter in filters)
  flatted_state_mappings: tuple[FlattedStateMapping[StateLeaf], ...] = tuple({} for _ in predicates)
  _graph_pop(node=node,
             id_to_index=id_to_index,
             path_parts=path_parts,
             flatted_state_mappings=flatted_state_mappings,
             predicates=predicates, )
  states = tuple(GraphStateMapping.from_flat(flat_state) for flat_state in flatted_state_mappings)

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
  if isinstance(states, NestedMapping):
    return (states,)
  assert len(states) > 0
  return states  # type: ignore[return-value]


@overload
def split(node: A, /) -> tuple[GraphDefinition[A], GraphStateMapping]: ...


@overload
def split(node: A, first: Filter, /) -> tuple[GraphDefinition[A], GraphStateMapping]: ...


@overload
def split(
    node: A, first: Filter, second: Filter, /, *filters: Filter,
) -> tuple[GraphDefinition[A], GraphStateMapping, Unpack[tuple[GraphStateMapping, ...]]]: ...


def split(
    node: A,
    *filters: Filter
) -> Tuple[GraphDefinition[A], GraphStateMapping, Unpack[Tuple[GraphStateMapping, ...]]]:
  """Split a graph node into a :class:`GraphDefinition` and one or more :class:`NestedMapping`s. NestedMapping is
  a ``Mapping`` from strings or integers to ``Variables``, Arrays or nested States. GraphDefinition
  contains all the static information needed to reconstruct a ``Module`` graph, it is analogous
  to JAX’s ``PyTreeDef``. :func:`split` is used in conjunction with :func:`merge` to switch
  seamlessly between stateful and stateless representations of the graph.

  Example usage::

    >>> from flax import nnx
    >>> import jax, jax.numpy as jnp
    ...
    >>> class Foo(nnx.Module):
    ...   def __init__(self, rngs):
    ...     self.batch_norm = nnx.BatchNorm(2, rngs=rngs)
    ...     self.linear = nnx.Linear(2, 3, rngs=rngs)
    ...
    >>> node = Foo(nnx.Rngs(0))
    >>> graphdef, params, batch_stats = nnx.split(node, nnx.Param, nnx.BatchStat)
    ...
    >>> jax.tree.map(jnp.shape, params)
    NestedMapping({
      'batch_norm': {
        'bias': StateRef(
          type=Param,
          value=(2,)
        ),
        'scale': StateRef(
          type=Param,
          value=(2,)
        )
      },
      'linear': {
        'bias': StateRef(
          type=Param,
          value=(3,)
        ),
        'kernel': StateRef(
          type=Param,
          value=(2, 3)
        )
      }
    })
    >>> jax.tree.map(jnp.shape, batch_stats)
    NestedMapping({
      'batch_norm': {
        'mean': StateRef(
          type=BatchStat,
          value=(2,)
        ),
        'var': StateRef(
          type=BatchStat,
          value=(2,)
        )
      }
    })

  :func:`split` and :func:`merge` are primarily used to interact directly with JAX
  transformations, see
  `Functional API <https://flax.readthedocs.io/en/latest/nnx/nnx_basics.html#the-functional-api>`__
  for more information.

  Arguments:
    node: graph node to split.
    *filters: some optional filters to group the state into mutually exclusive substates.
  Returns:
    ``GraphDefinition`` and one or more ``States`` equal to the number of filters passed. If no
    filters are passed, a single ``NestedMapping`` is returned.
  """
  graphdef, state = flatten(node)
  states = _split_state(state, filters)
  return graphdef, *states


def merge(
    graphdef: GraphDefinition[A],
    state_mapping: GraphStateMapping,
    /,
    *state_mappings: GraphStateMapping,
) -> A:
  """The inverse of :func:`split`.

  ``merge`` takes a :class:`GraphDefinition` and one or more :class:`NestedMapping`'s and creates
  a new node with the same structure as the original node.

  Example usage::

    >>> from flax import nnx
    >>> import jax, jax.numpy as jnp
    ...
    >>> class Foo(nnx.Module):
    ...   def __init__(self, rngs):
    ...     self.batch_norm = nnx.BatchNorm(2, rngs=rngs)
    ...     self.linear = nnx.Linear(2, 3, rngs=rngs)
    ...
    >>> node = Foo(nnx.Rngs(0))
    >>> graphdef, params, batch_stats = nnx.split(node, nnx.Param, nnx.BatchStat)
    ...
    >>> new_node = nnx.merge(graphdef, params, batch_stats)
    >>> assert isinstance(new_node, Foo)
    >>> assert isinstance(new_node.batch_norm, nnx.BatchNorm)
    >>> assert isinstance(new_node.linear, nnx.Linear)

  :func:`split` and :func:`merge` are primarily used to interact directly with JAX
  transformations, see
  `Functional API <https://flax.readthedocs.io/en/latest/nnx/nnx_basics.html#the-functional-api>`__
  for more information.

  Args:
    graphdef: A :class:`GraphDefinition` object.
    state_mapping: A :class:`NestedMapping` object.
    *state_mappings: Additional :class:`NestedMapping` objects.

  Returns:
    The merged :class:`Module`.
  """
  state_mapping = GraphStateMapping.merge(state_mapping, *state_mappings)
  node = unflatten(graphdef, state_mapping)
  return node


def _graph_update_dynamic(node: Any, state: Mapping[Key, Any]):
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
        value = value.copy()  # TODO: chenge it to state_ref
      node_impl.set_key(node, key, value)
      continue

    # check values are of the same type
    current_value = node_dict[key]

    # case 2: subgraph is being updated
    if _is_node(current_value):
      if _is_state_leaf(value):
        raise ValueError(f'Expected a subgraph for {key!r}, but got: {value!r}')
      _graph_update_dynamic(current_value, value)
    elif isinstance(value, StateRef):
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
    node: Node,
    state_mapping: NestedMapping,
    /,
    *state_mappings: NestedMapping
) -> None:
  """
  Update the given graph node with a new :class:`NestedMapping` in-place.

  Example usage::

    >>> from flax import nnx
    >>> import jax, jax.numpy as jnp

    >>> x = jnp.ones((1, 2))
    >>> y = jnp.ones((1, 3))
    >>> model = nnx.Linear(2, 3, rngs=nnx.Rngs(0))

    >>> def loss_fn(model, x, y):
    ...   return jnp.mean((y - model(x))**2)
    >>> prev_loss = loss_fn(model, x, y)

    >>> grads = nnx.grad(loss_fn)(model, x, y)
    >>> new_state = jax.tree.map(lambda p, g: p - 0.1*g, state_refs(model), grads)
    >>> update_states(model, new_state)
    >>> assert loss_fn(model, x, y) < prev_loss

  Args:
    node: A graph node to update.
    state_mapping: A :class:`NestedMapping` object.
    *state_mappings: Additional :class:`NestedMapping` objects.
  """
  if state_mappings:
    state_mapping = GraphStateMapping.merge(state_mapping, *state_mappings)
  _graph_update_dynamic(node, state_mapping.raw_mapping)


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


@overload
def nodes(node, /, allowed_hierarchy=(0, _max_int)) -> FlattedMapping[Key, Node]:
  ...


@overload
def nodes(node, first: Filter, /, allowed_hierarchy=(0, _max_int)) -> FlattedMapping[Key, Node]:
  ...


@overload
def nodes(
    node, first: Filter, second: Filter, /, *filters: Filter, allowed_hierarchy=(0, _max_int)
) -> Tuple[FlattedMapping[Key, Node], ...]:
  ...


def nodes(
    node,
    *filters: Filter,
    allowed_hierarchy: Tuple[int, int] = (0, _max_int)
) -> Union[FlattedMapping[Key, Node], Tuple[FlattedMapping[Key, Node], ...]]:
  """
  Similar to :func:`split` but only returns the :class:`NestedMapping`'s indicated by the filters.
  """
  num_filters = len(filters)
  if num_filters == 0:
    filters = (..., ...)
  else:
    filters = (*filters, ...)

  nodes_iterable = iter_node(node, allowed_hierarchy=allowed_hierarchy)
  flat_nodes = _split_flatted(nodes_iterable, (*filters, ...))
  node_maps = tuple(FlattedMapping(flat_node) for flat_node in flat_nodes)
  if num_filters < 2:
    return node_maps[0]
  return node_maps


def _states_generator(node, allowed_hierarchy) -> Iterable[Tuple[PathParts, State]]:
  for path, value in iter_leaf(node, allowed_hierarchy=allowed_hierarchy):
    if isinstance(value, State):
      yield path, value


@overload
def states(node, /, allowed_hierarchy=(0, _max_int)) -> FlattedMapping[Key, State]:
  ...


@overload
def states(node, first: Filter, /, allowed_hierarchy=(0, _max_int)) -> FlattedMapping[Key, State]:
  ...


@overload
def states(
    node, first: Filter, second: Filter, /, *filters: Filter, allowed_hierarchy=(0, _max_int)
) -> tuple[FlattedMapping[Key, State], ...]:
  ...


def states(
    node,
    *filters: Filter,
    allowed_hierarchy: Tuple[int, int] = (0, _max_int)
) -> Union[FlattedMapping[Key, State], tuple[FlattedMapping[Key, State], ...]]:
  """
  Similar to :func:`split` but only returns the :class:`NestedMapping`'s indicated by the filters.
  """
  num_filters = len(filters)
  if num_filters == 0:
    filters = (..., ...)
  else:
    filters = (*filters, ...)

  states_iterable = _states_generator(node, allowed_hierarchy=allowed_hierarchy)
  flat_states = _split_flatted(states_iterable, (*filters, ...))
  state_maps = tuple(FlattedMapping(flat_state) for flat_state in flat_states)
  if num_filters < 2:
    return state_maps[0]
  return state_maps


@overload
def state_refs(
    node, /, flatted: bool = False
) -> NestedMapping[Key, StateRef]:
  ...


@overload
def state_refs(
    node, first: Filter, /, flatted: bool = False
) -> NestedMapping[Key, StateRef]:
  ...


@overload
def state_refs(
    node, first: Filter, second: Filter, /, *filters: Filter, flatted: bool = False
) -> Tuple[NestedMapping[Key, StateRef], ...]:
  ...


def state_refs(
    node, *filters, flatted: bool = False
) -> NestedMapping[Key, StateRef] | tuple[NestedMapping[Key, StateRef], ...]:
  """
  Similar to :func:`split` but only returns the :class:`NestedMapping`'s indicated by the filters.

  Example usage::

    >>> import brainstate as bst
    >>> class Model(bst.nn.Module):
    ...   def __init__(self):
    ...     super().__init__()
    ...     self.l1 = bst.nn.Linear(2, 3)
    ...     self.l2 = bst.nn.Linear(3, 4)
    ...   def __call__(self, x):
    ...     return self.l2(self.l1(x))

    >>> model = Model()
    >>> # get the learnable parameters from the batch norm and linear layer
    >>> params = bst.graph.state_refs(model, bst.ParamState)
    >>> # get them separately
    >>> params, others = bst.graph.state_refs(model, bst.ParamState, bst.ShortTermState)
    >>> # get them together
    >>> states = bst.graph.state_refs(model)

  Args:
    node: A graph node object.
    *filters: One or more :class:`State` objects to filter by.
    flatted: If True, the path will be flattened. If not, the path will be nested.

  Returns:
    One or more :class:`NestedMapping` mappings.
  """
  _, state_mapping = flatten(node)
  state_mappings: GraphStateMapping | tuple[GraphStateMapping, ...]
  if len(filters) == 0:
    state_mappings = state_mapping
  elif len(filters) == 1:
    state_mappings = state_mapping.filter(filters[0])
  else:
    state_mappings = state_mapping.filter(filters[0], filters[1], *filters[2:])
  return state_mappings


def graphdef(node: Any, /) -> GraphDefinition[Any]:
  """Get the :class:`GraphDefinition` of the given graph node.

  Example usage::

    >>> from flax import nnx

    >>> model = nnx.Linear(2, 3, rngs=nnx.Rngs(0))
    >>> graphdef, _ = nnx.split(model)
    >>> assert graphdef == nnx.graphdef(model)

  Args:
    node: A graph node object.
  Returns:
    The :class:`GraphDefinition` of the :class:`Module` object.
  """
  graphdef, _ = flatten(node)
  return graphdef


def clone(node: Node) -> Node:
  """
  Create a deep copy of the given graph node.

  Example usage::

    >>> model = nnx.Linear(2, 3, rngs=nnx.Rngs(0))
    >>> cloned_model = clone(model)
    >>> model.bias.value += 1
    >>> assert (model.bias.value != cloned_model.bias.value).all()

  Args:
    node: A graph node object.

  Returns:
    A deep copy of the :class:`Module` object.
  """
  graphdef, state = split(node)
  return merge(graphdef, state)


def call(
    graphdef_state: Tuple[GraphDefinition[A], GraphStateMapping],
) -> ApplyCaller[Tuple[GraphDefinition[A], GraphStateMapping]]:
  """Calls a method underlying graph node defined by a (GraphDefinition, NestedMapping) pair.

  ``call`` takes a ``(GraphDefinition, NestedMapping)`` pair and creates a proxy object that can be
  used to call methods on the underlying graph node. When a method is called, the
  output is returned along with a new (GraphDefinition, NestedMapping) pair that represents the
  updated state of the graph node. ``call`` is equivalent to :func:`merge` > ``method``
  > :func:`split`` but is more convenient to use in pure JAX functions.

  Example::

    >>> from flax import nnx
    >>> import jax
    >>> import jax.numpy as jnp
    ...
    >>> class StatefulLinear(nnx.Module):
    ...   def __init__(self, din, dout, rngs):
    ...     self.w = nnx.Param(jax.random.uniform(rngs(), (din, dout)))
    ...     self.b = nnx.Param(jnp.zeros((dout,)))
    ...     self.count = nnx.State(jnp.array(0, dtype=jnp.uint32))
    ...
    ...   def increment(self):
    ...     self.count += 1
    ...
    ...   def __call__(self, x):
    ...     self.increment()
    ...     return x @ self.w + self.b
    ...
    >>> linear = StatefulLinear(3, 2, nnx.Rngs(0))
    >>> linear_state = split(linear)
    ...
    >>> @jax.jit
    ... def forward(x, linear_state):
    ...   y, linear_state = nnx.call(linear_state)(x)
    ...   return y, linear_state
    ...
    >>> x = jnp.ones((1, 3))
    >>> y, linear_state = forward(x, linear_state)
    >>> y, linear_state = forward(x, linear_state)
    ...
    >>> linear = merge(*linear_state)
    >>> linear.count.value
    Array(2, dtype=uint32)

  The proxy object returned by ``call`` supports indexing and attribute access
  to access nested methods. In the example below, the ``increment`` method indexing
  is used to call the ``increment`` method of the ``StatefulLinear`` module
  at the ``b`` key of a ``nodes`` dictionary.

    >>> class StatefulLinear(nnx.Module):
    ...   def __init__(self, din, dout, rngs):
    ...     self.w = nnx.Param(jax.random.uniform(rngs(), (din, dout)))
    ...     self.b = nnx.Param(jnp.zeros((dout,)))
    ...     self.count = nnx.State(jnp.array(0, dtype=jnp.uint32))
    ...
    ...   def increment(self):
    ...     self.count += 1
    ...
    ...   def __call__(self, x):
    ...     self.increment()
    ...     return x @ self.w + self.b
    ...
    >>> rngs = nnx.Rngs(0)
    >>> nodes = dict(
    ...   a=StatefulLinear(3, 2, rngs),
    ...   b=StatefulLinear(2, 1, rngs),
    ... )
    ...
    >>> node_state = split(nodes)
    >>> # use attribute access
    >>> _, node_state = nnx.call(node_state)['b'].increment()
    ...
    >>> nodes = merge(*node_state)
    >>> nodes['a'].count.value
    Array(0, dtype=uint32)
    >>> nodes['b'].count.value
    Array(1, dtype=uint32)
  """

  def pure_caller(accessor: DelayedAccessor, *args, **kwargs):
    node = merge(*graphdef_state)
    method = accessor(node)
    out = method(*args, **kwargs)
    return out, split(node)

  return CallableProxy(pure_caller)  # type: ignore


def iter_leaf(
    node: Any,
    allowed_hierarchy: Tuple[int, int] = (0, _max_int)
) -> Iterator[tuple[PathParts, Any]]:
  """Iterates over all nested leaves in the given graph node, including the current node.

  ``iter_graph`` creates a generator that yields path and value pairs, where
  the path is a tuple of strings or integers representing the path to the value from the
  root. Repeated nodes are visited only once. Leaves include static values.

  Example::
    >>> import brainstate as bst
    >>> import jax.numpy as jnp
    ...
    >>> class Linear(bst.nn.Module):
    ...   def __init__(self, din, dout):
    ...     super().__init__()
    ...     self.weight = bst.ParamState(bst.random.randn(din, dout))
    ...     self.bias = bst.ParamState(bst.random.randn(dout))
    ...     self.a = 1
    ...
    >>> module = Linear(3, 4)
    ...
    >>> for path, value in bst.graph.iter_leaf([module, module]):
    ...   print(path, type(value).__name__)
    ...
    (0, 'a') int
    (0, 'bias') ParamState
    (0, 'weight') ParamState

  Parameters
  ----------
  node: Node
    The node to iterate over.
  allowed_hierarchy: tuple of int
    The allowed hierarchy.

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
        yield from _iter_graph_leaf(value,
                                    visited_,
                                    (*path_parts_, key),
                                    level_ + 1 if _is_graph_node(value) else level_)
    else:
      if level_ >= allowed_hierarchy[0]:
        yield path_parts_, node_

  visited: set[int] = set()
  path_parts: PathParts = ()
  level: int = 0
  yield from _iter_graph_leaf(node, visited, path_parts, level)


def iter_node(
    node: Any,
    allowed_hierarchy: Tuple[int, int] = (0, _max_int)
) -> Iterator[Tuple[PathParts, Any]]:
  """
  Iterates over all nested nodes of the given graph node, including the current node.

  ``iter_graph`` creates a generator that yields path and value pairs, where
  the path is a tuple of strings or integers representing the path to the value from the
  root. Repeated nodes are visited only once. Leaves include static values.

  Example::
    >>> import brainstate as bst
    >>> import jax.numpy as jnp
    ...
    >>> class Model(bst.nn.Module):
    ...   def __init__(self):
    ...     super().__init__()
    ...     self.a = bst.nn.Linear(1, 2)
    ...     self.b = bst.nn.Linear(2, 3)
    ...     self.c = [bst.nn.Linear(3, 4), bst.nn.Linear(4, 5)]
    ...     self.d = {'x': bst.nn.Linear(5, 6), 'y': bst.nn.Linear(6, 7)}
    ...     self.b.a = bst.nn.LIF(2)
    ...
    >>> model = Model()
    ...
    >>> for path, node in bst.graph.iter_node([model, model]):
    ...    print(path, node.__class__.__name__)
    ...
    (0, 'a') Linear
    (0, 'b', 'a') LIF
    (0, 'b') Linear
    (0, 'c', 0) Linear
    (0, 'c', 1) Linear
    (0, 'd', 'x') Linear
    (0, 'd', 'y') Linear
    (0,) Model

  Parameters
  ----------
  node: Node
    The node to iterate over.
  allowed_hierarchy: tuple of int
    The allowed hierarchy.

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
  """An empty pytree node that treats its inner value as static.
  ``value`` must define ``__eq__`` and ``__hash__``.
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


def _flatten_pytree(pytree: Any):
  leaves, treedef = jax.tree_util.tree_flatten_with_path(pytree, is_leaf=lambda x: x is not pytree)
  nodes = tuple((_key_path_to_key(path[0]), value) for path, value in leaves)
  return nodes, treedef


def _unflatten_pytree(
    nodes: tuple[tuple[Key, Any], ...],
    treedef: jax.tree_util.PyTreeDef
):
  pytree = treedef.unflatten(value for _, value in nodes)
  return pytree


PYTREE_NODE_IMPL = PyTreeNodeImpl(type=PytreeType, flatten=_flatten_pytree, unflatten=_unflatten_pytree)
