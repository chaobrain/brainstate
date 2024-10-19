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

from abc import ABCMeta
from copy import deepcopy
from typing import Any, Callable, Type, TypeVar, Tuple, TYPE_CHECKING, Mapping, Iterator, Sequence, List as ListType

import brainunit as u
import jax
import numpy as np

from brainstate._state import State, StateRef
from brainstate.typing import Key
from brainstate.util._error import TraceContextError
from brainstate.util._pretty_repr import PrettyRepr, pretty_repr_avoid_duplicate, PrettyType, PrettyAttr
from brainstate.util._tracers import StateJaxTracer
from ._graph_operation import register_graph_node_type

__all__ = [
  'Node', 'Dict', 'List', 'Sequential',
]

G = TypeVar('G', bound='Node')
A = TypeVar('A')


class GraphNodeMeta(ABCMeta):
  if not TYPE_CHECKING:
    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
      node = cls.__new__(cls, *args, **kwargs)
      vars(node)['_trace_state'] = StateJaxTracer()
      node.__init__(*args, **kwargs)
      return node


class Node(PrettyRepr, metaclass=GraphNodeMeta):
  """
  Base class for all graph nodes.

  This class provides the following functionalities:
  - Register the node type with the graph tool.
  - Prevent mutation of the node from different trace level.
  - Provide a pretty repr for the node.
  - Provide a treescope repr for the node.
  - Deepcopy the node.

  """
  if TYPE_CHECKING:
    _trace_state: StateJaxTracer

  def __init_subclass__(cls) -> None:
    super().__init_subclass__()

    register_graph_node_type(
      type=cls,
      flatten=_node_flatten,
      set_key=_node_set_key,
      pop_key=_node_pop_key,
      create_empty=_node_create_empty,
      clear=_node_clear,
    )

  # if not TYPE_CHECKING:
  #   def __setattr__(self, name: str, value: Any) -> None:
  #     self._setattr(name, value)

  # def _setattr(self, name: str, value: Any) -> None:
  #   self.check_valid_context(lambda: f"Cannot mutate '{type(self).__name__}' from different trace level")
  #   object.__setattr__(self, name, value)

  def check_valid_context(self, error_msg: Callable[[], str]) -> None:
    """
    Check if the current context is valid for the object to be mutated.
    """
    if not self._trace_state.is_valid():
      raise TraceContextError(error_msg())

  def __deepcopy__(self: G, memo=None) -> G:
    """
    Deepcopy the object.
    """
    from ._graph_operation import split, merge

    graphdef, state = split(self)
    graphdef = deepcopy(graphdef)
    state = deepcopy(state)
    return merge(graphdef, state)

  def __pretty_repr__(self):
    """
    Pretty repr for the object.
    """
    yield from pretty_repr_avoid_duplicate(self, _default_repr_object, _default_repr_attr)

  def __treescope_repr__(self, path, subtree_renderer):
    """
    Treescope repr for the object.
    """
    children = {}
    for name, value in vars(self).items():
      name, value = self.__leaf_fn__(name, value)
      if name.startswith('_'):
        continue
      children[name] = value
    import treescope  # type: ignore[import-not-found,import-untyped]
    return treescope.repr_lib.render_object_constructor(
      object_type=type(self),
      attributes=children,
      path=path,
      subtree_renderer=subtree_renderer,
      color=treescope.formatting_util.color_from_string(type(self).__qualname__)
    )

  def __leaf_fn__(self, leaf, value):
    return leaf, value


def _default_repr_object(node: Node):
  yield PrettyType(type=type(node))


def _default_repr_attr(node: Node):
  for name, value in vars(node).items():
    name, value = node.__leaf_fn__(name, value)
    if name.startswith('_'):
      continue
    value = jax.tree.map(_to_shape_dtype, value, is_leaf=lambda x: isinstance(x, u.Quantity))
    yield PrettyAttr(name, repr(value))


def _to_shape_dtype(value):
  if isinstance(value, State):
    return value.replace(raw_value=jax.tree.map(_to_shape_dtype, value.value))
  elif isinstance(value, (np.ndarray, jax.Array)):
    return f'Array(shape={value.shape}, dtype={value.dtype.name})'
  elif isinstance(value, u.Quantity):
    return f'Quantity(mantissa=Array(shape={value.shape}, dtype={value.dtype.name}), unit={value.unit})'
  return value


# -------------------------------
# Graph Definition
# -------------------------------


def _node_flatten(
    node: Node
) -> Tuple[Tuple[Tuple[str, Any], ...], Tuple[Type]]:
  nodes = sorted((key, value) for key, value in vars(node).items() if key != '_trace_state')
  return nodes, (type(node),)


def _node_set_key(
    node: Node,
    key: Key,
    value: Any
) -> None:
  if not isinstance(key, str):
    raise KeyError(f'Invalid key: {key!r}')
  elif (
      hasattr(node, key)
      and isinstance(state := getattr(node, key), State)
      and isinstance(value, StateRef)
  ):
    state.update_from_ref(value)
  else:
    setattr(node, key, value)


def _node_pop_key(
    node: Node,
    key: Key
):
  if not isinstance(key, str):
    raise KeyError(f'Invalid key: {key!r}')
  return vars(node).pop(key)


def _node_create_empty(
    static: tuple[Type[G],]
) -> G:
  node_type, = static
  node = object.__new__(node_type)
  vars(node).update(_trace_state=StateJaxTracer())
  return node


def _node_clear(node: Node):
  module_state = node._trace_state
  module_vars = vars(node)
  module_vars.clear()
  module_vars['_trace_state'] = module_state


class Dict(Node, Mapping[str, A]):
  """
  A dictionary node.
  """

  def __init__(self, *args, **kwargs):
    for name, value in dict(*args, **kwargs).items():
      setattr(self, name, value)

  def __getitem__(self, key) -> A:
    return getattr(self, key)

  def __setitem__(self, key, value):
    setattr(self, key, value)

  def __getattr__(self, key) -> A:
    return super().__getattribute__(key)

  def __setattr__(self, key, value):
    super().__setattr__(key, value)

  def __iter__(self) -> Iterator[str]:
    return (k for k in vars(self) if k != '_object__state')

  def __len__(self) -> int:
    return len(vars(self))


class List(Node):
  """
  A list node.
  """

  def __init__(self, seq=()):
    vars(self).update({str(i): item for i, item in enumerate(seq)})

  def __getitem__(self, idx):
    return getattr(self, str(idx))

  def __setitem__(self, idx, value):
    setattr(self, str(idx), value)

  def __iter__(self):
    return iter(vars(self).values())

  def __len__(self):
    return len(vars(self))

  def __add__(self, other: Sequence[A]) -> List[A]:
    return List(list(self) + list(other))

  def append(self, value):
    self[len(vars(self))] = value

  def extend(self, values):
    for value in values:
      self.append(value)


class Sequential(Node):
  def __init__(self, *fns: Callable[..., Any]):
    self.layers = list(fns)

  def __call__(self, *args, **kwargs) -> Any:
    output: Any = None

    for i, f in enumerate(self.layers):
      if not callable(f):
        raise TypeError(f'Sequence[{i}] is not callable: {f}')
      if i > 0:
        if isinstance(output, tuple):
          args = output
          kwargs = {}
        elif isinstance(output, dict):
          args = ()
          kwargs = output
        else:
          args = (output,)
          kwargs = {}
      output = f(*args, **kwargs)

    return output