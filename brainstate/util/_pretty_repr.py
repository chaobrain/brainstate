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

import dataclasses
import functools
import threading
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Iterator, Mapping, TypeVar, Union, Callable, Optional

__all__ = [
  'PrettyType',
  'PrettyAttr',
  'PrettyRepr',
  'PrettyMapping',
  'MappingReprMixin',
]

A = TypeVar('A')
B = TypeVar('B')


@dataclasses.dataclass
class PrettyType:
  """
  Configuration for pretty representation of objects.
  """
  type: Union[str, type]
  start: str = '('
  end: str = ')'
  value_sep: str = '='
  elem_indent: str = '  '
  empty_repr: str = ''


@dataclasses.dataclass
class PrettyAttr:
  key: str
  value: Union[str, Any]
  start: str = ''
  end: str = ''


class PrettyRepr(ABC):
  """
  Interface for pretty representation of objects.

  Example:
  ```
  class MyObject(PrettyRepr):
    def __pretty_repr__(self):
      yield PrettyType(type='MyObject', start='{', end='}')
      yield PrettyAttr('key', self.key)
      yield PrettyAttr('value', self.value)
  ```

  """
  __slots__ = ()

  @abstractmethod
  def __pretty_repr__(self) -> Iterator[Union[PrettyType, PrettyAttr]]:
    raise NotImplementedError

  def __repr__(self) -> str:
    # repr the individual object with the pretty representation
    return get_repr(self)


def _repr_elem(obj: PrettyType, elem: Any) -> str:
  if not isinstance(elem, PrettyAttr):
    raise TypeError(f'Item must be Elem, got {type(elem).__name__}')

  value = elem.value if isinstance(elem.value, str) else repr(elem.value)
  value = value.replace('\n', '\n' + obj.elem_indent)

  return f'{obj.elem_indent}{elem.start}{elem.key}{obj.value_sep}{value}{elem.end}'


def get_repr(obj: PrettyRepr) -> str:
  if not isinstance(obj, PrettyRepr):
    raise TypeError(f'Object {obj!r} is not representable')

  iterator = obj.__pretty_repr__()
  obj_repr = next(iterator)

  # repr object
  if not isinstance(obj_repr, PrettyType):
    raise TypeError(f'First item must be Config, got {type(obj_repr).__name__}')

  # repr attributes
  elems = ',\n'.join(map(partial(_repr_elem, obj_repr), iterator))
  if elems:
    elems = '\n' + elems + '\n'
  else:
    elems = obj_repr.empty_repr

  # repr object type
  type_repr = obj_repr.type if isinstance(obj_repr.type, str) else obj_repr.type.__name__

  # return repr
  return f'{type_repr}{obj_repr.start}{elems}{obj_repr.end}'


class MappingReprMixin(Mapping[A, B]):
  def __pretty_repr__(self):
    yield PrettyType(type='', value_sep=': ', start='{', end='}')

    for key, value in self.items():
      yield PrettyAttr(repr(key), value)


@dataclasses.dataclass(repr=False)
class PrettyMapping(PrettyRepr):
  """
  Pretty representation of a mapping.
  """
  mapping: Mapping

  def __pretty_repr__(self):
    yield PrettyType(type='', value_sep=': ', start='{', end='}')

    for key, value in self.mapping.items():
      yield PrettyAttr(repr(key), value)


@dataclasses.dataclass
class GraphUtilsContext(threading.local):
  seen_modules_repr: set[int] | None = None


CONTEXT = GraphUtilsContext()


def _default_repr_object(node):
  yield PrettyType(type=type(node))


def _default_repr_attr(node):
  for name, value in vars(node).items():
    if name.startswith('_'):
      continue
    yield PrettyAttr(name, repr(value))


def pretty_repr_avoid_duplicate(
    node,
    repr_object: Optional[Callable] = None,
    repr_attr: Optional[Callable] = None
):
  """
  Pretty representation of an object avoiding duplicate representations.
  """
  if repr_object is None:
    repr_object = _default_repr_object
  if repr_attr is None:
    repr_attr = _default_repr_attr

  if CONTEXT.seen_modules_repr is None:
    CONTEXT.seen_modules_repr = set()
    clear_seen = True
  else:
    clear_seen = False

  # Avoid infinite recursion
  if id(node) in CONTEXT.seen_modules_repr:
    yield PrettyType(type=type(node), empty_repr='...')
    return

  # repr object
  yield from repr_object(node)

  # Add to seen modules
  CONTEXT.seen_modules_repr.add(id(node))

  try:
    # repr attributes
    yield from repr_attr(node)

  finally:
    if clear_seen:
      CONTEXT.seen_modules_repr = None
