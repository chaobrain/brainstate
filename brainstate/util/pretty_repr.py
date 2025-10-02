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

"""
Pretty representation utilities for creating human-readable string representations.

This module provides utilities for creating customizable pretty representations of
objects, with support for nested structures and circular reference detection.
"""

import dataclasses
import threading
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Iterator, Mapping, TypeVar, Union, Callable, Optional

__all__ = [
    'yield_unique_pretty_repr_items',
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

    Attributes
    ----------
    type : Union[str, type]
        The type name or type object to display.
    start : str, default='('
        The opening delimiter for the representation.
    end : str, default=')'
        The closing delimiter for the representation.
    value_sep : str, default='='
        The separator between keys and values.
    elem_indent : str, default='  '
        The indentation for nested elements.
    empty_repr : str, default=''
        The representation for empty objects.
    """
    type: Union[str, type]
    start: str = '('
    end: str = ')'
    value_sep: str = '='
    elem_indent: str = '  '
    empty_repr: str = ''


@dataclasses.dataclass
class PrettyAttr:
    """
    Configuration for pretty representation of attributes.

    Attributes
    ----------
    key : str
        The attribute name or key.
    value : Union[str, Any]
        The attribute value.
    start : str, default=''
        Optional prefix for the attribute.
    end : str, default=''
        Optional suffix for the attribute.
    """
    key: str
    value: Union[str, Any]
    start: str = ''
    end: str = ''


class PrettyRepr(ABC):
    """
    Interface for pretty representation of objects.

    This abstract base class provides a framework for creating custom
    pretty representations of objects by yielding PrettyType and PrettyAttr
    instances.

    Examples
    --------
    .. code-block:: python

        >>> class MyObject(PrettyRepr):
        ...     def __init__(self, key, value):
        ...         self.key = key
        ...         self.value = value
        ...
        ...     def __pretty_repr__(self):
        ...         yield PrettyType(type='MyObject', start='{', end='}')
        ...         yield PrettyAttr('key', self.key)
        ...         yield PrettyAttr('value', self.value)
        ...
        >>> obj = MyObject('foo', 42)
        >>> print(obj)
        MyObject{
          key=foo,
          value=42
        }
    """
    __slots__ = ()

    @abstractmethod
    def __pretty_repr__(self) -> Iterator[Union[PrettyType, PrettyAttr]]:
        """
        Generate the pretty representation of the object.

        Yields
        ------
        Union[PrettyType, PrettyAttr]
            First yield should be PrettyType, followed by PrettyAttr instances.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        """
        Generate string representation using pretty representation.

        Returns
        -------
        str
            The formatted string representation of the object.
        """
        return pretty_repr_object(self)


def pretty_repr_elem(obj: PrettyType, elem: Any) -> str:
    """
    Constructs a string representation of a single element within a pretty representation.

    This function takes a `PrettyType` object and an element, which must be an instance
    of `PrettyAttr`, and generates a formatted string that represents the element. The
    formatting is based on the configuration provided by the `PrettyType` object.

    Parameters
    ----------
    obj : PrettyType
        The configuration object that defines how the element should be formatted.
        It includes details such as indentation, separators, and surrounding characters.
    elem : Any
        The element to be represented. It must be an instance of `PrettyAttr`, which
        contains the key and value to be formatted.

    Returns
    -------
    str
        A string that represents the element in a formatted manner, adhering to the
        configuration specified by the `PrettyType` object.

    Raises
    ------
    TypeError
        If the provided element is not an instance of `PrettyAttr`.
    """
    if not isinstance(elem, PrettyAttr):
        raise TypeError(f'Item must be Elem, got {type(elem).__name__}')

    value = elem.value if isinstance(elem.value, str) else repr(elem.value)
    value = value.replace('\n', '\n' + obj.elem_indent)

    return f'{obj.elem_indent}{elem.start}{elem.key}{obj.value_sep}{value}{elem.end}'


def pretty_repr_object(obj: PrettyRepr) -> str:
    """
    Generates a pretty string representation of an object that implements the PrettyRepr interface.

    This function utilizes the __pretty_repr__ method of the PrettyRepr interface to obtain
    a structured representation of the object, which includes both the type and attributes
    of the object in a human-readable format.

    Parameters
    ----------
    obj : PrettyRepr
        The object for which the pretty representation is to be generated. The object must
        implement the PrettyRepr interface.

    Returns
    -------
    str
        A string that represents the object in a pretty format, including its type and attributes.
        The format is determined by the PrettyType and PrettyAttr instances yielded by the
        __pretty_repr__ method of the object.

    Raises
    ------
    TypeError
        If the provided object does not implement the PrettyRepr interface or if the first item
        yielded by the __pretty_repr__ method is not an instance of PrettyType.
    """
    if not isinstance(obj, PrettyRepr):
        raise TypeError(f'Object {obj!r} is not representable')

    iterator = obj.__pretty_repr__()
    obj_repr = next(iterator)

    # repr object
    if not isinstance(obj_repr, PrettyType):
        raise TypeError(f'First item must be PrettyType, got {type(obj_repr).__name__}')

    # repr attributes
    elem_reprs = tuple(map(partial(pretty_repr_elem, obj_repr), iterator))
    elems = ',\n'.join(elem_reprs)
    if elems:
        elems = '\n' + elems + '\n'
    else:
        elems = obj_repr.empty_repr

    # repr object type
    type_repr = obj_repr.type if isinstance(obj_repr.type, str) else obj_repr.type.__name__

    # return repr
    return f'{type_repr}{obj_repr.start}{elems}{obj_repr.end}'


class MappingReprMixin(Mapping[A, B]):
    """
    Mapping mixin for pretty representation.

    This mixin provides a default pretty representation for mapping-like objects.

    Examples
    --------
    .. code-block:: python

        >>> class MyMapping(dict, MappingReprMixin):
        ...     pass
        ...
        >>> m = MyMapping({'a': 1, 'b': 2})
        >>> print(m)
        {
          'a': 1,
          'b': 2
        }
    """

    def __pretty_repr__(self) -> Iterator[Union[PrettyType, PrettyAttr]]:
        """
        Generate pretty representation for mapping.

        Yields
        ------
        Union[PrettyType, PrettyAttr]
            PrettyType followed by PrettyAttr for each key-value pair.
        """
        yield PrettyType(type='', value_sep=': ', start='{', end='}')

        for key, value in self.items():
            yield PrettyAttr(repr(key), value)


@dataclasses.dataclass(repr=False)
class PrettyMapping(PrettyRepr):
    """
    Pretty representation of a mapping.

    Attributes
    ----------
    mapping : Mapping
        The mapping to represent.
    type_name : str, default=''
        Optional type name to display.

    Examples
    --------
    .. code-block:: python

        >>> m = PrettyMapping({'a': 1, 'b': 2}, type_name='MyDict')
        >>> print(m)
        MyDict{
          'a': 1,
          'b': 2
        }
    """
    mapping: Mapping
    type_name: str = ''

    def __pretty_repr__(self) -> Iterator[Union[PrettyType, PrettyAttr]]:
        """
        Generate pretty representation for the mapping.

        Yields
        ------
        Union[PrettyType, PrettyAttr]
            PrettyType followed by PrettyAttr for each key-value pair.
        """
        yield PrettyType(type=self.type_name, value_sep=': ', start='{', end='}')

        for key, value in self.mapping.items():
            yield PrettyAttr(repr(key), value)


@dataclasses.dataclass
class PrettyReprContext(threading.local):
    """
    A thread-local context for managing the state of pretty representation.

    This class is used to keep track of objects that have been seen during
    the generation of pretty representations, preventing infinite recursion
    in cases of circular references.

    Attributes
    ----------
    seen_modules_repr : dict[int, Any] | None
        A dictionary mapping object IDs to objects that have been seen
        during the pretty representation process. This is used to avoid
        representing the same object multiple times.
    """
    seen_modules_repr: dict[int, Any] | None = None


CONTEXT = PrettyReprContext()


def _default_repr_object(node: Any) -> Iterator[PrettyType]:
    """
    Generate a default pretty representation for an object.

    This function yields a `PrettyType` instance that represents the type
    of the given object. It is used as a default method for representing
    objects when no custom representation function is provided.

    Parameters
    ----------
    node : Any
        The object for which the pretty representation is to be generated.

    Yields
    ------
    PrettyType
        An instance of `PrettyType` that contains the type information of
        the object.
    """
    yield PrettyType(type=type(node))


def _default_repr_attr(node: Any) -> Iterator[PrettyAttr]:
    """
    Generate a default pretty representation for the attributes of an object.

    This function iterates over the attributes of the given object and yields
    a `PrettyAttr` instance for each attribute that does not start with an
    underscore. The `PrettyAttr` instances contain the attribute name and its
    string representation.

    Parameters
    ----------
    node : Any
        The object whose attributes are to be represented.

    Yields
    ------
    PrettyAttr
        An instance of `PrettyAttr` for each non-private attribute of the object,
        containing the attribute name and its string representation.
    """
    for name, value in vars(node).items():
        if name.startswith('_'):
            continue
        yield PrettyAttr(name, repr(value))


def yield_unique_pretty_repr_items(
    node: Any,
    repr_object: Optional[Callable] = None,
    repr_attr: Optional[Callable] = None
) -> Iterator[Union[PrettyType, PrettyAttr]]:
    """
    Generate a pretty representation of an object while avoiding duplicate representations.

    This function yields a structured representation of an object, using custom or default
    methods for representing the object itself and its attributes. It ensures that each
    object is only represented once to prevent infinite recursion in cases of circular
    references.

    Parameters
    ----------
    node : Any
        The object to be represented.
    repr_object : Optional[Callable], optional
        A callable that yields the representation of the object itself.
        If not provided, a default representation function is used.
    repr_attr : Optional[Callable], optional
        A callable that yields the representation of the object's attributes.
        If not provided, a default attribute representation function is used.

    Yields
    ------
    Union[PrettyType, PrettyAttr]
        The pretty representation of the object and its attributes,
        avoiding duplicates by tracking seen objects.

    Examples
    --------
    .. code-block:: python

        >>> class Node:
        ...     def __init__(self, value, next=None):
        ...         self.value = value
        ...         self.next = next
        ...
        >>> # Create circular reference
        >>> node1 = Node(1)
        >>> node2 = Node(2, node1)
        >>> node1.next = node2
        ...
        >>> # This will handle circular reference gracefully
        >>> for item in yield_unique_pretty_repr_items(node1):
        ...     print(item)
    """
    if repr_object is None:
        repr_object = _default_repr_object
    if repr_attr is None:
        repr_attr = _default_repr_attr

    if CONTEXT.seen_modules_repr is None:
        # CONTEXT.seen_modules_repr = set()
        CONTEXT.seen_modules_repr = dict()
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
    # CONTEXT.seen_modules_repr.add(id(node))
    CONTEXT.seen_modules_repr[id(node)] = node

    try:
        # repr attributes
        yield from repr_attr(node)
    finally:
        if clear_seen:
            CONTEXT.seen_modules_repr = None
