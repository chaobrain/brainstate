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
Custom data structures that work seamlessly with JAX transformations.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, KeysView, ValuesView, ItemsView
from typing import Any, TypeVar, Generic, Iterator, overload

import jax
import jax.tree_util
from typing_extensions import dataclass_transform

__all__ = [
    'field',
    'dataclass',
    'PyTreeNode',
    'FrozenDict',
    'freeze',
    'unfreeze',
    'copy',
    'pop',
    'pretty_repr',
]

# Type variables
K = TypeVar('K')
V = TypeVar('V')
T = TypeVar('T')
TNode = TypeVar('TNode', bound='PyTreeNode')


def field(pytree_node: bool = True, **kwargs) -> dataclasses.Field:
    """
    Create a dataclass field with JAX pytree metadata.

    Parameters
    ----------
    pytree_node : bool, optional
        If True (default), this field will be treated as part of the pytree.
        If False, it will be treated as metadata and not be touched
        by JAX transformations.
    **kwargs
        Additional arguments to pass to dataclasses.field().

    Returns
    -------
    dataclasses.Field
        A dataclass field with the appropriate metadata.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainstate.util import dataclass, field

        >>> @dataclass
        ... class Model:
        ...     weights: jnp.ndarray
        ...     bias: jnp.ndarray
        ...     # This field won't be affected by JAX transformations
        ...     name: str = field(pytree_node=False, default="model")
    """
    metadata = kwargs.pop('metadata', {})
    metadata['pytree_node'] = pytree_node
    return dataclasses.field(metadata=metadata, **kwargs)


@dataclass_transform(field_specifiers=(field,))
def dataclass(cls: type[T], **kwargs) -> type[T]:
    """
    Create a dataclass that works with JAX transformations.

    This decorator creates immutable dataclasses that can be used safely
    with JAX transformations like jit, grad, vmap, etc. The created class
    will be registered as a JAX pytree node.

    Parameters
    ----------
    cls : type
        The class to decorate.
    **kwargs
        Additional arguments for dataclasses.dataclass().
        If 'frozen' is not specified, it defaults to True.

    Returns
    -------
    type
        The decorated class as an immutable JAX-compatible dataclass.

    See Also
    --------
    PyTreeNode : Base class for creating JAX-compatible pytree nodes.
    field : Create dataclass fields with pytree metadata.

    Notes
    -----
    The decorated class will be frozen (immutable) by default to ensure
    compatibility with JAX's functional programming paradigm.

    Examples
    --------
    .. code-block:: python

        >>> import jax
        >>> import jax.numpy as jnp
        >>> from brainstate.util import dataclass, field

        >>> @dataclass
        ... class Model:
        ...     weights: jax.Array
        ...     bias: jax.Array
        ...     name: str = field(pytree_node=False, default="model")

        >>> model = Model(weights=jnp.ones((3, 3)), bias=jnp.zeros(3))

        >>> # JAX transformations will only apply to weights and bias, not name
        >>> grad_fn = jax.grad(lambda m: jnp.sum(m.weights))
        >>> grads = grad_fn(model)

        >>> # Use replace to create modified copies
        >>> model2 = model.replace(weights=jnp.ones((3, 3)) * 2)
    """
    # Check if already converted
    if hasattr(cls, '_brainstate_dataclass'):
        return cls

    # Default to frozen for immutability
    kwargs.setdefault('frozen', True)

    # Apply standard dataclass decorator
    cls = dataclasses.dataclass(**kwargs)(cls)

    # Separate fields into pytree and metadata
    pytree_fields = []
    meta_fields = []

    for field_info in dataclasses.fields(cls):
        if field_info.metadata.get('pytree_node', True):
            pytree_fields.append(field_info.name)
        else:
            meta_fields.append(field_info.name)

    # Add replace method
    def replace(self: T, **updates) -> T:
        """Replace specified fields with new values."""
        return dataclasses.replace(self, **updates)

    cls.replace = replace

    # Register with JAX
    _register_pytree(cls, pytree_fields, meta_fields)

    # Mark as BrainState dataclass
    cls._brainstate_dataclass = True

    return cls


def _register_pytree(cls: type, pytree_fields: list[str], meta_fields: list[str]) -> None:
    """Register a class as a JAX pytree."""

    def flatten_fn(obj):
        pytree_data = tuple(getattr(obj, name) for name in pytree_fields)
        metadata = tuple(getattr(obj, name) for name in meta_fields)
        return pytree_data, metadata

    def flatten_with_keys_fn(obj):
        pytree_data = tuple(
            (jax.tree_util.GetAttrKey(name), getattr(obj, name))
            for name in pytree_fields
        )
        metadata = tuple(getattr(obj, name) for name in meta_fields)
        return pytree_data, metadata

    def unflatten_fn(metadata, pytree_data):
        kwargs = {}
        for name, value in zip(meta_fields, metadata):
            kwargs[name] = value
        for name, value in zip(pytree_fields, pytree_data):
            kwargs[name] = value
        return cls(**kwargs)

    # Use new API if available, otherwise fall back
    if hasattr(jax.tree_util, 'register_dataclass'):
        jax.tree_util.register_dataclass(cls, pytree_fields, meta_fields)
    else:
        jax.tree_util.register_pytree_with_keys(
            cls,
            flatten_with_keys_fn,
            unflatten_fn,
            flatten_fn
        )


@dataclass_transform(field_specifiers=(field,))
class PyTreeNode:
    """
    Base class for creating JAX-compatible pytree nodes.

    Subclasses of PyTreeNode are automatically converted to immutable
    dataclasses that work with JAX transformations.

    See Also
    --------
    dataclass : Decorator for creating JAX-compatible dataclasses.
    field : Create dataclass fields with pytree metadata.

    Notes
    -----
    When subclassing PyTreeNode, all fields are automatically treated as
    part of the pytree unless explicitly marked with ``pytree_node=False``
    using the field() function.

    Examples
    --------
    .. code-block:: python

        >>> import jax
        >>> import jax.numpy as jnp
        >>> from brainstate.util import PyTreeNode, field

        >>> class Layer(PyTreeNode):
        ...     weights: jax.Array
        ...     bias: jax.Array
        ...     activation: str = field(pytree_node=False, default="relu")

        >>> layer = Layer(weights=jnp.ones((4, 4)), bias=jnp.zeros(4))

        >>> # Can be used in JAX transformations
        >>> def loss_fn(layer):
        ...     return jnp.sum(layer.weights ** 2)
        >>> grad_fn = jax.grad(loss_fn)
        >>> grads = grad_fn(layer)

        >>> # Create modified copies with replace
        >>> layer2 = layer.replace(bias=jnp.ones(4))
    """

    def __init_subclass__(cls, **kwargs):
        """Automatically apply dataclass decorator to subclasses."""
        dataclass(cls, **kwargs)

    def __init__(self, *args, **kwargs):
        """Stub for type checkers."""
        raise NotImplementedError("PyTreeNode is a base class")

    def replace(self: TNode, **updates) -> TNode:
        """
        Replace specified fields with new values.

        Parameters
        ----------
        **updates
            Field names and their new values.

        Returns
        -------
        TNode
            A new instance with updated fields.
        """
        raise NotImplementedError("Implemented by dataclass decorator")


@jax.tree_util.register_pytree_with_keys_class
class FrozenDict(Mapping[K, V], Generic[K, V]):
    """
    An immutable dictionary that works as a JAX pytree.

    FrozenDict provides an immutable mapping interface that can be used
    safely with JAX transformations. It supports all standard dictionary
    operations in an immutable fashion.

    Parameters
    ----------
    *args
        Positional arguments for dict construction.
    **kwargs
        Keyword arguments for dict construction.

    Attributes
    ----------
    _data : dict
        Internal immutable data storage.
    _hash : int or None
        Cached hash value.

    See Also
    --------
    freeze : Convert a mapping to a FrozenDict.
    unfreeze : Convert a FrozenDict to a regular dict.

    Notes
    -----
    FrozenDict is immutable - all operations that would modify the dictionary
    instead return a new FrozenDict instance with the changes applied.

    Examples
    --------
    .. code-block:: python

        >>> from brainstate.util import FrozenDict

        >>> # Create a FrozenDict
        >>> fd = FrozenDict({'a': 1, 'b': 2})
        >>> fd['a']
        1

        >>> # Copy with updates (returns new FrozenDict)
        >>> new_fd = fd.copy({'c': 3})
        >>> new_fd['c']
        3

        >>> # Pop an item (returns new dict and popped value)
        >>> new_fd, value = fd.pop('b')
        >>> value
        2
        >>> 'b' in new_fd
        False

        >>> # Nested dictionaries are automatically frozen
        >>> fd = FrozenDict({'x': {'y': 1}})
        >>> isinstance(fd['x'], FrozenDict)
        True
    """

    __slots__ = ('_data', '_hash')

    def __init__(self, *args, **kwargs):
        """Initialize a FrozenDict."""
        data = dict(*args, **kwargs)
        self._data = self._deep_freeze(data)
        self._hash = None

    @staticmethod
    def _deep_freeze(obj: Any) -> Any:
        """Recursively freeze nested dictionaries."""
        if isinstance(obj, FrozenDict):
            return obj._data
        elif isinstance(obj, dict):
            return {k: FrozenDict._deep_freeze(v) for k, v in obj.items()}
        else:
            return obj

    def __getitem__(self, key: K) -> V:
        """Get an item from the dictionary."""
        value = self._data[key]
        if isinstance(value, dict):
            return FrozenDict(value)
        return value

    def __setitem__(self, key: K, value: V) -> None:
        """Raise an error - FrozenDict is immutable."""
        raise TypeError("FrozenDict does not support item assignment")

    def __delitem__(self, key: K) -> None:
        """Raise an error - FrozenDict is immutable."""
        raise TypeError("FrozenDict does not support item deletion")

    def __contains__(self, key: object) -> bool:
        """Check if a key is in the dictionary."""
        return key in self._data

    def __iter__(self) -> Iterator[K]:
        """Iterate over keys."""
        return iter(self._data)

    def __len__(self) -> int:
        """Return the number of items."""
        return len(self._data)

    def __repr__(self) -> str:
        """Return a string representation."""
        return self.pretty_repr()

    def __hash__(self) -> int:
        """Return a hash of the dictionary."""
        if self._hash is None:
            items = []
            for key, value in self.items():
                if isinstance(value, dict):
                    value = FrozenDict(value)
                items.append((key, value))
            self._hash = hash(tuple(sorted(items)))
        return self._hash

    def __eq__(self, other: object) -> bool:
        """Check equality with another object."""
        if not isinstance(other, (FrozenDict, dict)):
            return NotImplemented
        if isinstance(other, FrozenDict):
            return self._data == other._data
        return self._data == other

    def __reduce__(self):
        """Support for pickling."""
        return FrozenDict, (self.unfreeze(),)

    def keys(self) -> KeysView[K]:
        """
        Return a view of the keys.

        Returns
        -------
        KeysView
            A view object of the dictionary's keys.
        """
        return FrozenKeysView(self)

    def values(self) -> ValuesView[V]:
        """
        Return a view of the values.

        Returns
        -------
        ValuesView
            A view object of the dictionary's values.
        """
        return FrozenValuesView(self)

    def items(self) -> ItemsView[K, V]:
        """
        Return a view of the items.

        Yields
        ------
        tuple
            Key-value pairs from the dictionary.
        """
        for key in self._data:
            yield (key, self[key])

    def get(self, key: K, default: V | None = None) -> V | None:
        """
        Get a value with a default.

        Parameters
        ----------
        key : K
            The key to look up.
        default : V or None, optional
            The default value to return if key is not found.

        Returns
        -------
        V or None
            The value associated with the key, or default.
        """
        try:
            return self[key]
        except KeyError:
            return default

    def copy(self, add_or_replace: Mapping[K, V] | None = None) -> FrozenDict[K, V]:
        """
        Create a new FrozenDict with added or replaced entries.

        Parameters
        ----------
        add_or_replace : Mapping or None, optional
            Entries to add or replace in the new dictionary.

        Returns
        -------
        FrozenDict
            A new FrozenDict with the updates applied.

        Examples
        --------
        .. code-block:: python

            >>> fd = FrozenDict({'a': 1, 'b': 2})
            >>> fd2 = fd.copy({'b': 3, 'c': 4})
            >>> fd2['b'], fd2['c']
            (3, 4)
        """
        if add_or_replace is None:
            add_or_replace = {}
        new_data = dict(self._data)
        new_data.update(add_or_replace)
        return type(self)(new_data)

    def pop(self, key: K) -> tuple[FrozenDict[K, V], V]:
        """
        Create a new FrozenDict with one entry removed.

        Parameters
        ----------
        key : K
            The key to remove.

        Returns
        -------
        tuple
            A tuple of (new FrozenDict without the key, removed value).

        Raises
        ------
        KeyError
            If the key is not found in the dictionary.

        Examples
        --------
        .. code-block:: python

            >>> fd = FrozenDict({'a': 1, 'b': 2})
            >>> fd2, value = fd.pop('a')
            >>> value
            1
            >>> 'a' in fd2
            False
        """
        if key not in self._data:
            raise KeyError(key)
        value = self[key]
        new_data = dict(self._data)
        del new_data[key]
        return type(self)(new_data), value

    def unfreeze(self) -> dict[K, V]:
        """
        Convert to a regular mutable dictionary.

        Returns
        -------
        dict
            A mutable dict with the same contents.

        Examples
        --------
        .. code-block:: python

            >>> fd = FrozenDict({'a': 1, 'b': {'c': 2}})
            >>> d = fd.unfreeze()
            >>> isinstance(d, dict)
            True
            >>> isinstance(d['b'], dict)  # Nested dicts also unfrozen
            True
        """
        return unfreeze(self)

    def pretty_repr(self, indent: int = 2) -> str:
        """
        Return a pretty-printed representation.

        Parameters
        ----------
        indent : int, optional
            Number of spaces per indentation level (default 2).

        Returns
        -------
        str
            A formatted string representation of the FrozenDict.
        """

        def format_value(v, level):
            if isinstance(v, dict):
                if not v:
                    return '{}'
                items = []
                for k, val in v.items():
                    formatted_val = format_value(val, level + 1)
                    items.append(f'{" " * (level + 1) * indent}{k!r}: {formatted_val}')
                return '{\n' + ',\n'.join(items) + f'\n{" " * level * indent}}}'
            else:
                return repr(v)

        if not self._data:
            return 'FrozenDict({})'

        return f'FrozenDict({format_value(self._data, 0)})'

    def tree_flatten_with_keys(self) -> tuple[list[tuple[Any, Any]], tuple[Any, ...]]:
        """Flatten for JAX pytree with keys."""
        sorted_keys = sorted(self._data.keys())
        values_with_keys = [
            (jax.tree_util.DictKey(k), self._data[k])
            for k in sorted_keys
        ]
        return values_with_keys, tuple(sorted_keys)

    @classmethod
    def tree_unflatten(cls, keys: tuple[Any, ...], values: list[Any]) -> FrozenDict:
        """Unflatten from JAX pytree."""
        return cls(dict(zip(keys, values)))


class FrozenKeysView(KeysView[K]):
    """View of keys in a FrozenDict."""

    def __repr__(self) -> str:
        return f'FrozenDict.keys({list(self)})'


class FrozenValuesView(ValuesView[V]):
    """View of values in a FrozenDict."""

    def __repr__(self) -> str:
        return f'FrozenDict.values({list(self)})'


def freeze(x: Mapping[K, V]) -> FrozenDict[K, V]:
    """
    Convert a mapping to a FrozenDict.

    Parameters
    ----------
    x : Mapping
        A mapping (dict, FrozenDict, etc.) to freeze.

    Returns
    -------
    FrozenDict
        An immutable FrozenDict.

    See Also
    --------
    unfreeze : Convert a FrozenDict to a regular dict.
    FrozenDict : The immutable dictionary class.

    Examples
    --------
    .. code-block:: python

        >>> from brainstate.util import freeze

        >>> d = {'a': 1, 'b': {'c': 2}}
        >>> fd = freeze(d)
        >>> isinstance(fd, FrozenDict)
        True
        >>> isinstance(fd['b'], FrozenDict)  # Nested dicts are frozen
        True
    """
    if isinstance(x, FrozenDict):
        return x
    return FrozenDict(x)


def unfreeze(x: FrozenDict[K, V] | dict[K, V]) -> dict[K, V]:
    """
    Convert a FrozenDict to a regular dict.

    Recursively converts FrozenDict instances to mutable dicts.

    Parameters
    ----------
    x : FrozenDict or dict
        A FrozenDict or dict to unfreeze.

    Returns
    -------
    dict
        A mutable dictionary.

    See Also
    --------
    freeze : Convert a mapping to a FrozenDict.
    FrozenDict : The immutable dictionary class.

    Examples
    --------
    .. code-block:: python

        >>> from brainstate.util import FrozenDict, unfreeze

        >>> fd = FrozenDict({'a': 1, 'b': {'c': 2}})
        >>> d = unfreeze(fd)
        >>> isinstance(d, dict)
        True
        >>> isinstance(d['b'], dict)  # Nested FrozenDicts are unfrozen
        True
        >>> d['a'] = 10  # Can modify the result
    """
    if isinstance(x, FrozenDict):
        result = {}
        for key, value in x._data.items():
            result[key] = unfreeze(value)
        return result
    elif isinstance(x, dict):
        result = {}
        for key, value in x.items():
            result[key] = unfreeze(value)
        return result
    else:
        return x


@overload
def copy(x: FrozenDict[K, V], add_or_replace: Mapping[K, V] | None = None) -> FrozenDict[K, V]:
    ...


@overload
def copy(x: dict[K, V], add_or_replace: Mapping[K, V] | None = None) -> dict[K, V]:
    ...


def copy(x, add_or_replace=None):
    """
    Copy a dictionary with optional updates.

    Works with both FrozenDict and regular dict.

    Parameters
    ----------
    x : FrozenDict or dict
        Dictionary to copy.
    add_or_replace : Mapping or None, optional
        Entries to add or replace in the copy.

    Returns
    -------
    FrozenDict or dict
        A copy of the same type as the input with updates applied.

    Raises
    ------
    TypeError
        If x is not a FrozenDict or dict.

    See Also
    --------
    FrozenDict.copy : Copy method for FrozenDict.

    Examples
    --------
    .. code-block:: python

        >>> from brainstate.util import FrozenDict, copy

        >>> # Works with FrozenDict
        >>> fd = FrozenDict({'a': 1})
        >>> fd2 = copy(fd, {'b': 2})
        >>> isinstance(fd2, FrozenDict)
        True
        >>> fd2['b']
        2

        >>> # Also works with regular dict
        >>> d = {'a': 1}
        >>> d2 = copy(d, {'b': 2})
        >>> isinstance(d2, dict)
        True
        >>> d2['b']
        2
    """
    if add_or_replace is None:
        add_or_replace = {}

    if isinstance(x, FrozenDict):
        return x.copy(add_or_replace)
    elif isinstance(x, dict):
        result = dict(x)
        result.update(add_or_replace)
        return result
    else:
        raise TypeError(f"Expected FrozenDict or dict, got {type(x)}")


@overload
def pop(x: FrozenDict[K, V], key: K) -> tuple[FrozenDict[K, V], V]:
    ...


@overload
def pop(x: dict[K, V], key: K) -> tuple[dict[K, V], V]:
    ...


def pop(x, key):
    """
    Remove and return an item from a dictionary.

    Works with both FrozenDict and regular dict, returning a new
    dictionary without the specified key along with the popped value.

    Parameters
    ----------
    x : FrozenDict or dict
        Dictionary to pop from.
    key : hashable
        Key to remove.

    Returns
    -------
    tuple
        A tuple of (new dictionary without the key, popped value).

    Raises
    ------
    TypeError
        If x is not a FrozenDict or dict.
    KeyError
        If the key is not found in the dictionary.

    See Also
    --------
    FrozenDict.pop : Pop method for FrozenDict.

    Examples
    --------
    .. code-block:: python

        >>> from brainstate.util import FrozenDict, pop

        >>> # Works with FrozenDict
        >>> fd = FrozenDict({'a': 1, 'b': 2})
        >>> fd2, value = pop(fd, 'a')
        >>> value
        1
        >>> 'a' in fd2
        False

        >>> # Also works with regular dict
        >>> d = {'a': 1, 'b': 2}
        >>> d2, value = pop(d, 'a')
        >>> value
        1
        >>> 'a' in d2
        False
    """
    if isinstance(x, FrozenDict):
        return x.pop(key)
    elif isinstance(x, dict):
        new_dict = dict(x)
        value = new_dict.pop(key)
        return new_dict, value
    else:
        raise TypeError(f"Expected FrozenDict or dict, got {type(x)}")


def pretty_repr(x: Any, indent: int = 2) -> str:
    """
    Create a pretty string representation.

    Parameters
    ----------
    x : any
        Object to represent. If a dict or FrozenDict, will be
        pretty-printed with indentation. Otherwise, returns repr(x).
    indent : int, optional
        Number of spaces per indentation level (default 2).

    Returns
    -------
    str
        A formatted string representation.

    See Also
    --------
    FrozenDict.pretty_repr : Pretty representation for FrozenDict.

    Examples
    --------
    .. code-block:: python

        >>> from brainstate.util import pretty_repr

        >>> d = {'a': 1, 'b': {'c': 2, 'd': 3}}
        >>> print(pretty_repr(d))
        {
          'a': 1,
          'b': {
            'c': 2,
            'd': 3
          }
        }

        >>> # Non-dict objects return normal repr
        >>> pretty_repr([1, 2, 3])
        '[1, 2, 3]'
    """
    if isinstance(x, FrozenDict):
        return x.pretty_repr(indent)
    elif isinstance(x, dict):
        def format_dict(d, level):
            if not d:
                return '{}'
            items = []
            for k, v in d.items():
                if isinstance(v, dict):
                    formatted = format_dict(v, level + 1)
                else:
                    formatted = repr(v)
                items.append(f'{" " * (level + 1) * indent}{k!r}: {formatted}')
            return '{\n' + ',\n'.join(items) + f'\n{" " * level * indent}}}'

        return format_dict(x, 0)
    else:
        return repr(x)
