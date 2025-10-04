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

"""
Utility functions and classes for BrainState.

This module provides various utility functions and enhanced dictionary classes
for managing collections, memory, and object operations in the BrainState framework.
"""

import copy
import functools
import gc
import threading
import types
import warnings
from collections.abc import Iterable, Mapping, MutableMapping
from typing import (
    Any, Callable, Dict, Iterator, List, Optional,
    Tuple, Type, TypeVar, Union, overload
)

import jax
from jax.lib import xla_bridge

from brainstate._utils import set_module_as

__all__ = [
    'split_total',
    'clear_buffer_memory',
    'not_instance_eval',
    'is_instance_eval',
    'DictManager',
    'DotDict',
    'get_unique_name',
    'merge_dicts',
    'flatten_dict',
    'unflatten_dict',
]

T = TypeVar('T')
V = TypeVar('V')
K = TypeVar('K')


def split_total(
    total: int,
    fraction: Union[int, float],
) -> int:
    """
    Calculate the number of epochs for simulation based on a total and a fraction.

    This function determines the number of epochs to simulate given a total number
    of epochs and either a fraction or a specific number of epochs to run.

    Parameters
    ----------
    total : int
        The total number of epochs. Must be a positive integer.
    fraction : Union[int, float]
        If ``float``: A value between 0 and 1 representing the fraction of total epochs to run.
        If ``int``: The specific number of epochs to run, must not exceed the total.

    Returns
    -------
    int
        The calculated number of epochs to simulate.

    Raises
    ------
    TypeError
        If total is not an integer.
    ValueError
        If total is not positive, fraction is negative, or if fraction as float is > 1
        or as int is > total.

    Examples
    --------
    >>> split_total(100, 0.5)
    50
    >>> split_total(100, 25)
    25
    >>> split_total(100, 1.5)  # Raises ValueError
    ValueError: 'fraction' value cannot be greater than 1.
    """
    if not isinstance(total, int):
        raise TypeError(f"'total' must be an integer, got {type(total).__name__}.")
    if total <= 0:
        raise ValueError(f"'total' must be a positive integer, got {total}.")

    if isinstance(fraction, float):
        if fraction < 0:
            raise ValueError(f"'fraction' value cannot be negative, got {fraction}.")
        if fraction > 1:
            raise ValueError(f"'fraction' value cannot be greater than 1, got {fraction}.")
        return int(total * fraction)

    elif isinstance(fraction, int):
        if fraction < 0:
            raise ValueError(f"'fraction' value cannot be negative, got {fraction}.")
        if fraction > total:
            raise ValueError(f"'fraction' value cannot be greater than total ({total}), got {fraction}.")
        return fraction

    else:
        raise TypeError(f"'fraction' must be an integer or float, got {type(fraction).__name__}.")


class NameContext(threading.local):
    """Thread-local context for managing unique names."""

    def __init__(self):
        self.typed_names: Dict[str, int] = {}

    def reset(self, type_: Optional[str] = None) -> None:
        """Reset the counter for a specific type or all types."""
        if type_ is None:
            self.typed_names.clear()
        elif type_ in self.typed_names:
            self.typed_names[type_] = 0


NAME = NameContext()


@set_module_as('brainstate.util')
def get_unique_name(type_: str, prefix: str = '') -> str:
    """
    Get a unique name for the given object type.

    Parameters
    ----------
    type_ : str
        The base type name.
    prefix : str, optional
        Additional prefix to add before the type name.

    Returns
    -------
    str
        A unique name combining prefix, type, and counter.

    Examples
    --------
    >>> get_unique_name('layer')
    'layer0'
    >>> get_unique_name('layer', 'conv_')
    'conv_layer1'
    """
    if type_ not in NAME.typed_names:
        NAME.typed_names[type_] = 0

    full_prefix = f'{prefix}{type_}' if prefix else type_
    name = f'{full_prefix}{NAME.typed_names[type_]}'
    NAME.typed_names[type_] += 1
    return name


@jax.tree_util.register_pytree_node_class
class DictManager(dict, MutableMapping[K, V]):
    """
    Enhanced dictionary for managing collections in BrainState.

    DictManager extends the standard Python dict with additional methods for
    filtering, splitting, and managing collections of objects. It's registered
    as a JAX pytree node for compatibility with JAX transformations.

    Examples
    --------
    >>> dm = DictManager({'a': 1, 'b': 2.0, 'c': 'text'})
    >>> dm.subset(int)  # Get only integer values
    DictManager({'a': 1})
    >>> dm.unique()  # Get unique values only
    DictManager({'a': 1, 'b': 2.0, 'c': 'text'})
    """

    __module__ = 'brainstate.util'
    _val_id_to_key: Dict[int, Any]

    def __init__(self, *args, **kwargs):
        """Initialize DictManager with optional dict-like arguments."""
        super().__init__(*args, **kwargs)
        self._val_id_to_key = {}

    def subset(self, sep: Union[Type, Tuple[Type, ...], Callable[[Any], bool]]) -> 'DictManager':
        """
        Get a new DictManager with a subset of items based on value type or predicate.

        Parameters
        ----------
        sep : Union[Type, Tuple[Type, ...], Callable]
            If Type or Tuple of Types: Select values that are instances of these types.
            If Callable: Select values where sep(value) returns True.

        Returns
        -------
        DictManager
            A new DictManager containing only matching items.

        Examples
        --------
        >>> dm = DictManager({'a': 1, 'b': 2.0, 'c': 'text'})
        >>> dm.subset(int)
        DictManager({'a': 1})
        >>> dm.subset(lambda x: isinstance(x, (int, float)))
        DictManager({'a': 1, 'b': 2.0})
        """
        gather = type(self)()
        if callable(sep) and not isinstance(sep, type):
            for k, v in self.items():
                if sep(v):
                    gather[k] = v
        else:
            for k, v in self.items():
                if isinstance(v, sep):
                    gather[k] = v
        return gather

    def not_subset(self, sep: Union[Type, Tuple[Type, ...]]) -> 'DictManager':
        """
        Get a new DictManager excluding items of specified types.

        Parameters
        ----------
        sep : Union[Type, Tuple[Type, ...]]
            Types to exclude from the result.

        Returns
        -------
        DictManager
            A new DictManager excluding items of specified types.
        """
        gather = type(self)()
        for k, v in self.items():
            if not isinstance(v, sep):
                gather[k] = v
        return gather

    def add_unique_key(self, key: K, val: V) -> None:
        """
        Add a new element ensuring the key maps to a unique value.

        Parameters
        ----------
        key : Any
            The key to add.
        val : Any
            The value to associate with the key.

        Raises
        ------
        ValueError
            If the key already exists with a different value.
        """
        self._check_elem(val)
        if key in self:
            if id(val) != id(self[key]):
                raise ValueError(
                    f"Key '{key}' already exists with a different value. "
                    f"Existing: {self[key]}, New: {val}"
                )
        else:
            self[key] = val

    def add_unique_value(self, key: K, val: V) -> bool:
        """
        Add a new element only if the value is unique across all entries.

        Parameters
        ----------
        key : Any
            The key to add.
        val : Any
            The value to associate with the key.

        Returns
        -------
        bool
            True if the value was added (was unique), False otherwise.
        """
        self._check_elem(val)
        if not hasattr(self, '_val_id_to_key'):
            self._val_id_to_key = {id(v): k for k, v in self.items()}

        val_id = id(val)
        if val_id not in self._val_id_to_key:
            self._val_id_to_key[val_id] = key
            self[key] = val
            return True
        return False

    def unique(self) -> 'DictManager':
        """
        Get a new DictManager with unique values only.

        If multiple keys map to the same value (by identity),
        only the first key-value pair is retained.

        Returns
        -------
        DictManager
            A new DictManager with unique values.
        """
        gather = type(self)()
        seen = set()
        for k, v in self.items():
            v_id = id(v)
            if v_id not in seen:
                seen.add(v_id)
                gather[k] = v
        return gather

    def unique_(self) -> 'DictManager':
        """
        Remove duplicate values in-place.

        Returns
        -------
        DictManager
            Self, for method chaining.
        """
        seen = set()
        keys_to_remove = []
        for k, v in self.items():
            v_id = id(v)
            if v_id in seen:
                keys_to_remove.append(k)
            else:
                seen.add(v_id)

        for k in keys_to_remove:
            del self[k]
        return self

    def assign(self, *args: Dict[K, V], **kwargs: V) -> None:
        """
        Update the DictManager with multiple dictionaries.

        Parameters
        ----------
        *args : Dict
            Dictionaries to merge into this one.
        **kwargs
            Additional key-value pairs to add.
        """
        for arg in args:
            if not isinstance(arg, dict):
                raise TypeError(f"Arguments must be dict instances, got {type(arg).__name__}")
            self.update(arg)
        if kwargs:
            self.update(kwargs)

    def split(self, *types: Type) -> Tuple['DictManager', ...]:
        """
        Split the DictManager into multiple based on value types.

        Parameters
        ----------
        *types : Type
            Types to use for splitting. Each type gets its own DictManager.

        Returns
        -------
        Tuple[DictManager, ...]
            A tuple of DictManagers, one for each type plus one for unmatched items.
        """
        results = tuple(type(self)() for _ in range(len(types) + 1))

        for k, v in self.items():
            for i, type_ in enumerate(types):
                if isinstance(v, type_):
                    results[i][k] = v
                    break
            else:
                results[-1][k] = v

        return results

    def filter_by_predicate(self, predicate: Callable[[K, V], bool]) -> 'DictManager':
        """
        Filter items using a predicate function.

        Parameters
        ----------
        predicate : Callable[[key, value], bool]
            Function that returns True for items to keep.

        Returns
        -------
        DictManager
            A new DictManager with filtered items.
        """
        return type(self)({k: v for k, v in self.items() if predicate(k, v)})

    def map_values(self, func: Callable[[V], Any]) -> 'DictManager':
        """
        Apply a function to all values.

        Parameters
        ----------
        func : Callable
            Function to apply to each value.

        Returns
        -------
        DictManager
            A new DictManager with transformed values.
        """
        return type(self)({k: func(v) for k, v in self.items()})

    def map_keys(self, func: Callable[[K], Any]) -> 'DictManager':
        """
        Apply a function to all keys.

        Parameters
        ----------
        func : Callable
            Function to apply to each key.

        Returns
        -------
        DictManager
            A new DictManager with transformed keys.

        Raises
        ------
        ValueError
            If the transformation creates duplicate keys.
        """
        result = type(self)()
        for k, v in self.items():
            new_key = func(k)
            if new_key in result:
                raise ValueError(f"Key transformation created duplicate: {new_key}")
            result[new_key] = v
        return result

    def pop_by_keys(self, keys: Iterable[K]) -> None:
        """Remove multiple keys from the DictManager."""
        keys_set = set(keys)
        for k in list(self.keys()):
            if k in keys_set:
                self.pop(k)

    def pop_by_values(self, values: Iterable[V], by: str = 'id') -> None:
        """
        Remove items by their values.

        Parameters
        ----------
        values : Iterable
            Values to remove.
        by : str
            Comparison method: 'id' (identity) or 'value' (equality).
        """
        if by == 'id':
            value_ids = {id(v) for v in values}
            keys_to_remove = [k for k, v in self.items() if id(v) in value_ids]
        elif by == 'value':
            values_set = set(values) if not isinstance(values, set) else values
            keys_to_remove = [k for k, v in self.items() if v in values_set]
        else:
            raise ValueError(f"Invalid comparison method: {by}. Use 'id' or 'value'.")

        for k in keys_to_remove:
            del self[k]

    def difference_by_keys(self, keys: Iterable[K]) -> 'DictManager':
        """Get items not in the specified keys."""
        keys_set = set(keys)
        return type(self)({k: v for k, v in self.items() if k not in keys_set})

    def difference_by_values(self, values: Iterable[V], by: str = 'id') -> 'DictManager':
        """Get items whose values are not in the specified collection."""
        if by == 'id':
            value_ids = {id(v) for v in values}
            return type(self)({k: v for k, v in self.items() if id(v) not in value_ids})
        elif by == 'value':
            values_set = set(values) if not isinstance(values, set) else values
            return type(self)({k: v for k, v in self.items() if v not in values_set})
        else:
            raise ValueError(f"Invalid comparison method: {by}. Use 'id' or 'value'.")

    def intersection_by_keys(self, keys: Iterable[K]) -> 'DictManager':
        """Get items with keys in the specified collection."""
        keys_set = set(keys)
        return type(self)({k: v for k, v in self.items() if k in keys_set})

    def intersection_by_values(self, values: Iterable[V], by: str = 'id') -> 'DictManager':
        """Get items whose values are in the specified collection."""
        if by == 'id':
            value_ids = {id(v) for v in values}
            return type(self)({k: v for k, v in self.items() if id(v) in value_ids})
        elif by == 'value':
            values_set = set(values) if not isinstance(values, set) else values
            return type(self)({k: v for k, v in self.items() if v in values_set})
        else:
            raise ValueError(f"Invalid comparison method: {by}. Use 'id' or 'value'.")

    def __add__(self, other: Mapping[K, V]) -> 'DictManager':
        """Combine with another mapping using the + operator."""
        if not isinstance(other, Mapping):
            return NotImplemented
        new_dict = type(self)(self)
        new_dict.update(other)
        return new_dict

    def __or__(self, other: Mapping[K, V]) -> 'DictManager':
        """Combine with another mapping using the | operator (Python 3.9+)."""
        if not isinstance(other, Mapping):
            return NotImplemented
        new_dict = type(self)(self)
        new_dict.update(other)
        return new_dict

    def __ior__(self, other: Mapping[K, V]) -> 'DictManager':
        """Update in-place with another mapping using |= operator."""
        if not isinstance(other, Mapping):
            return NotImplemented
        self.update(other)
        return self

    def tree_flatten(self) -> Tuple[Tuple[V, ...], Tuple[K, ...]]:
        """Flatten for JAX pytree."""
        return tuple(self.values()), tuple(self.keys())

    @classmethod
    def tree_unflatten(cls, keys: Tuple[K, ...], values: Tuple[V, ...]) -> 'DictManager':
        """Unflatten from JAX pytree."""
        return cls(zip(keys, values))

    def _check_elem(self, elem: Any) -> None:
        """Override in subclasses to validate elements."""
        pass

    def to_dict(self) -> Dict[K, V]:
        """Convert to a standard Python dict."""
        return dict(self)

    def __copy__(self) -> 'DictManager':
        """Shallow copy."""
        return type(self)(self)

    def __deepcopy__(self, memo: Dict[int, Any]) -> 'DictManager':
        """Deep copy."""
        return type(self)({
            copy.deepcopy(k, memo): copy.deepcopy(v, memo)
            for k, v in self.items()
        })

    def __repr__(self) -> str:
        """String representation."""
        items = ', '.join(f'{k!r}: {v!r}' for k, v in self.items())
        return f'{self.__class__.__name__}({{{items}}})'


@set_module_as('brainstate.util')
def clear_buffer_memory(
    platform: Optional[str] = None,
    array: bool = True,
    compilation: bool = False,
) -> None:
    """
    Clear on-device memory buffers and optionally compilation cache.

    This function is useful when running models in loops to prevent memory leaks
    by clearing cached arrays and freeing device memory.

    .. warning::
        This operation may invalidate existing array references.
        Regenerate data after calling this function.

    Parameters
    ----------
    platform : str, optional
        The specific device platform to clear. If None, clears the default platform.
    array : bool, default=True
        Whether to clear array buffers.
    compilation : bool, default=False
        Whether to clear the compilation cache.

    Examples
    --------
    >>> clear_buffer_memory()  # Clear array buffers
    >>> clear_buffer_memory(compilation=True)  # Also clear compilation cache
    """
    if array:
        try:
            backend = xla_bridge.get_backend(platform)
            for buf in backend.live_buffers():
                buf.delete()
        except Exception as e:
            warnings.warn(f"Failed to clear buffers: {e}", RuntimeWarning)

    if compilation:
        jax.clear_caches()

    gc.collect()


@jax.tree_util.register_pytree_node_class
class DotDict(dict, MutableMapping[str, Any]):
    """
    Dictionary with dot notation access to nested keys.

    DotDict allows accessing dictionary items using attribute syntax,
    making code more readable when dealing with nested configurations.

    Examples
    --------
    >>> config = DotDict({'model': {'layers': 3, 'units': 64}})
    >>> config.model.layers
    3
    >>> config.model.units = 128
    >>> config['model']['units']
    128

    Attributes
    ----------
    All dictionary keys become accessible as attributes unless they conflict
    with built-in methods.
    """

    __module__ = 'brainstate.util'

    def __init__(self, *args, **kwargs):
        """
        Initialize DotDict with dict-like arguments.

        Parameters
        ----------
        *args
            Positional arguments (dicts, iterables of pairs).
        **kwargs
            Keyword arguments become key-value pairs.
        """
        # Handle parent reference for nested updates
        object.__setattr__(self, '__parent', kwargs.pop('__parent', None))
        object.__setattr__(self, '__key', kwargs.pop('__key', None))

        # Process positional arguments
        for arg in args:
            if not arg:
                continue
            elif isinstance(arg, dict):
                for key, val in arg.items():
                    self[key] = self._hook(val)
            elif isinstance(arg, tuple) and len(arg) == 2 and not isinstance(arg[0], tuple):
                # Single key-value pair
                self[arg[0]] = self._hook(arg[1])
            else:
                # Iterable of key-value pairs
                try:
                    for key, val in arg:
                        self[key] = self._hook(val)
                except (TypeError, ValueError) as e:
                    raise TypeError(f"Invalid argument type for DotDict: {type(arg).__name__}") from e

        # Process keyword arguments
        for key, val in kwargs.items():
            self[key] = self._hook(val)

    def __setattr__(self, name: str, value: Any) -> None:
        """Set attribute as dictionary item."""
        if hasattr(self.__class__, name):
            raise AttributeError(
                f"Cannot set attribute '{name}': it's a built-in method of {self.__class__.__name__}"
            )
        self[name] = value

    def __setitem__(self, name: str, value: Any) -> None:
        """Set item and update parent if nested."""
        super().__setitem__(name, value)
        try:
            parent = object.__getattribute__(self, '__parent')
            key = object.__getattribute__(self, '__key')
            if parent is not None:
                parent[key] = self
                object.__delattr__(self, '__parent')
                object.__delattr__(self, '__key')
        except AttributeError:
            pass

    @classmethod
    def _hook(cls, item: Any) -> Any:
        """Convert nested dicts to DotDict."""
        if isinstance(item, dict) and not isinstance(item, cls):
            return cls(item)
        elif isinstance(item, (list, tuple)):
            return type(item)(cls._hook(elem) for elem in item)
        return item

    def __getattr__(self, item: str) -> Any:
        """Get attribute from dictionary."""
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{item}'")

    def __delattr__(self, name: str) -> None:
        """Delete attribute from dictionary."""
        try:
            del self[name]
        except KeyError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __dir__(self) -> List[str]:
        """List all attributes including dict keys."""
        return list(self.keys()) + dir(self.__class__)

    def get(self, key: str, default: Any = None) -> Any:
        """Get item with default value."""
        return super().get(key, default)

    def copy(self) -> 'DotDict':
        """Create a shallow copy."""
        return copy.copy(self)

    def deepcopy(self) -> 'DotDict':
        """Create a deep copy."""
        return copy.deepcopy(self)

    def __deepcopy__(self, memo: Dict[int, Any]) -> 'DotDict':
        """Deep copy implementation."""
        other = self.__class__()
        memo[id(self)] = other
        for key, value in self.items():
            other[copy.deepcopy(key, memo)] = copy.deepcopy(value, memo)
        return other

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to standard dict recursively.

        Returns
        -------
        dict
            A standard Python dict with nested DotDicts also converted.
        """
        result = {}
        for key, value in self.items():
            if isinstance(value, DotDict):
                result[key] = value.to_dict()
            elif isinstance(value, (list, tuple)):
                result[key] = type(value)(
                    item.to_dict() if isinstance(item, DotDict) else item
                    for item in value
                )
            else:
                result[key] = value
        return result

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'DotDict':
        """
        Create DotDict from standard dict.

        Parameters
        ----------
        d : dict
            Standard Python dictionary.

        Returns
        -------
        DotDict
            A new DotDict instance.
        """
        return cls(d)

    def update(self, *args, **kwargs) -> None:
        """
        Update with recursive merge for nested dicts.

        Parameters
        ----------
        *args
            Dict-like objects to merge.
        **kwargs
            Key-value pairs to merge.
        """
        if args:
            if len(args) > 1:
                raise TypeError(f"update expected at most 1 argument, got {len(args)}")
            other = args[0]
        else:
            other = {}

        if hasattr(other, 'items'):
            other = dict(other.items())
        other.update(kwargs)

        for k, v in other.items():
            if k in self and isinstance(self[k], dict) and isinstance(v, dict):
                # Recursive merge for nested dicts
                if isinstance(self[k], DotDict):
                    self[k].update(v)
                else:
                    self[k] = DotDict(self[k])
                    self[k].update(v)
            else:
                self[k] = self._hook(v)

    def setdefault(self, key: str, default: Any = None) -> Any:
        """Set default value if key doesn't exist."""
        if key not in self:
            self[key] = default
        return self[key]

    def __getstate__(self) -> Dict[str, Any]:
        """Get state for pickling."""
        return dict(self)

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Set state from pickling."""
        self.update(state)

    def tree_flatten(self) -> Tuple[Tuple[Any, ...], Tuple[str, ...]]:
        """Flatten for JAX pytree."""
        return tuple(self.values()), tuple(self.keys())

    @classmethod
    def tree_unflatten(cls, keys: Tuple[str, ...], values: Tuple[Any, ...]) -> 'DotDict':
        """Unflatten from JAX pytree."""
        return cls(zip(keys, values))

    def __repr__(self) -> str:
        """String representation."""
        items = ', '.join(f'{k!r}: {v!r}' for k, v in self.items())
        return f'DotDict({{{items}}})'


@set_module_as('brainstate.util')
def merge_dicts(*dicts: Dict[K, V], recursive: bool = True) -> Dict[K, V]:
    """
    Merge multiple dictionaries.

    Parameters
    ----------
    *dicts : Dict
        Dictionaries to merge (later ones override earlier ones).
    recursive : bool, default=True
        Whether to recursively merge nested dicts.

    Returns
    -------
    Dict
        Merged dictionary.

    Examples
    --------
    >>> d1 = {'a': 1, 'b': {'c': 2}}
    >>> d2 = {'b': {'d': 3}, 'e': 4}
    >>> merge_dicts(d1, d2)
    {'a': 1, 'b': {'c': 2, 'd': 3}, 'e': 4}
    """
    result = {}

    for d in dicts:
        if not isinstance(d, dict):
            raise TypeError(f"All arguments must be dicts, got {type(d).__name__}")

        for key, value in d.items():
            if recursive and key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = merge_dicts(result[key], value, recursive=True)
            else:
                result[key] = value

    return result


@set_module_as('brainstate.util')
def flatten_dict(
    d: Dict[str, Any],
    parent_key: str = '',
    sep: str = '.'
) -> Dict[str, Any]:
    """
    Flatten a nested dictionary.

    Parameters
    ----------
    d : Dict
        Dictionary to flatten.
    parent_key : str, default=''
        Prefix for keys.
    sep : str, default='.'
        Separator between nested keys.

    Returns
    -------
    Dict
        Flattened dictionary.

    Examples
    --------
    >>> d = {'a': 1, 'b': {'c': 2, 'd': {'e': 3}}}
    >>> flatten_dict(d)
    {'a': 1, 'b.c': 2, 'b.d.e': 3}
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


@set_module_as('brainstate.util')
def unflatten_dict(
    d: Dict[str, Any],
    sep: str = '.'
) -> Dict[str, Any]:
    """
    Unflatten a dictionary with separated keys.

    Parameters
    ----------
    d : Dict
        Flattened dictionary.
    sep : str, default='.'
        Separator in keys.

    Returns
    -------
    Dict
        Nested dictionary.

    Examples
    --------
    >>> d = {'a': 1, 'b.c': 2, 'b.d.e': 3}
    >>> unflatten_dict(d)
    {'a': 1, 'b': {'c': 2, 'd': {'e': 3}}}
    """
    result = {}

    for key, value in d.items():
        parts = key.split(sep)
        current = result

        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        current[parts[-1]] = value

    return result


def _is_not_instance(x: Any, cls: Union[Type, Tuple[Type, ...]]) -> bool:
    """Check if x is not an instance of cls."""
    return not isinstance(x, cls)


def _is_instance(x: Any, cls: Union[Type, Tuple[Type, ...]]) -> bool:
    """Check if x is an instance of cls."""
    return isinstance(x, cls)


@set_module_as('brainstate.util')
def not_instance_eval(*cls: Type) -> Callable[[Any], bool]:
    """
    Create a partial function to check if input is NOT an instance of given classes.

    Parameters
    ----------
    *cls : Type
        Classes to check against.

    Returns
    -------
    Callable
        A function that returns True if input is not an instance of any given class.

    Examples
    --------
    >>> not_int = not_instance_eval(int)
    >>> not_int(5)
    False
    >>> not_int("hello")
    True
    """
    return functools.partial(_is_not_instance, cls=cls)


@set_module_as('brainstate.util')
def is_instance_eval(*cls: Type) -> Callable[[Any], bool]:
    """
    Create a partial function to check if input IS an instance of given classes.

    Parameters
    ----------
    *cls : Type
        Classes to check against.

    Returns
    -------
    Callable
        A function that returns True if input is an instance of any given class.

    Examples
    --------
    >>> is_number = is_instance_eval(int, float)
    >>> is_number(5)
    True
    >>> is_number(3.14)
    True
    >>> is_number("hello")
    False
    """
    return functools.partial(_is_instance, cls=cls)