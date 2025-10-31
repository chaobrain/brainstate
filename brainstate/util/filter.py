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
Filter utilities for traversing and selecting objects in nested structures.

This module provides a flexible filtering system for working with nested data
structures in BrainState. It offers various filter classes and utilities to
select, match, and transform objects based on their properties, types, or
positions within a hierarchical structure.

Key Features
------------
- **Type-based filtering**: Select objects by their type or inheritance
- **Tag-based filtering**: Filter objects that have specific tags
- **Path-based filtering**: Select based on object paths in nested structures
- **Logical operations**: Combine filters with AND, OR, and NOT operations
- **Flexible conversion**: Convert various inputs to predicate functions

Filter Types
------------
The module provides several built-in filter classes:

- :class:`WithTag`: Filters objects with specific tags
- :class:`PathContains`: Filters based on path contents
- :class:`OfType`: Filters by object type
- :class:`Any`: Logical OR combination of filters
- :class:`All`: Logical AND combination of filters
- :class:`Not`: Logical negation of a filter
- :class:`Everything`: Matches all objects
- :class:`Nothing`: Matches no objects

Examples
--------

.. code-block:: python

    >>> import brainstate as bs
    >>> from brainstate.util.filter import WithTag, OfType, Any, All, Not
    >>>
    >>> # Filter objects with a specific tag
    >>> tag_filter = WithTag('trainable')
    >>>
    >>> # Filter objects of a specific type
    >>> type_filter = OfType(bs.nn.Linear)
    >>>
    >>> # Combine filters with logical operations
    >>> combined_filter = All(
    ...     WithTag('trainable'),
    ...     OfType(bs.nn.Linear)
    ... )
    >>>
    >>> # Negate a filter
    >>> not_trainable = Not(WithTag('trainable'))
    >>>
    >>> # Use Any for OR operations
    >>> any_filter = Any(
    ...     OfType(bs.nn.Linear),
    ...     OfType(bs.nn.Conv)
    ... )

Using Filters with Tree Operations
-----------------------------------

.. code-block:: python

    >>> import brainstate as bs
    >>> import jax.tree_util as tree
    >>> from brainstate.util.filter import to_predicate, WithTag
    >>>
    >>> # Create a model with tagged parameters
    >>> class Model(bs.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.layer1 = bs.nn.Linear(10, 20)
    ...         self.layer1.tag = 'trainable'
    ...         self.layer2 = bs.nn.Linear(20, 10)
    ...         self.layer2.tag = 'frozen'
    >>>
    >>> model = Model()
    >>>
    >>> # Filter trainable parameters
    >>> trainable_filter = to_predicate('trainable')
    >>>
    >>> # Apply filter in tree operations
    >>> def get_trainable_params(model):
    ...     return tree.tree_map_with_path(
    ...         lambda path, x: x if trainable_filter(path, x) else None,
    ...         model
    ...     )

Notes
-----
This module is adapted from the Flax library and provides similar functionality
for filtering and selecting components in neural network models and other
hierarchical data structures.

See Also
--------
brainstate.tree : Tree manipulation utilities
brainstate.typing : Type definitions for filters and predicates

"""

import builtins
import dataclasses
import typing
from typing import TYPE_CHECKING

from brainstate.typing import Filter, PathParts, Predicate, Key

if TYPE_CHECKING:
    ellipsis = builtins.ellipsis
else:
    ellipsis = typing.Any

__all__ = [
    'to_predicate',
    'WithTag',
    'PathContains',
    'OfType',
    'Any',
    'All',
    'Nothing',
    'Not',
    'Everything',
]


def to_predicate(the_filter: Filter) -> Predicate:
    """
    Convert a Filter to a predicate function.

    This function takes various types of filters and converts them into
    corresponding predicate functions that can be used for filtering objects
    in nested structures.

    Parameters
    ----------
    the_filter : Filter
        The filter to be converted. Can be of various types:

        - **str**: Converted to a :class:`WithTag` filter
        - **type**: Converted to an :class:`OfType` filter
        - **bool**: ``True`` becomes :class:`Everything`, ``False`` becomes :class:`Nothing`
        - **Ellipsis** (...): Converted to :class:`Everything`
        - **None**: Converted to :class:`Nothing`
        - **callable**: Returned as-is
        - **list or tuple**: Converted to :class:`Any` filter with elements as arguments

    Returns
    -------
    Predicate
        A callable predicate function that takes (path, object) and returns bool.

    Raises
    ------
    TypeError
        If the input filter is of an invalid type.

    Examples
    --------
    .. code-block:: python

        >>> from brainstate.util.filter import to_predicate
        >>>
        >>> # Convert string to WithTag filter
        >>> pred = to_predicate('trainable')
        >>> pred([], {'tag': 'trainable'})
        True
        >>>
        >>> # Convert type to OfType filter
        >>> import numpy as np
        >>> pred = to_predicate(np.ndarray)
        >>> pred([], np.array([1, 2, 3]))
        True
        >>>
        >>> # Convert bool to Everything/Nothing
        >>> pred_all = to_predicate(True)
        >>> pred_all([], 'anything')
        True
        >>> pred_none = to_predicate(False)
        >>> pred_none([], 'anything')
        False
        >>>
        >>> # Convert list to Any filter
        >>> pred = to_predicate(['tag1', 'tag2'])
        >>> # This will match objects with either 'tag1' or 'tag2'

    See Also
    --------
    WithTag : Filter for objects with specific tags
    OfType : Filter for objects of specific types
    Any : Logical OR combination of filters
    Everything : Filter that matches all objects
    Nothing : Filter that matches no objects

    Notes
    -----
    This function is the main entry point for creating predicate functions
    from various filter specifications. It provides a flexible way to define
    filtering criteria without explicitly instantiating filter classes.
    """

    if isinstance(the_filter, str):
        return WithTag(the_filter)
    elif isinstance(the_filter, type):
        return OfType(the_filter)
    elif isinstance(the_filter, bool):
        if the_filter:
            return Everything()
        else:
            return Nothing()
    elif the_filter is Ellipsis:
        return Everything()
    elif the_filter is None:
        return Nothing()
    elif callable(the_filter):
        return the_filter
    elif isinstance(the_filter, (list, tuple)):
        return Any(*the_filter)
    else:
        raise TypeError(f'Invalid collection filter: {the_filter!r}. ')


@dataclasses.dataclass(frozen=True)
class WithTag:
    """
    Filter objects that have a specific tag attribute.

    This filter checks if an object has a 'tag' attribute that matches
    the specified tag value. It's commonly used to filter parameters or
    modules in neural networks based on their assigned tags.

    Parameters
    ----------
    tag : str
        The tag value to match against.

    Attributes
    ----------
    tag : str
        The tag value to match against.

    Examples
    --------
    .. code-block:: python

        >>> from brainstate.util.filter import WithTag
        >>> import brainstate as bs
        >>>
        >>> # Create a filter for 'trainable' tag
        >>> filter_trainable = WithTag('trainable')
        >>>
        >>> # Test with an object that has the tag
        >>> class Param:
        ...     def __init__(self, tag):
        ...         self.tag = tag
        >>>
        >>> param1 = Param('trainable')
        >>> param2 = Param('frozen')
        >>>
        >>> filter_trainable([], param1)
        True
        >>> filter_trainable([], param2)
        False
        >>>
        >>> # Use with neural network modules
        >>> class MyModule(bs.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.weight = bs.State(bs.random.randn(10, 10))
        ...         self.weight.tag = 'trainable'
        ...         self.bias = bs.State(bs.zeros(10))
        ...         self.bias.tag = 'frozen'

    See Also
    --------
    PathContains : Filter based on path contents
    OfType : Filter based on object type
    to_predicate : Convert various inputs to predicates

    Notes
    -----
    The filter only matches objects that have a 'tag' attribute. Objects
    without this attribute will not match, even if the filter is looking
    for a specific tag value.
    """

    tag: str

    def __call__(self, path: PathParts, x: typing.Any) -> bool:
        """
        Check if the object has a matching tag.

        Parameters
        ----------
        path : PathParts
            The path to the current object (not used in this filter).
        x : Any
            The object to check for the tag.

        Returns
        -------
        bool
            True if the object has a 'tag' attribute matching the specified tag,
            False otherwise.
        """
        return hasattr(x, 'tag') and x.tag == self.tag

    def __repr__(self) -> str:
        return f'WithTag({self.tag!r})'


@dataclasses.dataclass(frozen=True)
class PathContains:
    """
    Filter objects based on whether their path contains a specific key.

    This filter checks if a given key appears anywhere in the path to an object
    within a nested structure. It's useful for selecting objects at specific
    locations or with specific names in a hierarchy.

    Parameters
    ----------
    key : Key
        The key to search for in the path.

    Attributes
    ----------
    key : Key
        The key to search for in the path.

    Examples
    --------
    .. code-block:: python

        >>> from brainstate.util.filter import PathContains
        >>>
        >>> # Create a filter for paths containing 'weight'
        >>> weight_filter = PathContains('weight')
        >>>
        >>> # Test with different paths
        >>> weight_filter(['model', 'layer1', 'weight'], None)
        True
        >>> weight_filter(['model', 'layer1', 'bias'], None)
        False
        >>>
        >>> # Filter for specific layer
        >>> layer2_filter = PathContains('layer2')
        >>> layer2_filter(['model', 'layer2', 'weight'], None)
        True
        >>> layer2_filter(['model', 'layer1', 'weight'], None)
        False
        >>>
        >>> # Use with nested structures
        >>> import jax.tree_util as tree
        >>> nested_dict = {
        ...     'layer1': {'weight': [1, 2, 3], 'bias': [4, 5]},
        ...     'layer2': {'weight': [6, 7, 8], 'bias': [9, 10]}
        ... }
        >>>
        >>> # Filter all 'weight' entries
        >>> def filter_weights(path, value):
        ...     return value if weight_filter(path, value) else None

    See Also
    --------
    WithTag : Filter based on tag attributes
    OfType : Filter based on object type
    to_predicate : Convert various inputs to predicates

    Notes
    -----
    The path is typically a sequence of keys representing the location of
    an object in a nested structure, such as the attribute names leading
    to a parameter in a neural network model.
    """

    key: Key

    def __call__(self, path: PathParts, x: typing.Any) -> bool:
        """
        Check if the key is present in the path.

        Parameters
        ----------
        path : PathParts
            The path to check for the presence of the key.
        x : Any
            The object associated with the path (not used in this filter).

        Returns
        -------
        bool
            True if the key is present in the path, False otherwise.
        """
        return self.key in path

    def __repr__(self) -> str:
        return f'PathContains({self.key!r})'


@dataclasses.dataclass(frozen=True)
class OfType:
    """
    Filter objects based on their type.

    This filter checks if an object is an instance of a specific type or
    if it has a 'type' attribute that is a subclass of the specified type.
    It's useful for filtering specific kinds of objects in a nested structure.

    Parameters
    ----------
    type : type
        The type to match against.

    Attributes
    ----------
    type : type
        The type to match against.

    Examples
    --------
    .. code-block:: python

        >>> from brainstate.util.filter import OfType
        >>> import numpy as np
        >>> import jax.numpy as jnp
        >>>
        >>> # Create a filter for numpy arrays
        >>> array_filter = OfType(np.ndarray)
        >>>
        >>> # Test with different objects
        >>> array_filter([], np.array([1, 2, 3]))
        True
        >>> array_filter([], [1, 2, 3])
        False
        >>>
        >>> # Filter for specific module types
        >>> import brainstate as bs
        >>> linear_filter = OfType(bs.nn.Linear)
        >>>
        >>> # Use in model filtering
        >>> class Model(bs.nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.linear1 = bs.nn.Linear(10, 20)
        ...         self.linear2 = bs.nn.Linear(20, 10)
        ...         self.activation = bs.nn.ReLU()
        >>>
        >>> # Filter all Linear layers
        >>> model = Model()
        >>> # linear_filter will match linear1 and linear2, not activation

    See Also
    --------
    WithTag : Filter based on tag attributes
    PathContains : Filter based on path contents
    to_predicate : Convert various inputs to predicates

    Notes
    -----
    This filter also checks for objects that have a 'type' attribute,
    which is useful for wrapped or proxy objects that maintain type
    information differently.
    """
    type: type

    def __call__(self, path: PathParts, x: typing.Any):
        """
        Check if the object is of the specified type.

        Parameters
        ----------
        path : PathParts
            The path to the current object (not used in this filter).
        x : Any
            The object to check.

        Returns
        -------
        bool
            True if the object is an instance of the specified type or
            has a 'type' attribute that is a subclass of the specified type.
        """
        return isinstance(x, self.type) or (
            hasattr(x, 'type') and issubclass(x.type, self.type)
        )

    def __repr__(self):
        return f'OfType({self.type!r})'


class Any:
    """
    Combine multiple filters using logical OR operation.

    This filter returns True if any of its constituent filters return True.
    It's useful for creating flexible filtering criteria where multiple
    conditions can be satisfied.

    Parameters
    ----------
    *filters : Filter
        Variable number of filters to be combined with OR logic.

    Attributes
    ----------
    predicates : tuple of Predicate
        Tuple of predicate functions converted from the input filters.

    Examples
    --------
    .. code-block:: python

        >>> from brainstate.util.filter import Any, WithTag, OfType
        >>> import numpy as np
        >>>
        >>> # Create a filter that matches either tag
        >>> trainable_or_frozen = Any('trainable', 'frozen')
        >>>
        >>> # Test with objects
        >>> class Param:
        ...     def __init__(self, tag):
        ...         self.tag = tag
        >>>
        >>> trainable = Param('trainable')
        >>> frozen = Param('frozen')
        >>> other = Param('other')
        >>>
        >>> trainable_or_frozen([], trainable)
        True
        >>> trainable_or_frozen([], frozen)
        True
        >>> trainable_or_frozen([], other)
        False
        >>>
        >>> # Combine different filter types
        >>> array_or_list = Any(
        ...     OfType(np.ndarray),
        ...     OfType(list)
        ... )
        >>>
        >>> array_or_list([], np.array([1, 2, 3]))
        True
        >>> array_or_list([], [1, 2, 3])
        True
        >>> array_or_list([], (1, 2, 3))
        False

    See Also
    --------
    All : Logical AND combination of filters
    Not : Logical negation of a filter
    to_predicate : Convert various inputs to predicates

    Notes
    -----
    The Any filter short-circuits evaluation, returning True as soon as
    one of its constituent filters returns True.
    """

    def __init__(self, *filters: Filter):
        """
        Initialize the Any filter.

        Parameters
        ----------
        *filters : Filter
            Variable number of filters to be combined.
        """
        self.predicates = tuple(
            to_predicate(collection_filter) for collection_filter in filters
        )

    def __call__(self, path: PathParts, x: typing.Any) -> bool:
        """
        Apply the composite filter to the given path and object.

        Args:
            path (PathParts): The path to the current object.
            x (typing.Any): The object to be filtered.

        Returns:
            bool: True if any of the constituent predicates return True, False otherwise.
        """
        return any(predicate(path, x) for predicate in self.predicates)

    def __repr__(self) -> str:
        """
        Return a string representation of the Any filter.

        Returns:
            str: A string representation of the Any filter, including its predicates.
        """
        return f'Any({", ".join(map(repr, self.predicates))})'

    def __eq__(self, other) -> bool:
        """
        Check if this Any filter is equal to another object.

        Args:
            other: The object to compare with.

        Returns:
            bool: True if the other object is an Any filter with the same predicates, False otherwise.
        """
        return isinstance(other, Any) and self.predicates == other.predicates

    def __hash__(self) -> int:
        """
        Compute the hash value for this Any filter.

        Returns:
            int: The hash value of the predicates tuple.
        """
        return hash(self.predicates)


class All:
    """
    A filter class that combines multiple filters using a logical AND operation.

    This class creates a composite filter that returns True only if all of its
    constituent filters return True.

    Attributes:
        predicates (tuple): A tuple of predicate functions converted from the input filters.
    """

    def __init__(self, *filters: Filter):
        """
        Initialize the All filter with a variable number of filters.

        Args:
            *filters (Filter): Variable number of filters to be combined.
        """
        self.predicates = tuple(
            to_predicate(collection_filter) for collection_filter in filters
        )

    def __call__(self, path: PathParts, x: typing.Any) -> bool:
        """
        Apply the composite filter to the given path and object.

        Args:
            path (PathParts): The path to the current object.
            x (typing.Any): The object to be filtered.

        Returns:
            bool: True if all of the constituent predicates return True, False otherwise.
        """
        return all(predicate(path, x) for predicate in self.predicates)

    def __repr__(self) -> str:
        """
        Return a string representation of the All filter.

        Returns:
            str: A string representation of the All filter, including its predicates.
        """
        return f'All({", ".join(map(repr, self.predicates))})'

    def __eq__(self, other) -> bool:
        """
        Check if this All filter is equal to another object.

        Args:
            other: The object to compare with.

        Returns:
            bool: True if the other object is an All filter with the same predicates, False otherwise.
        """
        return isinstance(other, All) and self.predicates == other.predicates

    def __hash__(self) -> int:
        """
        Compute the hash value for this All filter.

        Returns:
            int: The hash value of the predicates tuple.
        """
        return hash(self.predicates)


class Not:
    """
    A filter class that negates the result of another filter.

    This class creates a new filter that returns the opposite boolean value
    of the filter it wraps.

    Attributes:
        predicate (Predicate): The predicate function converted from the input filter.
    """

    def __init__(self, collection_filter: Filter, /):
        """
        Initialize the Not filter with another filter.

        Args:
            collection_filter (Filter): The filter to be negated.
        """
        self.predicate = to_predicate(collection_filter)

    def __call__(self, path: PathParts, x: typing.Any) -> bool:
        """
        Apply the negated filter to the given path and object.

        Args:
            path (PathParts): The path to the current object.
            x (typing.Any): The object to be filtered.

        Returns:
            bool: The negation of the result from the wrapped predicate.
        """
        return not self.predicate(path, x)

    def __repr__(self) -> str:
        """
        Return a string representation of the Not filter.

        Returns:
            str: A string representation of the Not filter, including its predicate.
        """
        return f'Not({self.predicate!r})'

    def __eq__(self, other) -> bool:
        """
        Check if this Not filter is equal to another object.

        Args:
            other: The object to compare with.

        Returns:
            bool: True if the other object is a Not filter with the same predicate, False otherwise.
        """
        return isinstance(other, Not) and self.predicate == other.predicate

    def __hash__(self) -> int:
        """
        Compute the hash value for this Not filter.

        Returns:
            int: The hash value of the predicate.
        """
        return hash(self.predicate)


class Everything:
    """
    Filter that matches all objects.

    This filter always returns True, effectively disabling filtering.
    It's useful as a default filter or when you want to select everything
    in a structure.

    Examples
    --------
    .. code-block:: python

        >>> from brainstate.util.filter import Everything
        >>>
        >>> # Create a filter that matches everything
        >>> all_filter = Everything()
        >>>
        >>> # Always returns True
        >>> all_filter([], 'any_object')
        True
        >>> all_filter(['some', 'path'], 42)
        True
        >>> all_filter([], None)
        True
        >>>
        >>> # Useful as a default filter
        >>> def process_data(data, filter=None):
        ...     if filter is None:
        ...         filter = Everything()
        ...     # Process all data when no specific filter is provided

    See Also
    --------
    Nothing : Filter that matches no objects
    to_predicate : Convert True to Everything filter

    Notes
    -----
    This filter is equivalent to using ``to_predicate(True)`` or
    ``to_predicate(...)`` (Ellipsis).
    """

    def __call__(self, path: PathParts, x: typing.Any) -> bool:
        """
        Always return True.

        Parameters
        ----------
        path : PathParts
            The path to the current object (ignored).
        x : Any
            The object to be filtered (ignored).

        Returns
        -------
        bool
            Always returns True.
        """
        return True

    def __repr__(self) -> str:
        """
        Return a string representation of the Everything filter.

        Returns:
            str: The string 'Everything()'.
        """
        return 'Everything()'

    def __eq__(self, other) -> bool:
        """
        Check if this Everything filter is equal to another object.

        Args:
            other: The object to compare with.

        Returns:
            bool: True if the other object is an instance of Everything, False otherwise.
        """
        return isinstance(other, Everything)

    def __hash__(self) -> int:
        """
        Compute the hash value for this Everything filter.

        Returns:
            int: The hash value of the Everything class.
        """
        return hash(Everything)


class Nothing:
    """
    Filter that matches no objects.

    This filter always returns False, effectively filtering out all objects.
    It's useful for disabling selection or creating empty filter results.

    Examples
    --------
    .. code-block:: python

        >>> from brainstate.util.filter import Nothing
        >>>
        >>> # Create a filter that matches nothing
        >>> none_filter = Nothing()
        >>>
        >>> # Always returns False
        >>> none_filter([], 'any_object')
        False
        >>> none_filter(['some', 'path'], 42)
        False
        >>> none_filter([], None)
        False
        >>>
        >>> # Useful for conditional filtering
        >>> def get_params(model, include_frozen=False):
        ...     if include_frozen:
        ...         filter = Everything()
        ...     else:
        ...         filter = Nothing()  # Exclude all frozen params
        ...     # Apply filter to model parameters

    See Also
    --------
    Everything : Filter that matches all objects
    to_predicate : Convert False or None to Nothing filter

    Notes
    -----
    This filter is equivalent to using ``to_predicate(False)`` or
    ``to_predicate(None)``.
    """

    def __call__(self, path: PathParts, x: typing.Any) -> bool:
        """
        Always return False.

        Parameters
        ----------
        path : PathParts
            The path to the current object (ignored).
        x : Any
            The object to be filtered (ignored).

        Returns
        -------
        bool
            Always returns False.
        """
        return False

    def __repr__(self) -> str:
        """
        Return a string representation of the Nothing filter.

        Returns:
            str: The string 'Nothing()'.
        """
        return 'Nothing()'

    def __eq__(self, other) -> bool:
        """
        Check if this Nothing filter is equal to another object.

        Args:
            other: The object to compare with.

        Returns:
            bool: True if the other object is an instance of Nothing, False otherwise.
        """
        return isinstance(other, Nothing)

    def __hash__(self) -> int:
        """
        Compute the hash value for this Nothing filter.

        Returns:
            int: The hash value of the Nothing class.
        """
        return hash(Nothing)
