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
    Converts a Filter to a predicate function.

    This function takes various types of filters and converts them into
    corresponding predicate functions that can be used for filtering.

    Args:
        the_filter (Filter): The filter to be converted. Can be of various types:
            - str: Converted to a WithTag filter.
            - type: Converted to an OfType filter.
            - bool: True becomes Everything(), False becomes Nothing().
            - Ellipsis: Converted to Everything().
            - None: Converted to Nothing().
            - callable: Returned as-is.
            - list or tuple: Converted to Any filter with elements as arguments.

    Returns:
        Predicate: A callable predicate function that can be used for filtering.

    Raises:
        TypeError: If the input filter is of an invalid type.
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
    A filter class that checks if an object has a specific tag.

    This class is a callable that can be used as a predicate function
    to filter objects based on their 'tag' attribute.

    Attributes:
        tag (str): The tag to match against.
    """

    tag: str

    def __call__(self, path: PathParts, x: typing.Any) -> bool:
        """
        Check if the object has a 'tag' attribute matching the specified tag.

        Args:
            path (PathParts): The path to the current object (not used in this filter).
            x (typing.Any): The object to check for the tag.

        Returns:
            bool: True if the object has a 'tag' attribute matching the specified tag, False otherwise.
        """
        return hasattr(x, 'tag') and x.tag == self.tag

    def __repr__(self) -> str:
        return f'WithTag({self.tag!r})'


@dataclasses.dataclass(frozen=True)
class PathContains:
    """
    A filter class that checks if a given key is present in the path.

    This class is a callable that can be used as a predicate function
    to filter objects based on whether a specific key is present in their path.

    Attributes:
        key (Key): The key to search for in the path.
    """

    key: Key

    def __call__(self, path: PathParts, x: typing.Any) -> bool:
        """
        Check if the key is present in the given path.

        Args:
            path (PathParts): The path to check for the presence of the key.
            x (typing.Any): The object associated with the path (not used in this filter).

        Returns:
            bool: True if the key is present in the path, False otherwise.
        """
        return self.key in path

    def __repr__(self) -> str:
        return f'PathContains({self.key!r})'


@dataclasses.dataclass(frozen=True)
class OfType:
    """
    A filter class that checks if an object is of a specific type.

    This class is a callable that can be used as a predicate function
    to filter objects based on their type.

    Attributes:
        type (type): The type to match against.
    """
    type: type

    def __call__(self, path: PathParts, x: typing.Any):
        return isinstance(x, self.type) or (
            hasattr(x, 'type') and issubclass(x.type, self.type)
        )

    def __repr__(self):
        return f'OfType({self.type!r})'


class Any:
    """
    A filter class that combines multiple filters using a logical OR operation.

    This class creates a composite filter that returns True if any of its
    constituent filters return True.

    Attributes:
        predicates (tuple): A tuple of predicate functions converted from the input filters.
    """

    def __init__(self, *filters: Filter):
        """
        Initialize the Any filter with a variable number of filters.

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
    A filter class that always returns True for any input.

    This class represents a filter that matches everything, effectively
    allowing all objects to pass through without any filtering.
    """

    def __call__(self, path: PathParts, x: typing.Any) -> bool:
        """
        Always return True, regardless of the input.

        Args:
            path (PathParts): The path to the current object (not used).
            x (typing.Any): The object to be filtered (not used).

        Returns:
            bool: Always returns True.
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
    A filter class that always returns False for any input.

    This class represents a filter that matches nothing, effectively
    filtering out all objects.
    """

    def __call__(self, path: PathParts, x: typing.Any) -> bool:
        """
        Always return False, regardless of the input.

        Args:
            path (PathParts): The path to the current object (not used).
            x (typing.Any): The object to be filtered (not used).

        Returns:
            bool: Always returns False.
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
