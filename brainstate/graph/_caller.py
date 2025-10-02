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
Delayed accessor and callable proxy utilities for graph operations.

This module provides utilities for creating delayed accessors and callable proxies
that can be used to defer attribute and item access until a later time. This is
particularly useful in graph operations where the actual values are not known
until execution time.
"""

import dataclasses
from typing import Any, TypeVar, Protocol, Generic, Union, Optional

import jax

__all__ = [
    'DelayedAccessor',
    'CallableProxy',
    'ApplyCaller',
]

A = TypeVar('A', covariant=True)  # type: ignore[not-supported-yet]


def _identity(x: Any) -> Any:
    """Identity function that returns its input unchanged."""
    return x


@dataclasses.dataclass(frozen=True)
class GetItem:
    """Represents a deferred item access operation (e.g., obj[key])."""
    key: Any


@dataclasses.dataclass(frozen=True)
class GetAttr:
    """Represents a deferred attribute access operation (e.g., obj.attr)."""
    name: str


@dataclasses.dataclass(frozen=True)
class DelayedAccessor:
    """
    A delayed accessor that records a sequence of attribute and item accesses
    to be applied later.

    This class allows building up a chain of attribute and item accesses that
    can be applied to an object at a later time. This is useful for deferring
    access operations in graph computations.

    Attributes
    ----------
    actions : tuple[Union[GetItem, GetAttr], ...]
        A tuple of GetItem or GetAttr operations to be applied in sequence.

    Examples
    --------
    .. code-block:: python

        >>> accessor = DelayedAccessor()
        >>> accessor = accessor.foo.bar[0]
        >>> # Later apply to an actual object
        >>> result = accessor(my_object)  # Equivalent to my_object.foo.bar[0]
    """
    actions: tuple[Union[GetItem, GetAttr], ...] = ()

    def __call__(self, x: Any) -> Any:
        """
        Apply the recorded access operations to the given object.

        Parameters
        ----------
        x : Any
            The object to apply the access operations to.

        Returns
        -------
        Any
            The result after applying all recorded operations.

        Raises
        ------
        TypeError
            If an unexpected action type is encountered.
        """
        for action in self.actions:
            if isinstance(action, GetItem):
                x = x[action.key]
            elif isinstance(action, GetAttr):
                x = getattr(x, action.name)
            else:
                raise TypeError(f"Unexpected action type: {type(action)}")
        return x

    def __getattr__(self, name: str) -> 'DelayedAccessor':
        """
        Record an attribute access operation.

        Parameters
        ----------
        name : str
            The name of the attribute to access.

        Returns
        -------
        DelayedAccessor
            A new DelayedAccessor with the attribute access recorded.
        """
        return DelayedAccessor(self.actions + (GetAttr(name),))

    def __getitem__(self, key: Any) -> 'DelayedAccessor':
        """
        Record an item access operation.

        Parameters
        ----------
        key : Any
            The key to use for item access.

        Returns
        -------
        DelayedAccessor
            A new DelayedAccessor with the item access recorded.
        """
        return DelayedAccessor(self.actions + (GetItem(key),))


jax.tree_util.register_static(DelayedAccessor)


class _AccessorCall(Protocol):
    """Protocol for callable objects that accept a DelayedAccessor."""

    def __call__(self, accessor: DelayedAccessor, /, *args, **kwargs) -> Any:
        ...


class CallableProxy:
    """
    A proxy object that wraps a callable and applies delayed access operations.

    This class allows creating a callable object that can have attribute and item
    accesses chained on it, which will be applied to the result when called.

    Attributes
    ----------
    _callable : _AccessorCall
        The underlying callable to invoke.
    _accessor : DelayedAccessor
        The DelayedAccessor that records access operations.

    Examples
    --------
    .. code-block:: python

        >>> def my_func(accessor, x):
        ...     return accessor(x)
        >>> proxy = CallableProxy(my_func)
        >>> result = proxy.foo.bar(my_object)  # Applies my_object.foo.bar
    """

    def __init__(
        self,
        fun: _AccessorCall,
        accessor: Optional[DelayedAccessor] = None
    ) -> None:
        """
        Initialize the CallableProxy.

        Parameters
        ----------
        fun : _AccessorCall
            The callable that accepts a DelayedAccessor as its first argument.
        accessor : Optional[DelayedAccessor], optional
            Optional DelayedAccessor to use. If None, creates a new one.
        """
        self._callable = fun
        self._accessor = DelayedAccessor() if accessor is None else accessor

    def __call__(self, *args, **kwargs) -> Any:
        """
        Invoke the callable with the recorded accessor and given arguments.

        Parameters
        ----------
        *args
            Positional arguments to pass to the callable.
        **kwargs
            Keyword arguments to pass to the callable.

        Returns
        -------
        Any
            The result of calling the wrapped callable.
        """
        return self._callable(self._accessor, *args, **kwargs)

    def __getattr__(self, name: str) -> 'CallableProxy':
        """
        Create a new proxy with an additional attribute access.

        Parameters
        ----------
        name : str
            The name of the attribute to access.

        Returns
        -------
        CallableProxy
            A new CallableProxy with the attribute access recorded.
        """
        return CallableProxy(self._callable, getattr(self._accessor, name))

    def __getitem__(self, key: Any) -> 'CallableProxy':
        """
        Create a new proxy with an additional item access.

        Parameters
        ----------
        key : Any
            The key to use for item access.

        Returns
        -------
        CallableProxy
            A new CallableProxy with the item access recorded.
        """
        return CallableProxy(self._callable, self._accessor[key])


class ApplyCaller(Protocol, Generic[A]):
    """
    Protocol for objects that support chained attribute/item access and calling.

    This protocol defines the interface for objects that can have attributes
    and items accessed on them, and can be called to produce a result along
    with some additional value of type A.

    Parameters
    ----------
    A : TypeVar
        The type of the second element in the returned tuple when called.
    """

    def __getattr__(self, __name: str) -> 'ApplyCaller[A]':
        """
        Get an attribute, returning another ApplyCaller.

        Parameters
        ----------
        __name : str
            The name of the attribute to access.

        Returns
        -------
        ApplyCaller[A]
            A new ApplyCaller instance.
        """
        ...

    def __getitem__(self, __key: Any) -> 'ApplyCaller[A]':
        """
        Get an item, returning another ApplyCaller.

        Parameters
        ----------
        __key : Any
            The key to use for item access.

        Returns
        -------
        ApplyCaller[A]
            A new ApplyCaller instance.
        """
        ...

    def __call__(self, *args, **kwargs) -> tuple[Any, A]:
        """
        Call the object, returning a tuple of the result and additional value.

        Parameters
        ----------
        *args
            Positional arguments for the call.
        **kwargs
            Keyword arguments for the call.

        Returns
        -------
        tuple[Any, A]
            A tuple containing the result and an additional value of type A.
        """
        ...
