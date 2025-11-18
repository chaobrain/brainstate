# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
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

# -*- coding: utf-8 -*-

"""
Compatibility layer for JAX version differences.

This module provides a compatibility layer to handle differences between various
versions of JAX, ensuring that BrainState works correctly across different JAX
versions. It imports the appropriate modules and functions based on the detected
JAX version and provides fallback implementations when necessary.

Key Features:
    - Version-aware imports for JAX core functionality
    - Compatibility wrappers for changed APIs
    - Fallback implementations for deprecated functions
    - Type-safe utility functions

Examples:
    Basic usage:

    >>> from brainstate._compatible_import import safe_map, safe_zip
    >>> result = safe_map(lambda x: x * 2, [1, 2, 3])
    >>> pairs = safe_zip([1, 2, 3], ['a', 'b', 'c'])

    Using JAX core types:

    >>> from brainstate._compatible_import import Primitive, ClosedJaxpr
    >>> # These imports work across different JAX versions
"""

from contextlib import contextmanager
from functools import partial
from typing import Iterable, Hashable, TypeVar, Callable

import jax
from jax.core import get_aval, Tracer
from saiunit._compatible_import import wrap_init

__all__ = [
    'ClosedJaxpr',
    'DropVar',
    'Primitive',
    'extend_axis_env_nd',
    'jaxpr_as_fun',
    'get_aval',
    'Tracer',
    'to_concrete_aval',
    'safe_map',
    'safe_zip',
    'unzip2',
    'wraps',
    'Device',
    'wrap_init',
    'Var',
    'JaxprEqn',
    'Jaxpr',
    'Literal',
    'make_iota',
    'to_elt',
    'BatchTracer',
    'BatchTrace',
    'is_jit_primitive',
]

T = TypeVar("T")
T1 = TypeVar("T1")
T2 = TypeVar("T2")
T3 = TypeVar("T3")

if jax.__version_info__ < (0, 5, 0):
    from jax.lib.xla_client import Device
else:
    from jax import Device

if jax.__version_info__ < (0, 7, 1):
    from jax.interpreters.batching import make_iota, to_elt, BatchTracer, BatchTrace
else:
    from jax._src.interpreters.batching import make_iota, to_elt, BatchTracer, BatchTrace

from jax.core import DropVar

if jax.__version_info__ < (0, 4, 38):
    from jax.core import ClosedJaxpr, extend_axis_env_nd, Primitive, jaxpr_as_fun
    from jax.core import Primitive, Var, JaxprEqn, Jaxpr, ClosedJaxpr, Literal
else:
    from jax.extend.core import ClosedJaxpr, Primitive, jaxpr_as_fun
    from jax.extend.core import Primitive, Var, JaxprEqn, Jaxpr, ClosedJaxpr, Literal
    from jax.core import trace_ctx


    @contextmanager
    def extend_axis_env_nd(name_size_pairs: Iterable[tuple[Hashable, int]]):
        """
        Context manager to temporarily extend the JAX axis environment.

        Extends the current JAX axis environment with new named axes for
        vectorized computations, then restores the previous environment.

        Args:
            name_size_pairs: Iterable of (name, size) tuples specifying
                           the named axes to add to the environment.

        Yields:
            None: Context with extended axis environment.

        Examples:
            >>> with extend_axis_env_nd([('batch', 32), ('seq', 128)]):
            ...     # Code using vectorized operations with named axes
            ...     pass
        """
        prev = trace_ctx.axis_env
        try:
            trace_ctx.set_axis_env(prev.extend_pure(name_size_pairs))
            yield
        finally:
            trace_ctx.set_axis_env(prev)

if jax.__version_info__ < (0, 6, 0):
    from jax.util import safe_map, safe_zip, unzip2, wraps

else:
    def safe_map(f, *args):
        """
        Map a function over multiple sequences with length checking.

        Applies a function to corresponding elements from multiple sequences,
        ensuring all sequences have the same length.

        Args:
            f: Function to apply to elements from each sequence.
            *args: Variable number of sequences to map over.

        Returns:
            list: Results of applying f to corresponding elements.

        Raises:
            AssertionError: If input sequences have different lengths.

        Examples:
            >>> safe_map(lambda x, y: x + y, [1, 2, 3], [4, 5, 6])
            [5, 7, 9]

            >>> safe_map(str.upper, ['a', 'b', 'c'])
            ['A', 'B', 'C']
        """
        args = list(map(list, args))
        n = len(args[0])
        for arg in args[1:]:
            assert len(arg) == n, f'length mismatch: {list(map(len, args))}'
        return list(map(f, *args))


    def safe_zip(*args):
        """
        Zip multiple sequences with length checking.

        Combines corresponding elements from multiple sequences into tuples,
        ensuring all sequences have the same length.

        Args:
            *args: Variable number of sequences to zip together.

        Returns:
            list: List of tuples containing corresponding elements.

        Raises:
            AssertionError: If input sequences have different lengths.

        Examples:
            >>> safe_zip([1, 2, 3], ['a', 'b', 'c'])
            [(1, 'a'), (2, 'b'), (3, 'c')]

            >>> safe_zip([1, 2], [3, 4], [5, 6])
            [(1, 3, 5), (2, 4, 6)]
        """
        args = list(map(list, args))
        n = len(args[0])
        for arg in args[1:]:
            assert len(arg) == n, f'length mismatch: {list(map(len, args))}'
        return list(zip(*args))


    def unzip2(xys: Iterable[tuple[T1, T2]]) -> tuple[tuple[T1, ...], tuple[T2, ...]]:
        """
        Unzip sequence of length-2 tuples into two tuples.

        Takes an iterable of 2-tuples and separates them into two tuples
        containing the first and second elements respectively.

        Args:
            xys: Iterable of 2-tuples to unzip.

        Returns:
            tuple: A 2-tuple containing:
                - Tuple of all first elements
                - Tuple of all second elements

        Examples:
            >>> pairs = [(1, 'a'), (2, 'b'), (3, 'c')]
            >>> nums, letters = unzip2(pairs)
            >>> nums
            (1, 2, 3)
            >>> letters
            ('a', 'b', 'c')

        Notes:
            We deliberately don't use zip(*xys) because it is lazily evaluated,
            is too permissive about inputs, and does not guarantee a length-2 output.
        """
        # Note: we deliberately don't use zip(*xys) because it is lazily evaluated,
        # is too permissive about inputs, and does not guarantee a length-2 output.
        xs: list[T1] = []
        ys: list[T2] = []
        for x, y in xys:
            xs.append(x)
            ys.append(y)
        return tuple(xs), tuple(ys)


    def fun_name(fun: Callable):
        """
        Extract the name of a function, handling special cases.

        Attempts to get the name of a function, with special handling for
        partial functions and fallback for unnamed functions.

        Args:
            fun: The function to get the name from.

        Returns:
            str: The function name, or "<unnamed function>" if no name available.

        Examples:
            >>> def my_function():
            ...     pass
            >>> fun_name(my_function)
            'my_function'

            >>> from functools import partial
            >>> add = lambda x, y: x + y
            >>> add_one = partial(add, 1)
            >>> fun_name(add_one)
            '<lambda>'
        """
        name = getattr(fun, "__name__", None)
        if name is not None:
            return name
        if isinstance(fun, partial):
            return fun_name(fun.func)
        else:
            return "<unnamed function>"


    def wraps(
        wrapped: Callable,
        namestr: str | None = None,
        docstr: str | None = None,
        **kwargs,
    ) -> Callable[[T], T]:
        """
        Enhanced function wrapper with fine-grained control.

        Like functools.wraps, but provides more control over the name and docstring
        of the resulting function. Useful for creating custom decorators.

        Args:
            wrapped: The function being wrapped.
            namestr: Optional format string for the wrapper function name.
                    Can use {fun} placeholder for the original function name.
            docstr: Optional format string for the wrapper function docstring.
                   Can use {fun}, {doc}, and other kwargs as placeholders.
            **kwargs: Additional keyword arguments for format string substitution.

        Returns:
            Callable: A decorator function that applies the wrapping.

        Examples:
            >>> def my_decorator(func):
            ...     @wraps(func, namestr="decorated_{fun}")
            ...     def wrapper(*args, **kwargs):
            ...         return func(*args, **kwargs)
            ...     return wrapper

            >>> @my_decorator
            ... def example():
            ...     pass
            >>> example.__name__
            'decorated_example'
        """

        def wrapper(fun: T) -> T:
            try:
                name = fun_name(wrapped)
                doc = getattr(wrapped, "__doc__", "") or ""
                fun.__dict__.update(getattr(wrapped, "__dict__", {}))
                fun.__annotations__ = getattr(wrapped, "__annotations__", {})
                fun.__name__ = name if namestr is None else namestr.format(fun=name)
                fun.__module__ = getattr(wrapped, "__module__", "<unknown module>")
                fun.__doc__ = (doc if docstr is None
                               else docstr.format(fun=name, doc=doc, **kwargs))
                fun.__qualname__ = getattr(wrapped, "__qualname__", fun.__name__)
                fun.__wrapped__ = wrapped
            except Exception:
                pass
            return fun

        return wrapper


def to_concrete_aval(aval):
    """
    Convert an abstract value to its concrete representation.

    Takes an abstract value and attempts to convert it to a concrete value,
    handling JAX Tracer objects appropriately.

    Args:
        aval: The abstract value to convert.

    Returns:
        The concrete value representation, or the original aval if already concrete.

    Examples:
        >>> import jax.numpy as jnp
        >>> arr = jnp.array([1, 2, 3])
        >>> concrete = to_concrete_aval(arr)
        # Returns the concrete array value
    """
    aval = get_aval(aval)
    if isinstance(aval, Tracer):
        return aval.to_concrete_value()
    return aval


def is_jit_primitive(eqn: JaxprEqn) -> bool:
    assert isinstance(eqn, JaxprEqn)
    if jax.__version_info__ < (0, 7, 0):
        return eqn.primitive.name in ['pjit', 'xla_call']
    else:
        return eqn.primitive.name in ['jit', 'xla_call']
