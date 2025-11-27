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

import functools
from typing import Sequence, Union, Callable, Optional

from brainstate._state import TRACE_CONTEXT
from ._jit import jit, JittedFunction

__all__ = [
    'jit_named_scope',
]


def fn_to_call(
    fn: Callable,
    name: str,
    static_argnums: Optional[Union[int, Sequence[int], Callable]] = None,
    static_argnames: Optional[Union[str, Sequence[str], Callable]] = None,
) -> Callable:
    """
    A wrapper for JIT-compiled functions with named scopes.

    This function provides a consistent interface for JIT-compiled functions,
    supporting both standard ``static_*`` arguments and the inverse ``non_static_*``
    arguments for convenience.

    When using ``non_static_*`` parameters, this function automatically computes
    which arguments should be static based on the actual arguments provided
    at call time, and caches the JIT-compiled functions for efficiency.

    Unlike a class-based implementation, the returned wrapper function properly
    supports being used as a bound method through Python's descriptor protocol.

    Parameters
    ----------
    fn : Callable
        The original unwrapped function.
    name : str
        The name assigned to the function for JAX traces/profiles.
    static_argnums : int or sequence of int, optional, Callable
        Positional argument indices treated as static.
    static_argnames : str or sequence of str, optional, Callable
        Keyword argument names treated as static.

    Returns
    -------
    Callable
        A wrapped function that JIT-compiles with the specified configuration.
    """

    def _normalize_argnums(argnums: Union[int, Sequence[int]], n_args: int) -> tuple:
        """Normalize argument indices to a tuple, handling negative indices."""
        if argnums is None:
            return ()
        if isinstance(argnums, int):
            argnums = (argnums,)
        else:
            argnums = tuple(argnums)
        return tuple(i + n_args if i < 0 else i for i in argnums)

    def _normalize_argnames(argnames: Union[str, Sequence[str]]) -> tuple:
        """Normalize argument names to a tuple."""
        if argnames is None:
            return ()
        if isinstance(argnames, str):
            return (argnames,)
        return tuple(argnames)

    def _create_jit_fn(
        s_argnums: Union[int, Sequence[int], None],
        s_argnames: Union[str, Sequence[str], None],
    ) -> JittedFunction:
        """Create a JIT-compiled function with the given static arguments."""
        return jit(
            fn,
            name=name,
            static_argnums=s_argnums,
            static_argnames=s_argnames,
        )

    def _get_jit_fn(*args, **kwargs) -> JittedFunction:
        """Get or create the appropriate JIT function for the given arguments."""

        s_argnums = static_argnums(*args, **kwargs) if callable(static_argnums) else static_argnums
        s_argnames = static_argnames(*args, **kwargs) if callable(static_argnames) else static_argnames
        s_argnums = _normalize_argnums(s_argnums, len(args))
        s_argnames = _normalize_argnames(s_argnames)

        return _create_jit_fn(s_argnums, s_argnames)

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        """Call the JIT-compiled function."""
        if TRACE_CONTEXT.get_trace_stack_level() > 0:
            jit_fn = _get_jit_fn(*args, **kwargs)
            return jit_fn(*args, **kwargs)
        else:
            return fn(*args, **kwargs)

    wrapper.fn = fn
    return wrapper


def jit_named_scope(
    name: str,
    static_argnums: Optional[Union[int, Sequence[int], Callable]] = None,
    static_argnames: Optional[Union[str, Sequence[str], Callable]] = None,
) -> Callable[[Callable], Callable]:
    """
    Decorator that wraps a function with JAX's JIT compilation and sets its name.

    This is a convenience decorator that combines ``jit()`` with named scope support.
    It also provides an inverse API via ``non_static_argnums``/``non_static_argnames``
    for specifying which arguments should NOT be static (the complement of ``static_*``).

    The decorated function supports being used as a class bound method.

    Parameters
    ----------
    name : str
        Name to set for the function. This name appears in JAX traces and profiles,
        making debugging and performance analysis easier.
    static_argnums : int or sequence of int, optional, Callable
        Positional argument indices to treat as static (compile-time constant).
    static_argnames : str or sequence of str, optional, Callable
        Keyword argument names to treat as static (compile-time constant).

    Returns
    -------
    Callable[[Callable], Callable]
        A decorator that returns a wrapped callable function.

    Examples
    --------
    Basic usage with just a name:

    >>> @jit_named_scope(name='my_layer')
    ... def layer(x, w):
    ...     return x @ w

    With static arguments:

    >>> @jit_named_scope(name='power_fn', static_argnums=1)
    ... def power(x, n):
    ...     return x ** n

    Using non_static_argnums (only first arg is traced, rest are static):

    >>> @jit_named_scope(name='scaled_power', non_static_argnums=0)
    ... def scaled_power(x, n, scale):
    ...     return (x ** n) * scale  # n and scale are automatically static

    As a class method:

    >>> class MyModule:
    ...     def __init__(self, scale):
    ...         self.scale = scale
    ...
    ...     @jit_named_scope(name='compute')
    ...     def compute(self, x):
    ...         return x * self.scale
    """

    def decorator(fn: Callable) -> Callable:
        return fn_to_call(
            fn=fn,
            name=name,
            static_argnums=static_argnums,
            static_argnames=static_argnames,
        )

    return decorator
