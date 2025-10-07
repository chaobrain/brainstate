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
This module implements how to create a JAX Jaxpr from a given function by considering the states that are read and
written by the function. These state transformations are foundational for the BrainCore library. These utilities
include two basic functions: `StatefulFunction` and `make_jaxpr`.


``StatefulFunction``
--------------------

The module provides a class called ``StatefulFunction`` that wraps a function and provides methods to get the
JAX Jaxpr, the output shapes, the states that are read and written by the function, and the output of the function.
The class provides the following methods:

- `make_jaxpr`: creates the JAX Jaxpr of the function.
- `jaxpr_call`: calls the function at the JAX Jaxpr level.
- `jaxpr_call_without_states`: calls the function at the JAX Jaxpr level without considering the states.
- `get_states`: returns the states that are read and written by the function.
- `get_read_states`: returns the states that are read by the function.
- `get_write_states`: returns the states that are written by the function.
- `get_static_args`: returns the static arguments from the arguments.
- `compile_and_get_states_by_static_args`: compiles the function and returns the states that are read and
   written by the function.
- `get_jaxpr`: returns the JAX Jaxpr of the function.
- `get_out_shapes`: returns the output shapes of the function.
- `get_out_treedef`: returns the output tree of the function.

``make_jaxpr``
--------------

The module provides a function called `make_jaxpr` that creates a function that produces its JAX Jaxpr given example
arguments. The function returns a wrapped version of the function that when applied to example arguments returns a
`ClosedJaxpr` representation of the function on those arguments. If the argument `return_shape` is `True`, then the
returned function instead returns a pair where the first element is the `ClosedJaxpr` representation of the function
and the second element is a pytree representing the structure, shape, dtypes, and named shapes of the output of the
function.

"""

import functools
import inspect
import operator
import threading
from collections import OrderedDict, defaultdict
from collections.abc import Hashable, Iterable, Sequence
from collections.abc import MutableSet
from contextlib import ExitStack
from typing import Any, Callable, Dict, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax._src import source_info_util
from jax._src.linear_util import annotate
from jax._src.traceback_util import api_boundary
from jax._src.util import memoize
from jax.api_util import shaped_abstractify
from jax.extend.linear_util import transformation_with_aux
from jax.interpreters import partial_eval as pe

from brainstate._compatible_import import (
    ClosedJaxpr, extend_axis_env_nd, safe_map, safe_zip, unzip2, wraps, wrap_init,
    Literal, Var, Jaxpr, make_iota, to_elt, BatchTracer, BatchTrace,
)
from brainstate._state import State, StateTraceStack
from brainstate._utils import set_module_as
from brainstate.random import RandomState
from brainstate.typing import Filter, PyTree
from brainstate.util import PrettyObject
from brainstate.util.filter import to_predicate

AxisName = Hashable

__all__ = [
    "StatefulFunction",
    "make_jaxpr",
    "StatefulMapping",
]


class hashabledict(dict):
    def __hash__(self):
        return hash(tuple(sorted(self.items())))


class _BoundedCache:
    """
    A thread-safe LRU cache with bounded size.

    This cache stores a limited number of items and evicts the least recently used item
    when the cache reaches its maximum size. All operations are thread-safe.

    Parameters
    ----------
    maxsize : int, default 128
        Maximum number of items to store in the cache.
    """

    def __init__(self, maxsize: int = 128):
        self._cache = OrderedDict()
        self._maxsize = maxsize
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    def get(
        self,
        key: Any,
        default: Any = None,
        raise_on_miss: bool = False,
        error_context: str = "item"
    ) -> Any:
        """
        Get an item from the cache.

        Parameters
        ----------
        key : Any
            The cache key.
        default : Any, optional
            The default value to return if the key is not found.
        raise_on_miss : bool, optional
            If True, raise a detailed ValueError when the key is not found.
        error_context : str, optional
            Context description for the error message (e.g., "Function", "JAX expression").

        Returns
        -------
        Any
            The cached value or the default value.

        Raises
        ------
        ValueError
            If raise_on_miss is True and the key is not found.
        """
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key]
            self._misses += 1

            if raise_on_miss:
                available_keys = list(self._cache.keys())
                error_msg = [
                    f"{error_context} not compiled for the requested cache key.",
                    f"",
                    f"Requested key:",
                    f"  {key}",
                    f"",
                    f"Available {{len(available_keys)}} keys:",
                ]
                if available_keys:
                    for i, k in enumerate(available_keys, 1):
                        error_msg.append(f"  [{i}] {k}")
                else:
                    error_msg.append("  (none - not compiled yet)")
                error_msg.append("")
                error_msg.append("Call make_jaxpr() first with matching arguments.")
                raise ValueError("\n".join(error_msg))

            return default

    def set(self, key: Any, value: Any) -> None:
        """
        Set an item in the cache.

        Parameters
        ----------
        key : Any
            The cache key.
        value : Any
            The value to cache.

        Raises
        ------
        ValueError
            If the key already exists in the cache.
        """
        with self._lock:
            if key in self._cache:
                raise ValueError(
                    f"Cache key already exists: {key}. "
                    f"Cannot overwrite existing cached value. "
                    f"Clear the cache first if you need to recompile."
                )
            if len(self._cache) >= self._maxsize:
                self._cache.popitem(last=False)
            self._cache[key] = value

    def pop(self, key: Any, default: Any = None) -> Any:
        """
        Remove and return an item from the cache.

        Parameters
        ----------
        key : Any
            The cache key to remove.
        default : Any, optional
            The default value to return if the key is not found.

        Returns
        -------
        Any
            The cached value or the default value if the key is not found.
        """
        with self._lock:
            if key in self._cache:
                return self._cache.pop(key)
            return default

    def replace(self, key: Any, value: Any) -> None:
        """
        Replace an existing item in the cache.

        Parameters
        ----------
        key : Any
            The cache key to replace.
        value : Any
            The new value to cache.

        Raises
        ------
        KeyError
            If the key does not exist in the cache.
        """
        with self._lock:
            if key not in self._cache:
                raise KeyError(
                    f"Cache key does not exist: {key}. "
                    f"Cannot replace non-existent cached value. "
                    f"Use set() to add a new cache entry."
                )
            self._cache[key] = value
            self._cache.move_to_end(key)

    def __contains__(self, key: Any) -> bool:
        """
        Check if a key exists in the cache.

        Parameters
        ----------
        key : Any
            The cache key to check.

        Returns
        -------
        bool
            True if the key exists in the cache, False otherwise.
        """
        with self._lock:
            return key in self._cache

    def __len__(self) -> int:
        """
        Get the number of items in the cache.

        Returns
        -------
        int
            The number of items currently in the cache.
        """
        with self._lock:
            return len(self._cache)

    def clear(self) -> None:
        """
        Clear all items from the cache and reset statistics.

        This method removes all cached items and resets hit/miss counters to zero.
        """
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def keys(self):
        """
        Return all keys in the cache.

        Returns
        -------
        list
            A list of all keys currently in the cache.
        """
        with self._lock:
            return list(self._cache.keys())

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns
        -------
        dict
            A dictionary with cache statistics including:

            - 'size': Current number of items in cache
            - 'maxsize': Maximum cache size
            - 'hits': Number of cache hits
            - 'misses': Number of cache misses
            - 'hit_rate': Hit rate percentage (0-100)
        """
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total * 100) if total > 0 else 0.0
            return {
                'size': len(self._cache),
                'maxsize': self._maxsize,
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': hit_rate,
            }


def _ensure_str(x: str) -> str:
    if not isinstance(x, str):
        raise TypeError(f"argument is not a string: {x}")
    return x


def _ensure_index_tuple(x: Any) -> tuple[int, ...]:
    """Convert x to a tuple of indices."""
    x = jax.core.concrete_or_error(None, x, "expected a static index or sequence of indices.")
    try:
        return (operator.index(x),)
    except TypeError:
        return tuple(safe_map(operator.index, x))


def _ensure_str_tuple(x: str | Iterable[str]) -> tuple[str, ...]:
    """Convert x to a tuple of strings."""
    if isinstance(x, str):
        return (x,)
    else:
        return tuple(safe_map(_ensure_str, x))


def _jax_v04_new_arg_fn(frame, trace, aval):
    """
    Transform a new argument to a tracer.

    Modified from jax.interpreters.partial_eval.DynamicJaxprTrace.new_arg()

    Args:
      frame: The frame.
      trace: The trace.
      aval: The abstract value.

    Returns:
      The tracer.
    """
    tracer = pe.DynamicJaxprTracer(trace, aval, source_info_util.current())
    frame.tracers.append(tracer)
    frame.tracer_to_var[id(tracer)] = var = frame.newvar(aval)
    frame.invars.append(var)
    return tracer


def _jax_v04_new_jax_trace():
    main = jax.core.thread_local_state.trace_state.trace_stack.stack[-1]
    frame = main.jaxpr_stack[-1]
    trace = pe.DynamicJaxprTrace(main, jax.core.cur_sublevel())
    return frame, trace


class StatefulFunction(PrettyObject):
    """
    A wrapper class for functions that tracks state reads and writes during execution.

    This class wraps a function to enable state management in JAX programs by tracking
    which states are read from and written to during function execution. It provides
    methods to compile the function into JAX's intermediate representation (jaxpr),
    inspect state usage, and execute the function with proper state handling.

    When you define a function:

    .. code-block:: python

        >>> state = brainstate.State(1.)
        >>> def f(x):
        ...     # Your function logic here
        ...     y = x * 2 + state.value
        ...     state.value = y

    Calling ``sf = StatefulFunction(f)`` creates a stateful version of ``f``. You can
    then call it directly with compatibility with JIT:

    .. code-block:: python

        >>> sf = brainstate.transform.StatefulFunction(f)
        >>> out = sf(x)  # Automatically compiles and executes

    Parameters
    ----------
    fun : callable
        The function whose ``jaxpr`` is to be computed. Its positional
        arguments and return value should be arrays, scalars, or standard Python
        containers (tuple/list/dict) thereof.
    static_argnums : int or iterable of int, optional
        Indices of positional arguments to treat as static (known at compile time).
        See :py:func:`jax.jit` for details. Default is ().
    static_argnames : str or iterable of str, optional
        Names of keyword arguments to treat as static (known at compile time).
        See :py:func:`jax.jit` for details. Default is ().
    axis_env : sequence of tuple, optional
        A sequence of pairs where the first element is an axis name and the second
        element is a positive integer representing the size of the mapped axis with
        that name. This parameter is useful when lowering functions that involve
        parallel communication collectives, and it specifies the axis name/size
        environment that would be set up by applications of :py:func:`jax.pmap`.
        Default is None.
    abstracted_axes : pytree, optional
        A pytree with the same structure as the input arguments to ``fun``. The
        leaves of the pytree can be either None or a dict with axis names as keys
        and integers as values. If the leaf is None, then the corresponding axis
        is not abstracted. If the leaf is a dict, then the corresponding axis is
        abstracted, and the dict specifies the axis name and size. The abstracted
        axes are used to infer the input type of the function. If None, then all
        axes are abstracted. Default is None.
    name : str, optional
        Name for the stateful function. Default is None.
    return_only_write : bool, optional
        If True, only return states that were written to during execution
        (not just read). This can reduce memory usage when you only care
        about modified states. Default is True.

    Attributes
    ----------
    fun : callable
        The wrapped function.
    static_argnums : tuple of int
        Indices of static positional arguments.
    static_argnames : tuple of str
        Names of static keyword arguments.
    axis_env : sequence of tuple or None
        Axis environment for parallel operations.
    abstracted_axes : pytree or None
        Abstract axes specification.
    name : str or None
        Name identifier for the function.
    return_only_write : bool
        Whether to return only written states.

    Examples
    --------
    Basic usage with state management:

    .. code-block:: python

        >>> import brainstate
        >>> import jax.numpy as jnp
        >>>
        >>> # Create a state
        >>> state = brainstate.State(jnp.array([1.0, 2.0]))
        >>>
        >>> def f(x):
        ...     state.value += x
        ...     return state.value * 2
        >>>
        >>> # Create a stateful function
        >>> sf = brainstate.transform.StatefulFunction(f)
        >>>
        >>> # Compile and get jaxpr
        >>> x = jnp.array([0.5, 0.5])
        >>> sf.make_jaxpr(x)
        >>>
        >>> # Get states that are read/written
        >>> cache_key = sf.get_arg_cache_key(x)
        >>> states = sf.get_states_by_cache(cache_key)
        >>> read_states = sf.get_read_states_by_cache(cache_key)
        >>> write_states = sf.get_write_states_by_cache(cache_key)

    Using with static arguments:

    .. code-block:: python

        >>> def g(x, n):
        ...     state.value = state.value ** n
        ...     return state.value
        >>>
        >>> sf_static = brainstate.transform.StatefulFunction(
        ...     g, static_argnums=(1,)
        ... )
        >>> sf_static.make_jaxpr(x, 2)

    Automatic state management:

    .. code-block:: python

        >>> # Execute with automatic state handling
        >>> result = sf.jaxpr_call_auto(x)
        >>> print(state.value)  # State is automatically updated

    See Also
    --------
    make_jaxpr : Function to create jaxpr from a function.
    brainstate.State : The state container class.

    Notes
    -----
    This class maintains internal thread-safe caches for compiled jaxprs, output
    shapes, and state traces. The cache size is bounded at 128 entries per cache
    type. Use ``clear_cache()`` to manually clear the caches if needed.

    State objects should not be passed as direct inputs or outputs to the wrapped
    function. Instead, they should be accessed within the function body, and the
    class will automatically track their usage.
    """
    __module__ = "brainstate.transform"

    def __init__(
        self,
        fun: Callable,
        static_argnums: Union[int, Iterable[int]] = (),
        static_argnames: Union[str, Iterable[str]] = (),
        axis_env: Optional[Sequence[tuple[Hashable, int]]] = None,
        abstracted_axes: Optional[Any] = None,
        name: Optional[str] = None,
        return_only_write: bool = True,
    ):
        # explicit parameters
        self.fun = fun
        self.static_argnums = tuple() if static_argnums is None else _ensure_index_tuple(static_argnums)
        self.static_argnames = tuple() if static_argnames is None else _ensure_str_tuple(static_argnames)
        self.axis_env = axis_env
        self.abstracted_axes = abstracted_axes
        self.name = name
        self.return_only_write = return_only_write

        # implicit parameters - thread-safe bounded caches
        self._cached_jaxpr = _BoundedCache(maxsize=128)
        self._cached_out_shapes = _BoundedCache(maxsize=128)
        self._cached_jaxpr_out_tree = _BoundedCache(maxsize=128)
        self._cached_state_trace = _BoundedCache(maxsize=128)
        self._cache_lock = threading.RLock()

    def __pretty_repr_item__(self, k, v):
        if k.startswith('_'):
            return None
        return k, v

    def get_jaxpr_by_cache(self, cache_key: Hashable) -> ClosedJaxpr:
        """
        Read the JAX Jaxpr representation of the function.

        Parameters
        ----------
        cache_key : Hashable
            The hashable cache key for retrieving the compiled jaxpr.

        Returns
        -------
        ClosedJaxpr
            The JAX Jaxpr representation of the function.

        Raises
        ------
        ValueError
            If the function has not been compiled for the given cache key.
        """
        return self._cached_jaxpr.get(cache_key, raise_on_miss=True, error_context="JAX expression")

    def get_jaxpr(self, *args, compile_if_miss: bool = True, **kwargs) -> ClosedJaxpr:
        """
        Read the JAX Jaxpr representation of the function by calling with args.

        Parameters
        ----------
        *args
            The arguments to the function.
        compile_if_miss : bool, optional
            Whether to compile the function if the cache key is not found. Default is True.
        **kwargs
            The keyword arguments to the function.

        Returns
        -------
        ClosedJaxpr
            The JAX Jaxpr representation of the function.
        """
        cache_key = self.get_arg_cache_key(*args, **kwargs, compile_if_miss=compile_if_miss)
        return self.get_jaxpr_by_cache(cache_key)

    def get_out_shapes_by_cache(self, cache_key: Hashable) -> PyTree:
        """
        Read the output shapes of the function.

        Parameters
        ----------
        cache_key : Hashable
            The hashable cache key.

        Returns
        -------
        PyTree
            The output shapes of the function.

        Raises
        ------
        ValueError
            If the function has not been compiled for the given cache key.
        """
        return self._cached_out_shapes.get(cache_key, raise_on_miss=True, error_context="Output shapes")

    def get_out_shapes(self, *args, compile_if_miss: bool = True, **kwargs) -> PyTree:
        """
        Read the output shapes of the function.

        Parameters
        ----------
        *args
            The arguments to the function.
        compile_if_miss : bool, optional
            Whether to compile the function if the cache key is not found. Default is True.
        **kwargs
            The keyword arguments to the function.

        Returns
        -------
        PyTree
            The output shapes of the function.
        """
        cache_key = self.get_arg_cache_key(*args, **kwargs, compile_if_miss=compile_if_miss)
        return self.get_out_shapes_by_cache(cache_key)

    def get_out_treedef_by_cache(self, cache_key: Hashable) -> PyTree:
        """
        Read the output tree definition of the function.

        Parameters
        ----------
        cache_key : Hashable
            The hashable cache key.

        Returns
        -------
        PyTree
            The output tree definition of the function.

        Raises
        ------
        ValueError
            If the function has not been compiled for the given cache key.
        """
        return self._cached_jaxpr_out_tree.get(cache_key, raise_on_miss=True, error_context="Output tree")

    def get_out_treedef(self, *args, compile_if_miss: bool = True, **kwargs) -> PyTree:
        """
        Read the output tree of the function.

        Parameters
        ----------
        *args
            The arguments to the function.
        compile_if_miss : bool, optional
            Whether to compile the function if the cache key is not found. Default is True.
        **kwargs
            The keyword arguments to the function.

        Returns
        -------
        PyTree
            The output tree of the function.
        """
        cache_key = self.get_arg_cache_key(*args, **kwargs, compile_if_miss=compile_if_miss)
        return self.get_out_treedef_by_cache(cache_key)

    def get_state_trace_by_cache(self, cache_key: Hashable) -> StateTraceStack:
        """
        Read the state trace of the function.

        Parameters
        ----------
        cache_key : Hashable
            The hashable cache key.

        Returns
        -------
        StateTraceStack
            The state trace stack containing all tracked states.

        Raises
        ------
        ValueError
            If the function has not been compiled for the given cache key.
        """
        return self._cached_state_trace.get(cache_key, raise_on_miss=True, error_context="State trace")

    def get_state_trace(self, *args, compile_if_miss: bool = True, **kwargs) -> StateTraceStack:
        """
        Read the state trace of the function.

        Parameters
        ----------
        *args
            The arguments to the function.
        compile_if_miss : bool, optional
            Whether to compile the function if the cache key is not found. Default is True.
        **kwargs
            The keyword arguments to the function.

        Returns
        -------
        StateTraceStack
            The state trace of the function.
        """
        cache_key = self.get_arg_cache_key(*args, **kwargs, compile_if_miss=compile_if_miss)
        return self.get_state_trace_by_cache(cache_key)

    def get_states_by_cache(self, cache_key: Hashable) -> Tuple[State, ...]:
        """
        Read the states that are accessed by the function.

        Parameters
        ----------
        cache_key : Hashable
            The hashable cache key.

        Returns
        -------
        Tuple[State, ...]
            The states that are read from or written to by the function.

        Raises
        ------
        ValueError
            If the function has not been compiled for the given cache key.
        """
        return tuple(self.get_state_trace_by_cache(cache_key).states)

    def get_states(self, *args, compile_if_miss: bool = True, **kwargs) -> Tuple[State, ...]:
        """
        Compile the function, and get the states that are read and written by this function.

        Parameters
        ----------
        *args
            The arguments to the function.
        compile_if_miss : bool, optional
            Whether to compile the function if the cache key is not found. Default is True.
        **kwargs
            The keyword arguments to the function.

        Returns
        -------
        Tuple[State, ...]
            The states that are read and written by the function.
        """
        cache_key = self.get_arg_cache_key(*args, **kwargs, compile_if_miss=compile_if_miss)
        return self.get_states_by_cache(cache_key)

    def get_read_states_by_cache(self, cache_key: Hashable) -> Tuple[State, ...]:
        """
        Read the states that are read by the function.

        Parameters
        ----------
        cache_key : Hashable
            The hashable key.

        Returns
        -------
        Tuple[State, ...]
            The states that are read by the function.
        """
        return self.get_state_trace_by_cache(cache_key).get_read_states()

    def get_read_states(self, *args, compile_if_miss: bool = True, **kwargs) -> Tuple[State, ...]:
        """
        Compile the function, and get the states that are read by this function.

        Parameters
        ----------
        *args
            The arguments to the function.
        compile_if_miss : bool, optional
            Whether to compile the function if the cache key is not found. Default is True.
        **kwargs
            The keyword arguments to the function.

        Returns
        -------
        Tuple[State, ...]
            The states that are read by the function.
        """
        cache_key = self.get_arg_cache_key(*args, **kwargs, compile_if_miss=compile_if_miss)
        return self.get_read_states_by_cache(cache_key)

    def get_write_states_by_cache(self, cache_key: Hashable) -> Tuple[State, ...]:
        """
        Read the states that are written by the function.

        Parameters
        ----------
        cache_key : Hashable
            The hashable cache key.

        Returns
        -------
        Tuple[State, ...]
            The states that are written by the function.
        """
        return self.get_state_trace_by_cache(cache_key).get_write_states()

    def get_write_states(self, *args, compile_if_miss: bool = True, **kwargs) -> Tuple[State, ...]:
        """
        Compile the function, and get the states that are written by this function.

        Parameters
        ----------
        *args
            The arguments to the function.
        compile_if_miss : bool, optional
            Whether to compile the function if the cache key is not found. Default is True.
        **kwargs
            The keyword arguments to the function.

        Returns
        -------
        Tuple[State, ...]
            The states that are written by the function.
        """
        cache_key = self.get_arg_cache_key(*args, **kwargs, compile_if_miss=compile_if_miss)
        return self.get_write_states_by_cache(cache_key)

    def _check_input_ouput(self, x):
        if isinstance(x, State):
            x.raise_error_with_source_info(
                ValueError(
                    'Inputs/outputs for brainstate transformations cannot be an instance of State. '
                    f'But we got {x}'
                )
            )

    def get_arg_cache_key(self, *args, compile_if_miss: bool = False, **kwargs) -> hashabledict:
        """
        Compute the cache key for the given arguments.

        This method separates static and dynamic arguments and creates a hashable
        key that can be used to cache compiled jaxpr representations.

        Parameters
        ----------
        *args
            The positional arguments to the function.
        compile_if_miss : bool, optional
            Whether to compile the function if the cache key does not exist.
            Default is False.
        **kwargs
            The keyword arguments to the function.

        Returns
        -------
        hashabledict
            A hashable dictionary containing the cache key with fields:
            'static_args', 'dyn_args', 'static_kwargs', 'dyn_kwargs'.

        Examples
        --------
        .. code-block:: python

            >>> import brainstate
            >>> import jax.numpy as jnp
            >>>
            >>> def f(x, n):
            ...     return x ** n
            >>>
            >>> sf = brainstate.transform.StatefulFunction(
            ...     f, static_argnums=(1,)
            ... )
            >>> cache_key = sf.get_arg_cache_key(jnp.array([1.0, 2.0]), 2)
        """
        static_args, dyn_args = [], []
        for i, arg in enumerate(args):
            if i in self.static_argnums:
                static_args.append(arg)
            else:
                dyn_args.append(arg)
        dyn_args = jax.tree.map(shaped_abstractify, dyn_args)
        static_kwargs, dyn_kwargs = [], []
        for k, v in sorted(kwargs.items()):
            if k in self.static_argnames:
                static_kwargs.append((k, v))
            else:
                dyn_kwargs.append((k, jax.tree.map(shaped_abstractify, v)))

        static_args = make_hashable(tuple(static_args))
        dyn_args = make_hashable(tuple(dyn_args))
        static_kwargs = make_hashable(static_kwargs)
        dyn_kwargs = make_hashable(dyn_kwargs)

        cache_key = hashabledict(
            static_args=static_args,
            dyn_args=dyn_args,
            static_kwargs=static_kwargs,
            dyn_kwargs=dyn_kwargs,
        )

        if cache_key not in self._cached_state_trace and compile_if_miss:
            self.make_jaxpr(*args, **kwargs)

        return cache_key

    def clear_cache(self) -> None:
        """
        Clear all compilation caches.

        This method removes all cached jaxprs, output shapes, output trees,
        and state traces. Use this when you need to recompile the function
        or free memory.

        Examples
        --------
        .. code-block:: python

            >>> import brainstate
            >>> import jax.numpy as jnp
            >>>
            >>> def f(x):
            ...     return x * 2
            >>>
            >>> sf = brainstate.transform.StatefulFunction(f)
            >>> sf.make_jaxpr(jnp.array([1.0, 2.0]))
            >>> sf.clear_cache()  # Clear all cached compilations
        """
        self._cached_jaxpr.clear()
        self._cached_out_shapes.clear()
        self._cached_jaxpr_out_tree.clear()
        self._cached_state_trace.clear()

    def __jax_v04_new_arg(self):
        # Should be within the calling of ``jax.make_jaxpr()``
        frame, trace = _jax_v04_new_jax_trace()
        # Set the function to transform the new argument to a tracer
        fn = functools.partial(_jax_v04_new_arg_fn, frame, trace)
        return fn

    def __jax_new_version_new_arg(self):
        trace = jax.core.trace_ctx.trace

        def wrapper(x):
            if jax.__version_info__ < (0, 6, 1):
                fn = lambda xx: trace.new_arg(shaped_abstractify(xx))
            else:
                fn = lambda xx: trace.new_arg(shaped_abstractify(xx), source_info=source_info_util.current())
            return jax.tree.map(fn, x._value)

        return wrapper

    def _wrapped_fun_to_eval(
        self,
        cache_key,
        static_kwargs: dict,
        *args,
        **dyn_kwargs,
    ) -> Tuple[Any, Tuple[State, ...]]:
        """
        Internal wrapper that executes the function and tracks state operations.

        This method wraps the original function to track which states are read
        and written during execution. It is used internally during jaxpr compilation.

        Parameters
        ----------
        cache_key
            The cache key for storing the state trace.
        static_kwargs : dict
            Static keyword arguments that were separated out.
        *args
            The positional arguments to the function.
        **dyn_kwargs
            Dynamic keyword arguments to the function.

        Returns
        -------
        tuple
            A tuple of (output, state_values) where output is the function result
            and state_values are the tracked state values (either all or write-only
            depending on return_only_write setting).
        """
        # state trace
        state_trace: StateTraceStack = StateTraceStack(self.name)
        if jax.__version_info__ < (0, 4, 36):
            state_trace.set_new_arg(self.__jax_v04_new_arg())
        else:
            state_trace.set_new_arg(self.__jax_new_version_new_arg())
        self._cached_state_trace.set(cache_key, state_trace)
        with state_trace:
            out = self.fun(*args, **dyn_kwargs, **static_kwargs)
            state_values = (
                state_trace.get_write_state_values(True)
                if self.return_only_write else
                state_trace.get_state_values()
            )
        state_trace.recovery_original_values()

        # State instance as functional returns is not allowed.
        # Checking whether the states are returned.
        jax.tree.map(self._check_input_ouput, out, is_leaf=lambda x: isinstance(x, State))
        return out, state_values

    def make_jaxpr(self, *args, **kwargs):
        """
        Create the JAX Jaxpr representation given example arguments.

        This method compiles the function with the given arguments and caches
        the resulting Jaxpr, output shapes, and state trace for later use.

        Parameters
        ----------
        *args
            The arguments to the function.
        **kwargs
            The keyword arguments to the function.

        Returns
        -------
        StatefulFunction
            Returns self for method chaining.

        Raises
        ------
        TypeError
            If State objects are passed as arguments or returned from the function.
        """

        # check input types
        jax.tree.map(self._check_input_ouput, (args, kwargs), is_leaf=lambda x: isinstance(x, State))

        # static args
        cache_key = self.get_arg_cache_key(*args, **kwargs)

        if cache_key not in self._cached_state_trace:
            try:

                # jaxpr
                static_kwargs, dyn_kwargs = {}, {}
                for k, v in kwargs.items():
                    if k in self.static_argnames:
                        static_kwargs[k] = v
                    else:
                        dyn_kwargs[k] = v
                jaxpr, (out_shapes, state_shapes) = _make_jaxpr(
                    functools.partial(
                        self._wrapped_fun_to_eval,
                        cache_key,
                        static_kwargs,
                    ),
                    static_argnums=self.static_argnums,
                    axis_env=self.axis_env,
                    return_shape=True,
                    abstracted_axes=self.abstracted_axes,
                )(*args, **dyn_kwargs)

                # returns
                self._cached_jaxpr_out_tree.set(cache_key, jax.tree.structure((out_shapes, state_shapes)))
                self._cached_out_shapes.set(cache_key, (out_shapes, state_shapes))
                self._cached_jaxpr.set(cache_key, jaxpr)

            except Exception as e:
                # Clean up partial cache entries on error
                self._cached_state_trace.pop(cache_key, None)
                self._cached_out_shapes.pop(cache_key, None)
                self._cached_jaxpr.pop(cache_key, None)
                raise e

        return self

    def jaxpr_call(self, state_vals, *args, **kwargs) -> Any:
        """
        Call the function at the JAX Jaxpr level.

        This method evaluates the compiled Jaxpr with the provided state values
        and arguments, returning updated state values and function outputs.

        Parameters
        ----------
        state_vals : Sequence
            The current state values.
        *args
            The arguments to the function.
        **kwargs
            The keyword arguments to the function.

        Returns
        -------
        tuple
            A tuple of (new_state_vals, out) where new_state_vals are the
            updated state values and out is the function output.

        Raises
        ------
        ValueError
            If the number of state values doesn't match the expected number.
        """
        # state checking
        cache_key = self.get_arg_cache_key(*args, **kwargs)
        states: Sequence[State] = self.get_states_by_cache(cache_key)
        if len(state_vals) != len(states):
            raise ValueError(f'State length mismatch: expected {len(states)} states, got {len(state_vals)}')

        # parameters
        kwargs = {k: v for k, v in kwargs.items() if k not in self.static_argnames}  # remove static kwargs
        args = tuple(args[i] for i in range(len(args)) if i not in self.static_argnums)
        args = jax.tree.flatten((args, kwargs, state_vals))[0]

        # calling the function,
        # note that this function always returns state values
        # that both write and read by the function
        closed_jaxpr = self.get_jaxpr_by_cache(cache_key)
        out_treedef = self.get_out_treedef_by_cache(cache_key)
        jaxpr_outs = jax.core.eval_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.consts, *args)

        # output processing
        out, new_state_vals = out_treedef.unflatten(jaxpr_outs)
        if len(new_state_vals) != len(state_vals):
            raise ValueError(f'State length mismatch in output: expected '
                             f'{len(state_vals)} states, got {len(new_state_vals)}')
        return new_state_vals, out

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive cache statistics for all internal caches.

        Returns
        -------
        dict
            A dictionary with statistics for each cache including size, hits,
            misses, and hit rates. Keys are 'jaxpr_cache', 'out_shapes_cache',
            'jaxpr_out_tree_cache', and 'state_trace_cache'.
        """
        return {
            'jaxpr_cache': self._cached_jaxpr.get_stats(),
            'out_shapes_cache': self._cached_out_shapes.get_stats(),
            'jaxpr_out_tree_cache': self._cached_jaxpr_out_tree.get_stats(),
            'state_trace_cache': self._cached_state_trace.get_stats(),
        }

    def validate_states(self, cache_key: Hashable) -> bool:
        """
        Validate that all tracked states for a given cache key are still valid.

        Parameters
        ----------
        cache_key : Hashable
            The cache key to validate states for.

        Returns
        -------
        bool
            True if all states are valid.

        Raises
        ------
        ValueError
            If any states are invalid or missing required attributes.
        """
        state_trace = self.get_state_trace_by_cache(cache_key)
        invalid_states = []
        for i, state in enumerate(state_trace.states):
            if not hasattr(state, 'value'):
                invalid_states.append((i, state))

        if invalid_states:
            raise ValueError(
                f"Found {len(invalid_states)} invalid states at indices: "
                f"{[idx for idx, _ in invalid_states]}. "
                f"States must have a 'value' attribute."
            )
        return True

    def validate_all_states(self) -> Dict[Any, bool]:
        """
        Validate states for all cached compilations.

        Returns
        -------
        dict
            A dictionary mapping cache keys to validation results. Each value
            is either True (valid) or an error message string (invalid).
        """
        results = {}
        for cache_key in self._cached_state_trace.keys():
            try:
                results[cache_key] = self.validate_states(cache_key)
            except ValueError as e:
                results[cache_key] = str(e)
        return results

    def jaxpr_call_auto(self, *args, **kwargs) -> Any:
        """
        Execute the function at the jaxpr level with automatic state management.

        This method automatically retrieves current state values, executes the
        jaxpr-compiled function, and updates the states with the new values.
        It provides a convenient interface that handles all state management
        automatically.

        Parameters
        ----------
        *args
            The positional arguments to the function.
        **kwargs
            The keyword arguments to the function.

        Returns
        -------
        Any
            The output of the function.

        Examples
        --------
        .. code-block:: python

            >>> import brainstate
            >>> import jax.numpy as jnp
            >>>
            >>> state = brainstate.State(jnp.array([1.0, 2.0]))
            >>>
            >>> def f(x):
            ...     state.value += x
            ...     return state.value * 2
            >>>
            >>> sf = brainstate.transform.StatefulFunction(f)
            >>> x = jnp.array([0.5, 0.5])
            >>> sf.make_jaxpr(x)
            >>>
            >>> # Automatic state management
            >>> result = sf.jaxpr_call_auto(x)
            # # or
            >>> result = sf(x)
            >>> print(state.value)  # State is automatically updated
        """
        state_trace = self.get_state_trace_by_cache(self.get_arg_cache_key(*args, **kwargs, compile_if_miss=True))
        all_read_state_vals = state_trace.get_read_state_values(True)
        state_vals, out = self.jaxpr_call(state_trace.get_state_values(), *args, **kwargs)
        state_trace.assign_state_vals_v2(all_read_state_vals, state_vals)
        return out

    def __call__(self, *args, **kwargs):
        return self.jaxpr_call_auto(*args, **kwargs)


@set_module_as("brainstate.transform")
def make_jaxpr(
    fun: Callable,
    static_argnums: Union[int, Iterable[int]] = (),
    static_argnames: Union[str, Iterable[str]] = (),
    axis_env: Optional[Sequence[tuple[Hashable, int]]] = None,
    return_shape: bool = False,
    abstracted_axes: Optional[Any] = None,
    return_only_write: bool = False,
) -> Callable[
    ...,
    (Tuple[ClosedJaxpr, Tuple[State, ...]] |
     Tuple[ClosedJaxpr, Tuple[State, ...], PyTree])
]:
    """
    Creates a function that produces its jaxpr given example args.

    A ``jaxpr`` is JAX's intermediate representation for program traces. The
    ``jaxpr`` language is based on the simply-typed first-order lambda calculus
    with let-bindings. :py:func:`make_jaxpr` adapts a function to return its
    ``jaxpr``, which we can inspect to understand what JAX is doing internally.
    The ``jaxpr`` returned is a trace of ``fun`` abstracted to
    :py:class:`ShapedArray` level. Other levels of abstraction exist internally.

    Parameters
    ----------
    fun : callable
        The function whose ``jaxpr`` is to be computed. Its positional
        arguments and return value should be arrays, scalars, or standard Python
        containers (tuple/list/dict) thereof.
    static_argnums : int or iterable of int, optional
        See the :py:func:`jax.jit` docstring.
    static_argnames : str or iterable of str, optional
        See the :py:func:`jax.jit` docstring.
    axis_env : sequence of tuple, optional
        A sequence of pairs where the first element is an axis
        name and the second element is a positive integer representing the size of
        the mapped axis with that name. This parameter is useful when lowering
        functions that involve parallel communication collectives, and it
        specifies the axis name/size environment that would be set up by
        applications of :py:func:`jax.pmap`.
    return_shape : bool, default False
        If ``True``, the
        wrapped function returns a pair where the first element is the XLA
        computation and the second element is a pytree with the same structure as
        the output of ``fun`` and where the leaves are objects with ``shape``,
        ``dtype``, and ``named_shape`` attributes representing the corresponding
        types of the output leaves.
    abstracted_axes : pytree, optional
        A pytree with the same structure as the input
        arguments to ``fun``. The leaves of the pytree can be either None or a
        dict with axis names as keys and integers as values. If the leaf is None,
        then the corresponding axis is not abstracted. If the leaf is a dict, then
        the corresponding axis is abstracted, and the dict specifies the axis name
        and size. The abstracted axes are used to infer the input type of the
        function. If None, then all axes are abstracted.
    return_only_write : bool, default False
        If True, only return states that were written to during execution
        (not just read). This can reduce memory usage when you only care
        about modified states.

    Returns
    -------
    callable
        A wrapped version of ``fun`` that when applied to example arguments returns
        a ``ClosedJaxpr`` representation of ``fun`` on those arguments. If the
        argument ``return_shape`` is ``True``, then the returned function instead
        returns a pair where the first element is the ``ClosedJaxpr``
        representation of ``fun`` and the second element is a pytree representing
        the structure, shape, dtypes, and named shapes of the output of ``fun``.

    Examples
    --------
    Basic usage:

    .. code-block:: python

        >>> import jax
        >>> import brainstate
        >>> import jax.numpy as jnp
        >>>
        >>> def f(x):
        ...     return jnp.sin(jnp.cos(x))
        >>>
        >>> # Create jaxpr maker
        >>> jaxpr_maker = brainstate.transform.make_jaxpr(f)
        >>> jaxpr, states = jaxpr_maker(3.0)

    With gradient:

    .. code-block:: python

        >>> jaxpr_grad_maker = brainstate.transform.make_jaxpr(jax.grad(f))
        >>> jaxpr, states = jaxpr_grad_maker(3.0)

    With shape information:

    .. code-block:: python

        >>> jaxpr_maker_with_shape = brainstate.transform.make_jaxpr(f, return_shape=True)
        >>> jaxpr, states, shapes = jaxpr_maker_with_shape(3.0)

    With stateful function:

    .. code-block:: python

        >>> state = brainstate.State(jnp.array([1.0, 2.0]))
        >>>
        >>> def stateful_f(x):
        ...     state.value += x
        ...     return state.value
        >>>
        >>> jaxpr_maker = brainstate.transform.make_jaxpr(stateful_f)
        >>> jaxpr, states = jaxpr_maker(jnp.array([0.5, 0.5]))
    """

    stateful_fun = StatefulFunction(
        fun,
        static_argnums=static_argnums,
        static_argnames=static_argnames,
        axis_env=axis_env,
        abstracted_axes=abstracted_axes,
        return_only_write=return_only_write,
        name='make_jaxpr'
    )

    @wraps(fun)
    def make_jaxpr_f(*args, **kwargs):
        stateful_fun.make_jaxpr(*args, **kwargs)
        cache_key = stateful_fun.get_arg_cache_key(*args, **kwargs)
        if return_shape:
            return (
                stateful_fun.get_jaxpr_by_cache(cache_key),
                stateful_fun.get_states_by_cache(cache_key),
                stateful_fun.get_out_shapes_by_cache(cache_key)[0]
            )
        else:
            return (
                stateful_fun.get_jaxpr_by_cache(cache_key),
                stateful_fun.get_states_by_cache(cache_key)
            )

    # wrapped jaxpr builder function
    make_jaxpr_f.__module__ = "brainstate.transform"
    if hasattr(fun, "__qualname__"):
        make_jaxpr_f.__qualname__ = f"make_jaxpr({fun.__qualname__})"
    if hasattr(fun, "__name__"):
        make_jaxpr_f.__name__ = f"make_jaxpr({fun.__name__})"
    return make_jaxpr_f


class StatefulMapping(StatefulFunction):
    __module__ = "brainstate.transform"

    def __init__(
        self,
        fun: Callable,
        in_axes: Union[int, Tuple[int, ...], None] = 0,
        out_axes: Union[int, Tuple[int, ...], None] = 0,
        state_in_axes: Optional[Union[Dict[AxisName, Filter], Filter]] = None,
        state_out_axes: Optional[Union[Dict[AxisName, Filter], Filter]] = None,
        # jit specific parameters
        static_argnums: Union[int, Iterable[int]] = (),
        static_argnames: Union[str, Iterable[str]] = (),
        axis_env: Optional[Sequence[tuple[Hashable, int]]] = None,
        abstracted_axes: Optional[Any] = None,
        # mapping specific parameters
        axis_size: Optional[int] = None,
        axis_name: AxisName | None = None,
        name: Optional[str] = None,
        # mapping function
        mapping_fn: Callable = jax.vmap,
    ):
        self.origin_fun = fun
        super().__init__(
            fun=self._wrapped_fun,
            static_argnums=static_argnums,
            static_argnames=static_argnames,
            axis_env=axis_env,
            abstracted_axes=abstracted_axes,
            name=name,
            return_only_write=False,
        )
        self.in_axes = in_axes
        self.out_axes = out_axes
        if state_in_axes is None:
            state_in_axes = dict()
        elif not isinstance(state_in_axes, dict):
            state_in_axes = {0: to_predicate(state_in_axes)}
        state_in_axes = {k: to_predicate(v) for k, v in state_in_axes.items()}  # type: ignore
        self.state_in_axes = state_in_axes

        if state_out_axes is None:
            state_out_axes = dict()
        elif not isinstance(state_out_axes, dict):
            state_out_axes = {0: to_predicate(state_out_axes)}
        state_out_axes = {k: to_predicate(v) for k, v in state_out_axes.items()}  # type: ignore
        self.state_out_axes = state_out_axes

        self.axis_size = axis_size
        self.axis_name = axis_name
        self.mapping_fn = mapping_fn

        # Cache for discovered state-to-axis mappings
        self._cached_map_dim_to_in_states = _BoundedCache(maxsize=128)
        self._cached_map_dim_to_out_states = _BoundedCache(maxsize=128)
        self._cached_map_state_trace = _BoundedCache(maxsize=128)
        self._cached_map_batch_size = _BoundedCache(maxsize=128)

    def _infer_batch_size(self, args, in_axes):
        if in_axes is None:
            raise ValueError("Cannot infer batch size when in_axes is None")

        batch_sizes = []

        def get_batch_size_from_arg(arg_, axis_):
            if axis_ is None:
                return None

            def _get_size(arr):
                if not hasattr(arr, 'shape'):
                    return None
                if arr.ndim == 0:
                    return None
                ax = axis_ if axis_ >= 0 else arr.ndim + axis_
                if ax < 0 or ax >= arr.ndim:
                    raise IndexError(f"Axis {ax} is out of bounds for array of shape {arr.shape}")
                return arr.shape[ax]

            # Get all sizes from the pytree
            sizes = [s for s in jax.tree.leaves(jax.tree.map(_get_size, arg_)) if s is not None]
            return sizes[0] if sizes else None

        if isinstance(in_axes, int):
            # All args batched along the same axis
            for arg in args:
                size = get_batch_size_from_arg(arg, in_axes)
                if size is not None:
                    batch_sizes.append(size)
        elif isinstance(in_axes, (tuple, list)):
            # Different axes for different args
            if len(in_axes) != len(args):
                raise ValueError(
                    f"Length of in_axes ({len(in_axes)}) must match number of arguments ({len(args)})"
                )
            for arg, axis in zip(args, in_axes):
                size = get_batch_size_from_arg(arg, axis)
                if size is not None:
                    batch_sizes.append(size)
        else:
            raise TypeError(f"Unsupported in_axes type: {type(in_axes)}")

        if not batch_sizes:
            if self.axis_size is None:
                raise ValueError("Cannot infer batch size when axis_size is None")
            batch_sizes.append(self.axis_size)

        # Check all batch sizes are consistent
        if not all(s == batch_sizes[0] for s in batch_sizes):
            raise ValueError(
                f"Inconsistent batch sizes found: {batch_sizes}. "
                f"All batched arguments must have the same size along their batch axes."
            )

        return batch_sizes[0]

    def __new_batch_arg(self, batch_size: int, dim_to_states: dict):
        trace = jax.core.trace_ctx.trace
        assert isinstance(trace, BatchTrace), f"Expected to be called within a BatchTrace context, but got {trace}"

        def wrapper(x):
            if isinstance(x, RandomState):
                idx = memoize(lambda: BatchTracer(trace, make_iota(batch_size), 0, source_info_util.current()))
                dim_to_states['random'].append(x)
                return to_elt(trace, idx, jnp.ones((batch_size,) + x._value.shape, x._value.dtype), 0)
            for dim, filter_ in self.state_in_axes.items():
                idx = memoize(lambda: BatchTracer(trace, make_iota(batch_size), dim, source_info_util.current()))
                if filter_(tuple(), x):
                    dim_to_states[dim].append(x)
                    return jax.tree.map(lambda xx: to_elt(trace, idx, xx, dim), x._value)
            return x._value

        return wrapper

    def __eval(self, cache_key, *args, **kwargs):
        def fn_to_eval(*new_args, **new_kwargs):
            dim_to_in_states = defaultdict(list)
            state_trace = StateTraceStack(name=self.name)
            state_trace.set_new_arg(
                self.__new_batch_arg(self._cached_map_batch_size.get(cache_key), dim_to_in_states)
            )
            self._cached_map_state_trace.set(cache_key, state_trace)

            # call functions
            with state_trace:
                out_ = self.origin_fun(*new_args, **new_kwargs)

            # cache
            self._cached_map_dim_to_in_states.set(cache_key, dim_to_in_states)

            # vmapped state values
            out_states = defaultdict(list)
            out_states['random'] = [st for st in state_trace.states if isinstance(st, RandomState)]
            for st in state_trace.states:
                if not isinstance(st, RandomState):
                    leaves = jax.tree.leaves(st._value)
                    batch_dims = set([leaf.batch_dim if isinstance(leaf, BatchTracer) else None for leaf in leaves])
                    if len(batch_dims) != 1:
                        raise ValueError(
                            f"State {st} has inconsistent batch dimensions in its leaves: {batch_dims}. "
                            "All leaves must have the same batch dimension."
                        )
                    batch_dim = batch_dims.pop()
                    out_states[batch_dim].append(st)
            self._cached_map_dim_to_out_states.set(cache_key, out_states)

        try:
            jax.vmap(
                fn_to_eval,
                in_axes=self.in_axes,
                out_axes=self.out_axes,
                axis_name=self.axis_name,
                axis_size=self.axis_size
            )(*args, **kwargs)
            self._cached_map_state_trace.get(cache_key).recovery_original_values()
        except Exception as e:
            if cache_key in self._cached_map_state_trace:
                self._cached_map_state_trace.get(cache_key).recovery_original_values()
            self._cached_map_state_trace.pop(cache_key, None)
            self._cached_map_dim_to_in_states.pop(cache_key, None)
            self._cached_map_dim_to_out_states.pop(cache_key, None)
            self._cached_map_batch_size.pop(cache_key, None)
            raise RuntimeError(f"Failed to evaluate {self}") from e

    def __assign_vals_from_in_states(self, cache_key, rand_st, *other_st):
        in_states = self._cached_map_dim_to_in_states.get(cache_key)
        for st, val in zip(in_states['random'], rand_st):
            assert isinstance(st, RandomState)
            st.restore_value(val)
        for group, group_vals in zip([in_states[dim] for dim in in_states.keys() if dim != 'random'], other_st):
            for st, val in zip(group, group_vals):
                st.restore_value(val)

    def __assign_vals_from_out_states(self, cache_key, rand_st, *other_st):
        out_states = self._cached_map_dim_to_out_states.get(cache_key)
        for st, val in zip(out_states['random'], rand_st):
            assert isinstance(st, RandomState)
            st.restore_value(val)
        for group, group_vals in zip([out_states[dim] for dim in out_states.keys() if dim != 'random'], other_st):
            for st, val in zip(group, group_vals):
                st.restore_value(val)

    def __get_in_state_vals(self, cache_key: Hashable):
        in_states = self._cached_map_dim_to_in_states.get(cache_key)
        in_axes = []
        in_values = []
        for dim, states in in_states.items():
            if dim == 'random':
                continue
            in_axes.append(dim)
            in_values.append([st.value for st in states])
        return tuple(in_axes), in_values

    def __get_out_state_vals(self, cache_key: Hashable):
        out_states = self._cached_map_dim_to_out_states.get(cache_key)
        out_axes = []
        out_values = []
        for dim, state in out_states.items():
            if dim == 'random':
                continue
            out_axes.append(dim)
            out_values.append([st.value for st in state])
        return tuple(out_axes), out_values

    def __get_rand_state_vals(self, cache_key: Hashable):
        in_states = self._cached_map_dim_to_in_states.get(cache_key)
        batch_size = self._cached_map_batch_size.get(cache_key)
        rand_vals, rand_recover_vals = [], []
        for st in in_states['random']:
            assert isinstance(st, RandomState)
            rand_vals.append(st.split_key(batch_size))
            rand_recover_vals.append(st.value)
        return tuple(rand_vals), tuple(rand_recover_vals)

    def __recover_rand_state_vals(self, cache_key: Hashable, rand_recover_vals):
        state_trace = self._cached_map_state_trace.get(cache_key)
        rand_states = [st for st in state_trace.states if isinstance(st, RandomState)]
        for st, val in zip(rand_states, rand_recover_vals):
            st.restore_value(val)

    def _wrapped_fun(self, *args, **kwargs) -> Tuple[Any, Tuple[State, ...]]:
        batch_size = self._infer_batch_size(args, self.in_axes)
        cache_key = self.get_arg_cache_key(*args, **kwargs)
        self._cached_map_batch_size.set(cache_key, batch_size)
        if cache_key not in self._cached_map_state_trace:
            self.__eval(cache_key, *args, **kwargs)

        def fn_to_map(origin_args, rand_st, *non_rand_st):
            self.__assign_vals_from_in_states(cache_key, rand_st, *non_rand_st)
            out = self.origin_fun(*origin_args[0], **origin_args[1])
            return out, *self.__get_out_state_vals(cache_key)[1]

        in_axes, in_state_vals = self.__get_in_state_vals(cache_key)
        out_axes, out_state_vals = self.__get_out_state_vals(cache_key)
        rand_vals, rand_recover_vals = self.__get_rand_state_vals(cache_key)
        mapped_fn = self.mapping_fn(
            fn_to_map,
            in_axes=(self.in_axes, 0) + in_axes,
            out_axes=(self.out_axes,) + out_axes,
            axis_size=self.axis_size,
            axis_name=self.axis_name,
        )
        out_, *out_state_vals = mapped_fn((args, kwargs), rand_vals, *in_state_vals)
        self.__assign_vals_from_out_states(cache_key, rand_recover_vals, *out_state_vals)
        return out_


def _check_callable(fun):
    # In Python 3.10+, the only thing stopping us from supporting static methods
    # is that we can't take weak references to them, which the C++ JIT requires.
    if isinstance(fun, staticmethod):
        raise TypeError(f"staticmethod arguments are not supported, got {fun}")
    if not callable(fun):
        raise TypeError(f"Expected a callable value, got {fun}")
    if inspect.isgeneratorfunction(fun):
        raise TypeError(f"Expected a function, got a generator function: {fun}")


def _broadcast_prefix(
    prefix_tree: Any,
    full_tree: Any,
    is_leaf: Callable[[Any], bool] | None = None
) -> list[Any]:
    # If prefix_tree is not a tree prefix of full_tree, this code can raise a
    # ValueError; use prefix_errors to find disagreements and raise more precise
    # error messages.
    result = []
    num_leaves = lambda t: jax.tree.structure(t).num_leaves
    add_leaves = lambda x, subtree: result.extend([x] * num_leaves(subtree))
    jax.tree.map(add_leaves, prefix_tree, full_tree, is_leaf=is_leaf)
    return result


def _flat_axes_specs(
    abstracted_axes, *args, **kwargs
) -> list[pe.AbstractedAxesSpec]:
    if kwargs:
        raise NotImplementedError

    def ax_leaf(l):
        return (isinstance(l, dict) and jax.tree_util.all_leaves(l.values()) or
                isinstance(l, tuple) and jax.tree_util.all_leaves(l, lambda x: x is None))

    return _broadcast_prefix(abstracted_axes, args, ax_leaf)


@transformation_with_aux
def _flatten_fun(in_tree, *args_flat):
    py_args, py_kwargs = jax.tree.unflatten(in_tree, args_flat)
    ans = yield py_args, py_kwargs
    yield jax.tree.flatten(ans)


def _make_jaxpr(
    fun: Callable,
    static_argnums: int | Iterable[int] = (),
    axis_env: Sequence[tuple[AxisName, int]] | None = None,
    return_shape: bool = False,
    abstracted_axes: Any | None = None,
) -> Callable[..., (ClosedJaxpr | tuple[ClosedJaxpr, Any])]:
    """
    Create a function that produces its jaxpr given example args (internal implementation).

    This is an internal implementation function. Users should use the public
    ``make_jaxpr`` function instead.

    Parameters
    ----------
    fun : Callable
        The function whose ``jaxpr`` is to be computed. Its positional
        arguments and return value should be arrays, scalars, or standard Python
        containers (tuple/list/dict) thereof.
    static_argnums : int or iterable of int, optional
        See the :py:func:`jax.jit` docstring.
    axis_env : sequence of tuple, optional
        A sequence of pairs where the first element is an axis
        name and the second element is a positive integer representing the size of
        the mapped axis with that name. This parameter is useful when lowering
        functions that involve parallel communication collectives, and it
        specifies the axis name/size environment that would be set up by
        applications of :py:func:`jax.pmap`.
    return_shape : bool, default False
        If ``True``, the wrapped function returns a pair where the first element
        is the ``ClosedJaxpr`` representation of ``fun`` and the second element
        is a pytree with the same structure as the output of ``fun`` and where
        the leaves are objects with ``shape``, ``dtype``, and ``named_shape``
        attributes representing the corresponding types of the output leaves.
    abstracted_axes : Any, optional
        Axes specifications for abstract interpretation.

    Returns
    -------
    Callable
        A wrapped version of ``fun`` that when applied to example arguments returns
        a ``ClosedJaxpr`` representation of ``fun`` on those arguments. If the
        argument ``return_shape`` is ``True``, then the returned function instead
        returns a pair where the first element is the ``ClosedJaxpr``
        representation of ``fun`` and the second element is a pytree representing
        the structure, shape, dtypes, and named shapes of the output of ``fun``.

    Notes
    -----
    A ``jaxpr`` is JAX's intermediate representation for program traces. The
    ``jaxpr`` language is based on the simply-typed first-order lambda calculus
    with let-bindings. This function adapts a function to return its
    ``jaxpr``, which we can inspect to understand what JAX is doing internally.
    The ``jaxpr`` returned is a trace of ``fun`` abstracted to
    :py:class:`ShapedArray` level. Other levels of abstraction exist internally.

    Examples
    --------
    .. code-block:: python

        >>> import jax
        >>>
        >>> def f(x): return jax.numpy.sin(jax.numpy.cos(x))
        >>> print(f(3.0))
        -0.83602
        >>> _make_jaxpr(f)(3.0)
        { lambda ; a:f32[]. let b:f32[] = cos a; c:f32[] = sin b in (c,) }
        >>> _make_jaxpr(jax.grad(f))(3.0)
        { lambda ; a:f32[]. let
            b:f32[] = cos a
            c:f32[] = sin a
            _:f32[] = sin b
            d:f32[] = cos b
            e:f32[] = mul 1.0 d
            f:f32[] = neg e
            g:f32[] = mul f c
          in (g,) }
    """
    _check_callable(fun)
    static_argnums = _ensure_index_tuple(static_argnums)

    def _abstractify(args, kwargs):
        flat_args, in_tree = jax.tree.flatten((args, kwargs))
        if abstracted_axes is None:
            return map(shaped_abstractify, flat_args), in_tree, [True] * len(flat_args)
        else:
            axes_specs = _flat_axes_specs(abstracted_axes, *args, **kwargs)
            in_type = pe.infer_lambda_input_type(axes_specs, flat_args)
            in_avals, keep_inputs = unzip2(in_type)
            return in_avals, in_tree, keep_inputs

    @wraps(fun)
    @api_boundary
    def make_jaxpr_f(*args, **kwargs):
        f = wrap_init(fun, (), {}, 'brainstate.transform.make_jaxpr')
        if static_argnums:
            dyn_argnums = [i for i in range(len(args)) if i not in static_argnums]
            f, args = jax.api_util.argnums_partial(f, dyn_argnums, args)
        in_avals, in_tree, keep_inputs = _abstractify(args, kwargs)
        in_type = tuple(safe_zip(in_avals, keep_inputs))
        f, out_tree = _flatten_fun(f, in_tree)
        f = annotate(f, in_type)
        if jax.__version_info__ < (0, 5, 0):
            debug_info_ = pe.debug_info(fun, in_tree, out_tree, True, 'make_jaxpr')
        with ExitStack() as stack:
            if axis_env is not None:
                stack.enter_context(extend_axis_env_nd(axis_env))
            if jax.__version_info__ < (0, 5, 0):
                jaxpr, out_type, consts = pe.trace_to_jaxpr_dynamic2(f, debug_info=debug_info_)
            else:
                jaxpr, out_type, consts = pe.trace_to_jaxpr_dynamic2(f)
        closed_jaxpr = ClosedJaxpr(jaxpr, consts)
        if return_shape:
            out_avals, _ = unzip2(out_type)
            out_shapes_flat = [jax.ShapeDtypeStruct(a.shape, a.dtype) for a in out_avals]
            return closed_jaxpr, jax.tree.unflatten(out_tree(), out_shapes_flat)
        return closed_jaxpr

    make_jaxpr_f.__module__ = "brainstate.transform"
    if hasattr(fun, "__qualname__"):
        make_jaxpr_f.__qualname__ = f"make_jaxpr({fun.__qualname__})"
    if hasattr(fun, "__name__"):
        make_jaxpr_f.__name__ = f"make_jaxpr({fun.__name__})"
    return make_jaxpr_f


def make_hashable(obj):
    """
    Convert a pytree into a hashable representation.

    Parameters
    ----------
    obj : Any
        A pytree object (list, tuple, dict, set, or JAX pytree structure).

    Returns
    -------
    Hashable
        A hashable representation of the input object. Lists become tuples,
        dicts become sorted tuples of key-value pairs, sets become frozensets,
        and other pytrees are flattened using JAX's tree utilities.
    """
    if isinstance(obj, (list, tuple)):
        return tuple(make_hashable(item) for item in obj)
    elif isinstance(obj, dict):
        return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
    elif isinstance(obj, set):
        return frozenset(make_hashable(item) for item in obj)
    else:
        # return obj
        # Use JAX's tree_util for any other pytree structures
        try:
            leaves, treedef = jax.tree.flatten(obj)
            return treedef, tuple(leaves)
        except (TypeError, ValueError):
            # Assume obj is already hashable
            return obj


class IdentitySet(MutableSet):
    """Set that compares objects by identity.

    This is a set that compares objects by identity instead of equality. It is
    useful for storing objects that are not hashable or that should be compared
    by identity.

    This is a mutable set, but it does not support the ``__hash__`` method and
    therefore cannot be used as a dictionary key or as an element of another set.
    """

    def __init__(self, iterable=None):
        self._data = {}
        if iterable is not None:
            self.update(iterable)

    def __contains__(self, value):
        return id(value) in self._data

    def __iter__(self):
        return iter(self._data.values())

    def __len__(self):
        return len(self._data)

    def add(self, value):
        self._data[id(value)] = value

    def discard(self, value):
        self._data.pop(id(value), None)

    def __repr__(self):
        return f"IdentitySet({list(repr(x) for x in self._data.values())})"

    def __str__(self):
        return f"IdentitySet({list(str(x) for x in self._data.values())})"


def constant_fold_jaxpr(jaxpr: Jaxpr):
    """
    Given a jaxpr, return a new jaxpr with all constant folding done.
    """
    return _partial_eval_jaxpr(jaxpr, {})


def _partial_eval_jaxpr(jaxpr, env):
    env = env.copy()
    new_eqns = []

    def read(var):
        if isinstance(var, Literal):
            return var.val
        else:
            return env.get(var, None)

    def read_or_self(var):
        out = read(var)
        if out is None:
            return var
        elif isinstance(out, Var):
            return out
        elif isinstance(out, Literal):
            return Literal(out.val, var.aval)
        else:
            assert not isinstance(out, Jaxpr)
            return Literal(out, var.aval)

    for eqn in jaxpr.eqns:
        vals = [read(var) for var in eqn.invars]
        if eqn.primitive.name in _constant_fold_blacklist:
            new_eqns.append(eqn)
        elif all(val is not None for val in vals):
            # go ahead and eval it
            out = _eval_eqn(eqn, vals)

            # two options: either it's a jaxpr result (partial eval) or it's a value or a list of values
            if isinstance(out, Jaxpr):
                # we need to inline this
                new_eqns.extend(out.eqns)
                out = out.outvars
            elif not isinstance(out, tuple) and not isinstance(out, list):
                out = (out,)

            for var, val in zip(eqn.outvars, out):
                assert not isinstance(val, Jaxpr)
                if isinstance(val, Literal):
                    env[var] = val.val
                else:
                    env[var] = val
        else:
            new_eqns.append(eqn)

    # now that we've eval everything, inline all the constants
    out_eqns = []
    for eqn in new_eqns:
        eqn = eqn.replace(invars=tuple(read_or_self(var) for var in eqn.invars))
        out_eqns.append(eqn)

    invars_still_used = IdentitySet()
    for eqn in out_eqns:
        for var in eqn.invars:
            invars_still_used.add(var)

    invars = tuple(var for var in jaxpr.invars if var in invars_still_used)

    # sub in any constants for outvars
    outvars = tuple(read_or_self(var) for var in jaxpr.outvars)

    return jaxpr.replace(eqns=out_eqns, outvars=outvars, invars=invars, debug_info=None)


def _eval_eqn(eqn, vals) -> Union[Jaxpr, tuple, list, jax.Array]:
    if eqn.primitive.name == "closed_call":
        assert eqn.primitive.call_primitive
        assert not eqn.primitive.map_primitive

        out = _partial_eval_jaxpr(
            eqn.params['call_jaxpr'].jaxpr,
            {
                var: val
                for var, val in
                zip(eqn.params['call_jaxpr'].jaxpr.invars, vals)
            }
        )
    elif eqn.primitive.name == "scan":
        out = eqn.primitive.bind(*vals, **eqn.params)
    else:
        out = eqn.primitive.bind(*vals, **eqn.params)
    return out


_constant_fold_blacklist = {
    'broadcast_in_dim',
    'broadcast',
}
