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
from collections.abc import Hashable, Iterable, Sequence
from contextlib import ExitStack
from typing import Any, Callable, Dict, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax._src import source_info_util
from jax.api_util import shaped_abstractify
from jax.extend.linear_util import transformation_with_aux
from jax.interpreters import partial_eval as pe

from brainstate._compatible_import import (
    ClosedJaxpr, extend_axis_env_nd, safe_map, safe_zip, unzip2, wraps, wrap_init,
)
from brainstate._state import State, StateTraceStack
from brainstate._utils import set_module_as
from brainstate.typing import PyTree
from brainstate.util import PrettyObject
from brainstate.util._cache import BoundedCache
from ._ir_optim import optimize_jaxpr

__all__ = [
    "StatefulFunction",
    "make_jaxpr",
]

AxisName = Hashable


class hashabledict(dict):
    def __hash__(self):
        return hash(tuple(sorted(self.items())))


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


def get_arg_cache_key(
    static_argnums,
    static_argnames,
    args: Tuple,
    kwargs: Dict,
    fn_to_check: Callable = None
) -> hashabledict:
    # args
    static_args, dyn_args = [], []
    for i, arg in enumerate(args):
        if i in static_argnums:
            static_args.append(arg)
        else:
            dyn_args.append(arg)
    if fn_to_check is not None:
        jax.tree.map(fn_to_check, dyn_args, is_leaf=lambda x: isinstance(x, State))
    dyn_args = jax.tree.map(shaped_abstractify, dyn_args)

    # kwargs
    static_kwargs, dyn_kwargs = [], []
    for k, v in sorted(kwargs.items()):
        if k in static_argnames:
            static_kwargs.append((k, v))
        else:
            dyn_kwargs.append((k, jax.tree.map(shaped_abstractify, v)))
    if fn_to_check is not None:
        jax.tree.map(fn_to_check, dyn_kwargs, is_leaf=lambda x: isinstance(x, State))

    # hashable
    static_args = _make_hashable(tuple(static_args))
    dyn_args = _make_hashable(tuple(dyn_args))
    static_kwargs = _make_hashable(static_kwargs)
    dyn_kwargs = _make_hashable(dyn_kwargs)

    cache_key = hashabledict(
        static_args=static_args,
        dyn_args=dyn_args,
        static_kwargs=static_kwargs,
        dyn_kwargs=dyn_kwargs,
    )
    return cache_key


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
    name : str, optional
        Name for the stateful function. Default is None.
    return_only_write : bool, optional
        If True, only return states that were written to during execution
        (not just read). This can reduce memory usage when you only care
        about modified states. Default is True.
    ir_optimizations: str or sequence of str, optional
        The IR optimizations to apply to the generated jaxpr. Can be a single
        optimization name or a sequence of names. If None, no optimizations are applied.

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
        name: Optional[str] = None,
        return_only_write: bool = True,
        ir_optimizations: Union[str, Sequence[str]] = None
    ):
        # explicit parameters
        self.fun = fun
        self.static_argnums = tuple() if static_argnums is None else _ensure_index_tuple(static_argnums)
        self.static_argnames = tuple() if static_argnames is None else _ensure_str_tuple(static_argnames)
        self.axis_env = axis_env
        self.name = name
        self.return_only_write = return_only_write
        self.ir_optimizations = ir_optimizations

        # implicit parameters - thread-safe bounded caches
        self._cached_jaxpr = BoundedCache(maxsize=128)
        self._cached_out_shapes = BoundedCache(maxsize=128)
        self._cached_jaxpr_out_tree = BoundedCache(maxsize=128)
        self._cached_state_trace = BoundedCache(maxsize=128)
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

        cache_key = get_arg_cache_key(
            self.static_argnums,
            self.static_argnames,
            args,
            kwargs,
            self._check_input_ouput
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

        # static args
        cache_key = self.get_arg_cache_key(*args, **kwargs)

        if cache_key not in self._cached_state_trace:
            try:
                # kwargs separation
                static_kwargs, dyn_kwargs = {}, {}
                for k, v in kwargs.items():
                    if k in self.static_argnames:
                        static_kwargs[k] = v
                    else:
                        dyn_kwargs[k] = v

                # args separation
                dyn_args = tuple(args[i] for i in range(len(args)) if i not in self.static_argnums)

                # jaxpr
                if jax.__version_info__ >= (0, 8, 2):
                    jaxpr, (out_shapes, state_shapes) = jax.make_jaxpr(
                        functools.partial(
                            self._wrapped_fun_to_eval,
                            cache_key,
                            static_kwargs,
                        ),
                        static_argnums=self.static_argnums,
                        axis_env=self.axis_env,
                        return_shape=True,
                    )(*args, **dyn_kwargs)
                else:
                    jaxpr, (out_shapes, state_shapes) = _make_jaxpr(
                        functools.partial(
                            self._wrapped_fun_to_eval,
                            cache_key,
                            static_kwargs,
                        ),
                        static_argnums=self.static_argnums,
                        axis_env=self.axis_env,
                        return_shape=True,
                        ir_optimizations=self.ir_optimizations,
                    )(*args, **dyn_kwargs)

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
        if jax.config.jax_disable_jit:
            return self.debug_call(state_vals, *args, **kwargs)

        else:
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
                raise ValueError(
                    f'State length mismatch in output: expected '
                    f'{len(state_vals)} states, got {len(new_state_vals)}'
                )
            return new_state_vals, out

    def debug_call(self, state_vals, *args, **kwargs) -> Any:
        cache_key = self.get_arg_cache_key(*args, **kwargs)
        states: Sequence[State] = self.get_states_by_cache(cache_key)
        if len(state_vals) != len(states):
            raise ValueError(f'State length mismatch: expected {len(states)} states, got {len(state_vals)}')
        for st, val in zip(states, state_vals):
            st.restore_value(val)
        out = self.fun(*args, **kwargs)
        state_trace = self.get_state_trace_by_cache(cache_key)
        new_state_vals = (
            state_trace.get_write_state_values(True)
            if self.return_only_write else
            state_trace.get_state_values()
        )
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
        if jax.config.jax_disable_jit:
            state_vals, out = self.debug_call(state_trace.get_state_values(), *args, **kwargs)
        else:
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


def _check_callable(fun):
    # In Python 3.10+, the only thing stopping us from supporting static methods
    # is that we can't take weak references to them, which the C++ JIT requires.
    if isinstance(fun, staticmethod):
        raise TypeError(f"staticmethod arguments are not supported, got {fun}")
    if not callable(fun):
        raise TypeError(f"Expected a callable value, got {fun}")
    if inspect.isgeneratorfunction(fun):
        raise TypeError(f"Expected a function, got a generator function: {fun}")


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
    ir_optimizations: Union[str, Sequence[str]] = None,
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
    ir_optimizations: str or sequence of str, optional
        A string or sequence of strings specifying IR optimizations to apply
        during jaxpr tracing. If None, no optimizations are applied.

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
    from jax._src.traceback_util import api_boundary
    from jax._src.linear_util import annotate

    _check_callable(fun)
    static_argnums = _ensure_index_tuple(static_argnums)

    def _abstractify(args, kwargs):
        flat_args, in_tree = jax.tree.flatten((args, kwargs))
        return map(shaped_abstractify, flat_args), in_tree, [True] * len(flat_args)

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
        with ExitStack() as stack:
            if axis_env is not None:
                stack.enter_context(extend_axis_env_nd(axis_env))
            jaxpr, out_type, consts = pe.trace_to_jaxpr_dynamic2(f)
        closed_jaxpr = ClosedJaxpr(jaxpr, consts)
        if ir_optimizations is not None:
            closed_jaxpr = closed_jaxpr.replace(
                jaxpr=optimize_jaxpr(closed_jaxpr.jaxpr, optimizations=ir_optimizations)
            )
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


def _make_hashable(obj):
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
        return tuple(_make_hashable(item) for item in obj)
    elif isinstance(obj, dict):
        return tuple(sorted((k, _make_hashable(v)) for k, v in obj.items()))
    elif isinstance(obj, set):
        return frozenset(_make_hashable(item) for item in obj)
    else:
        # return obj
        # Use JAX's tree_util for any other pytree structures
        try:
            leaves, treedef = jax.tree.flatten(obj)
            return treedef, tuple(leaves)
        except (TypeError, ValueError):
            # Assume obj is already hashable
            return obj
