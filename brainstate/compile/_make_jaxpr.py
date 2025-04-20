# Copyright 2024 BDP Ecosystem Limited. All Rights Reserved.
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

from __future__ import annotations

import functools
import inspect
import jax
import operator
from collections.abc import Hashable, Iterable, Sequence
from contextlib import ExitStack
from jax._src import source_info_util
from jax._src.linear_util import annotate
from jax._src.traceback_util import api_boundary
from jax.api_util import shaped_abstractify
from jax.extend.linear_util import transformation_with_aux, wrap_init
from jax.interpreters import partial_eval as pe
from jax.util import wraps
from typing import Any, Callable, Tuple, Union, Dict, Optional

from brainstate._state import State, StateTraceStack
from brainstate._utils import set_module_as
from brainstate.typing import PyTree
from brainstate.util import PrettyObject

if jax.__version_info__ < (0, 4, 38):
    from jax.core import ClosedJaxpr
else:
    from jax.extend.core import ClosedJaxpr

AxisName = Hashable

__all__ = [
    "StatefulFunction",
    "make_jaxpr",
]


def _ensure_index_tuple(x: Any) -> tuple[int, ...]:
    """Convert x to a tuple of indices."""
    x = jax.core.concrete_or_error(None, x, "expected a static index or sequence of indices.")
    try:
        return (operator.index(x),)
    except TypeError:
        return tuple(jax.util.safe_map(operator.index, x))


def _new_arg_fn(frame, trace, aval):
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


def _new_jax_trace():
    main = jax.core.thread_local_state.trace_state.trace_stack.stack[-1]
    frame = main.jaxpr_stack[-1]
    trace = pe.DynamicJaxprTrace(main, jax.core.cur_sublevel())
    return frame, trace


def _init_state_trace_stack(name) -> StateTraceStack:
    state_trace: StateTraceStack = StateTraceStack(name=name)

    if jax.__version_info__ < (0, 4, 36):
        # Should be within the calling of ``jax.make_jaxpr()``
        frame, trace = _new_jax_trace()
        # Set the function to transform the new argument to a tracer
        state_trace.set_new_arg(functools.partial(_new_arg_fn, frame, trace))
        return state_trace

    else:
        trace = jax.core.trace_ctx.trace
        state_trace.set_new_arg(trace.new_arg)
        return state_trace


class StatefulFunction(PrettyObject):
    """
    A wrapper class for a function that collects the states that are read and written by the function. The states are
    collected by the function and returned as a StateDictManager instance. The StateDictManager instance can be used to
    manage the states in the JAX program. The class provides a function called `states` that returns the states
    that are read and written by the function. The class provides a function called `to_state_manager` that returns
    a StateDictManager instance that contains the states that are read and written by the function. The class provides
    a function called `__call__` that wraps the function and returns the states that are read and written by the
    function and the output of the function.

    Args:
      fun: The function whose ``jaxpr`` is to be computed. Its positional
        arguments and return value should be arrays, scalars, or standard Python
        containers (tuple/list/dict) thereof.
      static_argnums: See the :py:func:`jax.jit` docstring.
      axis_env: Optional, a sequence of pairs where the first element is an axis
          name and the second element is a positive integer representing the size of
          the mapped axis with that name. This parameter is useful when lowering
          functions that involve parallel communication collectives, and it
          specifies the axis name/size environment that would be set up by
          applications of :py:func:`jax.pmap`.
      abstracted_axes: Optional, a pytree with the same structure as the input
          arguments to ``fun``. The leaves of the pytree can be either None or a
          dict with axis names as keys and integers as values. If the leaf is None,
          then the corresponding axis is not abstracted. If the leaf is a dict, then
          the corresponding axis is abstracted, and the dict specifies the axis name
          and size. The abstracted axes are used to infer the input type of the
          function. If None, then all axes are abstracted.
      state_returns: Optional, a string or a tuple of strings. The default is
          ``('read', 'write')``. The strings specify the categories of states to be
          returned by the wrapped function. The categories are ``'read'`` and
          ``'write'``. If the category is ``'read'``, then the wrapped function
          returns the states that are read by the function. If the category is
          ``'write'``, then the wrapped function returns the states that are written
          by the function. If the category is ``'read'`` and ``'write'``, then the
          wrapped function returns both the read and write states.

    """
    __module__ = "brainstate.compile"

    def __init__(
        self,
        fun: Callable,
        static_argnums: Union[int, Iterable[int]] = (),
        axis_env: Optional[Sequence[tuple[Hashable, int]]] = None,
        abstracted_axes: Optional[Any] = None,
        state_returns: Union[str, Tuple[str, ...]] = ('read', 'write'),
        cache_type: Optional[str] = None,
        name: Optional[str] = None,
    ):
        # explicit parameters
        self.fun = fun
        self.static_argnums = _ensure_index_tuple(tuple() if static_argnums is None else static_argnums)
        self.axis_env = axis_env
        self.abstracted_axes = abstracted_axes
        self.state_returns = tuple(state_returns) if isinstance(state_returns, (tuple, list)) else (state_returns,)
        assert cache_type in [None, 'jit']
        self.name = name

        # implicit parameters
        self.cache_type = cache_type
        self._cached_jaxpr: Dict[Any, ClosedJaxpr] = dict()
        self._cached_out_shapes: Dict[Any, PyTree] = dict()
        self._cached_jaxpr_out_tree: Dict[Any, PyTree] = dict()
        self._cached_state_trace: Dict[Any, StateTraceStack] = dict()

    def __pretty_repr_item__(self, k, v):
        if k.startswith('_'):
            return None
        return k, v

    def get_jaxpr(self, cache_key: Hashable = ()) -> ClosedJaxpr:
        """
        Read the JAX Jaxpr representation of the function.

        Args:
          cache_key: The hashable key.

        Returns:
          The JAX Jaxpr representation of the function.
        """
        if cache_key not in self._cached_jaxpr:
            raise ValueError(f"the function is not called with the static arguments: {cache_key}")
        return self._cached_jaxpr[cache_key]

    def get_out_shapes(self, cache_key: Hashable = ()) -> PyTree:
        """
        Read the output shapes of the function.

        Args:
          cache_key: The hashable key.

        Returns:
          The output shapes of the function.
        """
        if cache_key not in self._cached_out_shapes:
            raise ValueError(f"the function is not called with the static arguments: {cache_key}")
        return self._cached_out_shapes[cache_key]

    def get_out_treedef(self, cache_key: Hashable = ()) -> PyTree:
        """
        Read the output tree of the function.

        Args:
          cache_key: The hashable key.

        Returns:
          The output tree of the function.
        """
        if cache_key not in self._cached_jaxpr_out_tree:
            raise ValueError(f"the function is not called with the static arguments: {cache_key}")
        return self._cached_jaxpr_out_tree[cache_key]

    def get_state_trace(self, cache_key: Hashable = ()) -> StateTraceStack:
        """
        Read the state trace of the function.

        Args:
          cache_key: The hashable key.

        Returns:
          The state trace of the function.
        """
        if cache_key not in self._cached_state_trace:
            raise ValueError(f"the function is not called with the static arguments: {cache_key}")
        return self._cached_state_trace[cache_key]

    def get_states(self, cache_key: Hashable = ()) -> Tuple[State, ...]:
        """
        Read the states that are read and written by the function.

        Args:
          cache_key: The hashable key.

        Returns:
          The states that are read and written by the function.
        """
        return tuple(self.get_state_trace(cache_key).states)

    def get_read_states(self, cache_key: Hashable = ()) -> Tuple[State, ...]:
        """
        Read the states that are read by the function.

        Args:
          cache_key: The hashable key.

        Returns:
          The states that are read by the function.
        """
        return self.get_state_trace(cache_key).get_read_states()

    def get_write_states(self, cache_key: Hashable = ()) -> Tuple[State, ...]:
        """
        Read the states that are written by the function.

        Args:
          cache_key: The hashable key.

        Returns:
          The states that are written by the function.
        """
        return self.get_state_trace(cache_key).get_write_states()

    def get_arg_cache_key(self, *args, **kwargs) -> Tuple:
        """
        Get the static arguments from the arguments.

        Args:
          *args: The arguments to the function.

        Returns:
          The static arguments.
        """
        if self.cache_type == 'jit':
            static_args, dyn_args = [], []
            for i, arg in enumerate(args):
                if i in self.static_argnums:
                    static_args.append(arg)
                else:
                    dyn_args.append(arg)
            dyn_args = jax.tree.map(shaped_abstractify, jax.tree.leaves(dyn_args))
            dyn_kwargs = jax.tree.map(shaped_abstractify, jax.tree.leaves(kwargs))
            return tuple([tuple(static_args), tuple(dyn_args), tuple(dyn_kwargs)])
        elif self.cache_type is None:
            num_arg = len(args)
            return tuple(args[i] for i in self.static_argnums if i < num_arg)
        else:
            raise ValueError(f"Invalid cache type: {self.cache_type}")

    def compile_function_and_get_states(self, *args, **kwargs) -> Tuple[State, ...]:
        """
        Compile the function, and get the states that are read and written by this function.

        Args:
          *args: The arguments to the function.
          **kwargs: The keyword arguments to the function.

        Returns:
          The states that are read and written by the function.
        """
        cache_key = self.get_arg_cache_key(*args, **kwargs)
        if cache_key not in self._cached_state_trace:
            self.make_jaxpr(*args, **kwargs)
        return self.get_states(cache_key)

    def compile_function_and_get_state_trace(
        self, *args, return_only_write: bool = False, **kwargs
    ) -> StateTraceStack:
        """
        Compile the function, and get the states that are read and written by this function.

        Args:
          *args: The arguments to the function.
          **kwargs: The keyword arguments to the function.
          return_only_write: If True, only return the states that are written by the function.

        Returns:
          The state trace stack.
        """
        cache_key = self.get_arg_cache_key(*args, **kwargs)
        if cache_key not in self._cached_state_trace:
            self.make_jaxpr(*args, **kwargs, return_only_write=return_only_write)
        return self.get_state_trace(cache_key)

    def clear_cache(self) -> None:
        """
        Clear the compilation cache.
        """
        self._cached_jaxpr.clear()
        self._cached_out_shapes.clear()
        self._cached_jaxpr_out_tree.clear()
        self._cached_state_trace.clear()

    def _wrapped_fun_to_eval(
        self, cache_key, *args, return_only_write: bool = False, **kwargs,
    ) -> Tuple[Any, Tuple[State, ...]]:
        """
        Wrap the function and return the states that are read and written by the function and the output of the function.

        Args:
          *args: The arguments to the function.
          **kwargs: The keyword arguments to the function.

        Returns:
          A tuple of the states that are read and written by the function and the output of the function.
        """
        # state trace
        state_trace = _init_state_trace_stack(self.name)
        self._cached_state_trace[cache_key] = state_trace
        with state_trace:
            out = self.fun(*args, **kwargs)
            state_values = (
                state_trace.get_write_state_values(True)
                if return_only_write else
                state_trace.get_state_values()
            )
        state_trace.recovery_original_values()

        # State instance as functional returns is not allowed.
        # Checking whether the states are returned.
        for leaf in jax.tree.leaves(out):
            if isinstance(leaf, State):
                leaf.raise_error_with_source_info(ValueError(f"State object is not allowed to be returned: {leaf}"))
        return out, state_values

    def make_jaxpr(self, *args, return_only_write: bool = False, **kwargs):
        """Creates a function that produces its jaxpr given example args.

        A ``ClosedJaxpr`` representation of ``fun`` on those arguments. If the
        argument ``return_shape`` is ``True``, then the returned function instead
        returns a pair where the first element is the ``ClosedJaxpr``
        representation of ``fun`` and the second element is a pytree representing
        the structure, shape, dtypes, and named shapes of the output of ``fun``.

        Args:
          *args: The arguments to the function.
          **kwargs: The keyword arguments to the function.
        """

        # static args
        cache_key = self.get_arg_cache_key(*args, **kwargs)

        if cache_key not in self._cached_state_trace:
            try:
                # jaxpr
                jaxpr, (out_shapes, state_shapes) = _make_jaxpr(
                    functools.partial(
                        self._wrapped_fun_to_eval,
                        cache_key,
                        return_only_write=return_only_write
                    ),
                    static_argnums=self.static_argnums,
                    axis_env=self.axis_env,
                    return_shape=True,
                    abstracted_axes=self.abstracted_axes
                )(*args, **kwargs)
                # returns
                self._cached_jaxpr_out_tree[cache_key] = jax.tree.structure((out_shapes, state_shapes))
                self._cached_out_shapes[cache_key] = (out_shapes, state_shapes)
                self._cached_jaxpr[cache_key] = jaxpr

            except Exception as e:
                try:
                    self._cached_state_trace.pop(cache_key)
                except KeyError:
                    pass
                raise e

        return self

    def jaxpr_call(self, state_vals, *args, **kwargs) -> Any:
        """
        Call the function at the JAX Jaxpr level.

        Args:
          state_vals: The state values.
          *args: The arguments to the function.
          **kwargs: The keyword arguments to the function.

        Returns:
          State values and the function output.
        """
        # state checking
        cache_key = self.get_arg_cache_key(*args, **kwargs)
        states: Sequence[State] = self.get_states(cache_key)
        assert len(state_vals) == len(states), 'State length mismatch.'

        # parameters
        args = tuple(args[i] for i in range(len(args)) if i not in self.static_argnums)
        args = jax.tree.flatten((args, kwargs, state_vals))[0]

        # calling the function,
        # note that this function always returns state values
        # that both write and read by the function
        closed_jaxpr = self.get_jaxpr(cache_key)
        out_treedef = self.get_out_treedef(cache_key)
        jaxpr_outs = jax.core.eval_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.consts, *args)

        # output processing
        out, new_state_vals = out_treedef.unflatten(jaxpr_outs)
        assert len(new_state_vals) == len(state_vals), 'State length mismatch.'
        return new_state_vals, out

    def jaxpr_call_auto(self, *args, **kwargs) -> Any:
        """
        Call the function at the JAX Jaxpr level with automatic state management.

        Args:
          *args: The arguments to the function.
          **kwargs: The keyword arguments to the function.

        Returns:
          The output of the function.
        """
        state_trace = self.get_state_trace(self.get_arg_cache_key(*args, **kwargs))
        state_vals, out = self.jaxpr_call([st.value for st in state_trace.states], *args, **kwargs)
        state_trace.assign_state_vals(state_vals)
        return out


@set_module_as("brainstate.compile")
def make_jaxpr(
    fun: Callable,
    static_argnums: Union[int, Iterable[int]] = (),
    axis_env: Optional[Sequence[tuple[Hashable, int]]] = None,
    return_shape: bool = False,
    abstracted_axes: Optional[Any] = None,
    state_returns: Union[str, Tuple[str, ...]] = ('read', 'write')
) -> Callable[..., (Tuple[ClosedJaxpr, Tuple[State, ...]] |
                    Tuple[ClosedJaxpr, Tuple[State, ...], PyTree])]:
    """
    Creates a function that produces its jaxpr given example args.

    Args:
      fun: The function whose ``jaxpr`` is to be computed. Its positional
        arguments and return value should be arrays, scalars, or standard Python
        containers (tuple/list/dict) thereof.
      static_argnums: See the :py:func:`jax.jit` docstring.
      axis_env: Optional, a sequence of pairs where the first element is an axis
        name and the second element is a positive integer representing the size of
        the mapped axis with that name. This parameter is useful when lowering
        functions that involve parallel communication collectives, and it
        specifies the axis name/size environment that would be set up by
        applications of :py:func:`jax.pmap`.
      return_shape: Optional boolean, defaults to ``False``. If ``True``, the
        wrapped function returns a pair where the first element is the XLA
        computation and the second element is a pytree with the same structure as
        the output of ``fun`` and where the leaves are objects with ``shape``,
        ``dtype``, and ``named_shape`` attributes representing the corresponding
        types of the output leaves.
      abstracted_axes: Optional, a pytree with the same structure as the input
        arguments to ``fun``. The leaves of the pytree can be either None or a
        dict with axis names as keys and integers as values. If the leaf is None,
        then the corresponding axis is not abstracted. If the leaf is a dict, then
        the corresponding axis is abstracted, and the dict specifies the axis name
        and size. The abstracted axes are used to infer the input type of the
        function. If None, then all axes are abstracted.
      state_returns: Optional, a string or a tuple of strings. The default is
        ``('read', 'write')``. The strings specify the categories of states to be
        returned by the wrapped function. The categories are ``'read'`` and
        ``'write'``. If the category is ``'read'``, then the wrapped function
        returns the states that are read by the function. If the category is
        ``'write'``, then the wrapped function returns the states that are written
        by the function. If the category is ``'read'`` and ``'write'``, then the
        wrapped function returns both the read and write states.


    Returns:
      A wrapped version of ``fun`` that when applied to example arguments returns
      a ``ClosedJaxpr`` representation of ``fun`` on those arguments. If the
      argument ``return_shape`` is ``True``, then the returned function instead
      returns a pair where the first element is the ``ClosedJaxpr``
      representation of ``fun`` and the second element is a pytree representing
      the structure, shape, dtypes, and named shapes of the output of ``fun``.

    A ``jaxpr`` is JAX's intermediate representation for program traces. The
    ``jaxpr`` language is based on the simply-typed first-order lambda calculus
    with let-bindings. :py:func:`make_jaxpr` adapts a function to return its
    ``jaxpr``, which we can inspect to understand what JAX is doing internally.
    The ``jaxpr`` returned is a trace of ``fun`` abstracted to
    :py:class:`ShapedArray` level. Other levels of abstraction exist internally.

    We do not describe the semantics of the ``jaxpr`` language in detail here, but
    instead give a few examples.

    >>> import jax
    >>> import brainstate as bst
    >>>
    >>> def f(x): return jax.numpy.sin(jax.numpy.cos(x))
    >>> print(f(3.0))
    -0.83602
    >>> jaxpr, states = bst.compile.make_jaxpr(f)(3.0)
    >>> jaxpr
    { lambda ; a:f32[]. let b:f32[] = cos a; c:f32[] = sin b in (c,) }
    >>> jaxpr, states = bst.compile.make_jaxpr(jax.grad(f))(3.0)
    >>> jaxpr
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

    stateful_fun = StatefulFunction(
        fun,
        static_argnums=static_argnums,
        axis_env=axis_env,
        abstracted_axes=abstracted_axes,
        state_returns=state_returns,
        name='make_jaxpr'

    )

    @wraps(fun)
    def make_jaxpr_f(*args, **kwargs):
        stateful_fun.make_jaxpr(*args, **kwargs)
        cache_key = stateful_fun.get_arg_cache_key(*args, **kwargs)
        if return_shape:
            return (stateful_fun.get_jaxpr(cache_key),
                    stateful_fun.get_states(cache_key),
                    stateful_fun.get_out_shapes(cache_key)[0])
        else:
            return (stateful_fun.get_jaxpr(cache_key),
                    stateful_fun.get_states(cache_key))

    # wrapped jaxpr builder function
    make_jaxpr_f.__module__ = "brainstate.compile"
    if hasattr(fun, "__qualname__"):
        make_jaxpr_f.__qualname__ = f"make_jaxpr({fun.__qualname__})"
    if hasattr(fun, "__name__"):
        make_jaxpr_f.__name__ = f"make_jaxpr({fun.__name__})"
    return make_jaxpr_f


def _check_callable(fun):
    # In Python 3.10+, the only thing stopping us from supporting staticmethods
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
    """Creates a function that produces its jaxpr given example args.

    Args:
      fun: The function whose ``jaxpr`` is to be computed. Its positional
        arguments and return value should be arrays, scalars, or standard Python
        containers (tuple/list/dict) thereof.
      static_argnums: See the :py:func:`jax.jit` docstring.
      axis_env: Optional, a sequence of pairs where the first element is an axis
        name and the second element is a positive integer representing the size of
        the mapped axis with that name. This parameter is useful when lowering
        functions that involve parallel communication collectives, and it
        specifies the axis name/size environment that would be set up by
        applications of :py:func:`jax.pmap`.
      return_shape: Optional boolean, defaults to ``False``. If ``True``, the
        wrapped function returns a pair where the first element is the
        ``ClosedJaxpr`` representation of ``fun`` and the second element is a
        pytree with the same structure as the output of ``fun`` and where the
        leaves are objects with ``shape``, ``dtype``, and ``named_shape``
        attributes representing the corresponding types of the output leaves.

    Returns:
      A wrapped version of ``fun`` that when applied to example arguments returns
      a ``ClosedJaxpr`` representation of ``fun`` on those arguments. If the
      argument ``return_shape`` is ``True``, then the returned function instead
      returns a pair where the first element is the ``ClosedJaxpr``
      representation of ``fun`` and the second element is a pytree representing
      the structure, shape, dtypes, and named shapes of the output of ``fun``.

    A ``jaxpr`` is JAX's intermediate representation for program traces. The
    ``jaxpr`` language is based on the simply-typed first-order lambda calculus
    with let-bindings. :py:func:`make_jaxpr` adapts a function to return its
    ``jaxpr``, which we can inspect to understand what JAX is doing internally.
    The ``jaxpr`` returned is a trace of ``fun`` abstracted to
    :py:class:`ShapedArray` level. Other levels of abstraction exist internally.

    We do not describe the semantics of the ``jaxpr`` language in detail here, but
    instead give a few examples.

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
            in_avals, keep_inputs = jax.util.unzip2(in_type)
            return in_avals, in_tree, keep_inputs

    @wraps(fun)
    @api_boundary
    def make_jaxpr_f(*args, **kwargs):
        f = wrap_init(fun)
        if static_argnums:
            dyn_argnums = [i for i in range(len(args)) if i not in static_argnums]
            f, args = jax.api_util.argnums_partial(f, dyn_argnums, args)
        in_avals, in_tree, keep_inputs = _abstractify(args, kwargs)
        in_type = tuple(jax.util.safe_zip(in_avals, keep_inputs))
        f, out_tree = _flatten_fun(f, in_tree)
        f = annotate(f, in_type)
        if jax.__version_info__ < (0, 5, 0):
            debug_info = pe.debug_info(fun, in_tree, out_tree, True, 'make_jaxpr')
        with ExitStack() as stack:
            if axis_env is not None:
                stack.enter_context(jax.core.extend_axis_env_nd(axis_env))
            if jax.__version_info__ < (0, 5, 0):
                jaxpr, out_type, consts = pe.trace_to_jaxpr_dynamic2(f, debug_info=debug_info)
            else:
                jaxpr, out_type, consts = pe.trace_to_jaxpr_dynamic2(f)
        closed_jaxpr = ClosedJaxpr(jaxpr, consts)
        if return_shape:
            out_avals, _ = jax.util.unzip2(out_type)
            out_shapes_flat = [jax.ShapeDtypeStruct(a.shape, a.dtype) for a in out_avals]
            return closed_jaxpr, jax.tree.unflatten(out_tree(), out_shapes_flat)
        return closed_jaxpr

    make_jaxpr_f.__module__ = "brainstate.compile"
    if hasattr(fun, "__qualname__"):
        make_jaxpr_f.__qualname__ = f"make_jaxpr({fun.__qualname__})"
    if hasattr(fun, "__name__"):
        make_jaxpr_f.__name__ = f"make_jaxpr({fun.__name__})"
    return make_jaxpr_f
