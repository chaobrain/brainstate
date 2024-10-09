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
import operator
from collections.abc import Hashable, Iterable, Sequence
from contextlib import ExitStack
from typing import Any, Callable, Tuple, Union, Dict, Optional

import jax
from jax._src import source_info_util
from jax._src.linear_util import annotate
from jax._src.traceback_util import api_boundary
from jax.extend.linear_util import transformation_with_aux, wrap_init
from jax.interpreters import partial_eval as pe
from jax.interpreters.xla import abstractify
from jax.util import wraps

from brainstate._state import State, StateTrace
from brainstate._utils import set_module_as
from brainstate.typing import PyTree

AxisName = Hashable

__all__ = [
  "StatefulFunction",
  "make_jaxpr",
]


def _assign_state_values(states, state_vals) -> None:
  """
  Assign the state values to the states.

  Args:
    states: The states.
    state_vals: The state values.
  """
  assert len(states) == len(state_vals), f'State length mismatch. {len(states)} != {len(state_vals)}.'
  for st, val in zip(states, state_vals):
    st.value = val


def _ensure_index_tuple(x: Any) -> tuple[int, ...]:
  """Convert x to a tuple of indices."""
  x = jax.core.concrete_or_error(None, x, "expected a static index or sequence of indices.")
  try:
    return (operator.index(x),)
  except TypeError:
    return tuple(jax.util.safe_map(operator.index, x))


def _new_arg(frame, trace, aval):
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


def wrapped_abstractify(x: Any) -> Any:
  """
  Abstractify the input.

  Args:
    x: The input.

  Returns:
    The abstractified input.
  """
  if isinstance(x, pe.DynamicJaxprTracer):
    return jax.core.ShapedArray(x.aval.shape, x.aval.dtype, weak_type=x.aval.weak_type)
  return abstractify(x)


class StatefulFunction(object):
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
  ):
    # explicit parameters
    self.fun = fun
    self.static_argnums = _ensure_index_tuple(tuple() if static_argnums is None else static_argnums)
    self.axis_env = axis_env
    self.abstracted_axes = abstracted_axes
    self.state_returns = tuple(state_returns) if isinstance(state_returns, (tuple, list)) else (state_returns,)
    assert cache_type in [None, 'jit']
    self.cache_type = cache_type

    # implicit parameters
    self._jaxpr: Dict[Any, jax.core.ClosedJaxpr] = dict()
    self._out_shapes: Dict[Any, PyTree] = dict()
    self._jaxpr_out_tree: Dict[Any, PyTree] = dict()
    self._state_trace: Dict[Any, StateTrace] = dict()

  def __repr__(self) -> str:
    return (f"{self.__class__.__name__}({self.fun}, "
            f"static_argnums={self.static_argnums}, "
            f"axis_env={self.axis_env}, "
            f"abstracted_axes={self.abstracted_axes}, "
            f"state_returns={self.state_returns})")

  def get_jaxpr(self, cache_key: Hashable = ()) -> jax.core.ClosedJaxpr:
    """
    Read the JAX Jaxpr representation of the function.

    Args:
      cache_key: The hashable key.

    Returns:
      The JAX Jaxpr representation of the function.
    """
    if cache_key not in self._jaxpr:
      raise ValueError(f"the function is not called with the static arguments: {cache_key}")
    return self._jaxpr[cache_key]

  def get_out_shapes(self, cache_key: Hashable = ()) -> PyTree:
    """
    Read the output shapes of the function.

    Args:
      cache_key: The hashable key.

    Returns:
      The output shapes of the function.
    """
    if cache_key not in self._out_shapes:
      raise ValueError(f"the function is not called with the static arguments: {cache_key}")
    return self._out_shapes[cache_key]

  def get_out_treedef(self, cache_key: Hashable = ()) -> PyTree:
    """
    Read the output tree of the function.

    Args:
      cache_key: The hashable key.

    Returns:
      The output tree of the function.
    """
    if cache_key not in self._jaxpr_out_tree:
      raise ValueError(f"the function is not called with the static arguments: {cache_key}")
    return self._jaxpr_out_tree[cache_key]

  def get_states(self, cache_key: Hashable = ()) -> Tuple[State, ...]:
    """
    Read the states that are read and written by the function.

    Args:
      cache_key: The hashable key.

    Returns:
      The states that are read and written by the function.
    """
    if cache_key not in self._state_trace:
      raise ValueError(f"the function is not called with the static arguments: {cache_key}")
    return tuple(self._state_trace[cache_key].states)

  def get_read_states(self, cache_key: Hashable = ()) -> Tuple[State, ...]:
    """
    Read the states that are read by the function.

    Args:
      cache_key: The hashable key.

    Returns:
      The states that are read by the function.
    """
    _state_trace = self._state_trace[cache_key]
    return tuple([st for st, ty in zip(_state_trace.states, _state_trace.types) if ty == 'read'])

  def get_write_states(self, cache_key: Hashable = ()) -> Tuple[State, ...]:
    """
    Read the states that are written by the function.

    Args:
      cache_key: The hashable key.

    Returns:
      The states that are written by the function.
    """
    state_trace = self._state_trace[cache_key]
    return tuple([st for st, ty in zip(state_trace.states, state_trace.types) if ty == 'write'])

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
      dyn_args = jax.tree.map(wrapped_abstractify, jax.tree.leaves(dyn_args))
      dyn_kwargs = jax.tree.map(wrapped_abstractify, jax.tree.leaves(kwargs))
      return tuple([tuple(static_args), tuple(dyn_args), tuple(dyn_kwargs)])
    elif self.cache_type is None:
      num_arg = len(args)
      return tuple(args[i] for i in self.static_argnums if i < num_arg)
    else:
      raise ValueError(f"Invalid cache type: {self.cache_type}")

  def compile_and_get_states_by_static_args(self, *args, **kwargs) -> Tuple[State, ...]:
    """
    Get the states that are read and written by the function.

    Args:
      *args: The arguments to the function.
      **kwargs: The keyword arguments to the function.

    Returns:
      The states that are read and written by the function.
    """
    cache_key = self.get_arg_cache_key(*args, **kwargs)
    if cache_key not in self._state_trace:
      self.make_jaxpr(*args, **kwargs)
    return self.get_states(cache_key)

  def clear_cache(self) -> None:
    """
    Clear the compilation cache.
    """
    self._jaxpr.clear()
    self._out_shapes.clear()
    self._jaxpr_out_tree.clear()
    self._state_trace.clear()

  @staticmethod
  def _init_trace_and_newarg() -> StateTrace:
    # Should be within the calling of ``jax.make_jaxpr()``
    state_trace: StateTrace = StateTrace()
    main = jax.core.thread_local_state.trace_state.trace_stack.stack[-1]
    frame = main.jaxpr_stack[-1]
    trace = pe.DynamicJaxprTrace(main, jax.core.cur_sublevel())
    state_trace.set_new_arg(functools.partial(_new_arg, frame, trace))
    return state_trace

  def _wrapped_fun_to_eval(self, cache_key, *args, **kwargs) -> Tuple[Any, Tuple[State, ...]]:
    """
    Wrap the function and return the states that are read and written by the function and the output of the function.

    Args:
      *args: The arguments to the function.
      **kwargs: The keyword arguments to the function.

    Returns:
      A tuple of the states that are read and written by the function and the output of the function.
    """
    # state trace
    _state_trace = self._init_trace_and_newarg()
    self._state_trace[cache_key] = _state_trace
    with _state_trace:
      out = self.fun(*args, **kwargs)
      state_values = _state_trace.collect_values('read', 'write', check_val_tree=True)
    _state_trace.recovery_original_values()

    # State instance as functional returns is not allowed.
    # Checking whether the states are returned.
    for leaf in jax.tree.leaves(out):
      if isinstance(leaf, State):
        leaf._raise_error_with_source_info(ValueError(f"State object is not allowed to be returned: {leaf}"))
    return out, state_values

  def make_jaxpr(self, *args, **kwargs):
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

    if cache_key not in self._state_trace:
      try:
        # jaxpr
        jaxpr, (out_shapes, state_shapes) = _make_jaxpr(
          functools.partial(self._wrapped_fun_to_eval, cache_key),
          static_argnums=self.static_argnums,
          axis_env=self.axis_env,
          return_shape=True,
          abstracted_axes=self.abstracted_axes
        )(*args, **kwargs)

        # returns
        self._jaxpr_out_tree[cache_key] = jax.tree.structure((out_shapes, state_shapes))
        self._out_shapes[cache_key] = (out_shapes, state_shapes)
        self._jaxpr[cache_key] = jaxpr
      except Exception as e:
        try:
          self._state_trace.pop(cache_key)
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
    # # No need to check, because the make_jaxpr() has been checked whether the value's tree is correct.
    # for val, st in zip(state_vals, states):  # check state's value tree structure
    #   st._check_value_tree(val)

    # parameters
    args = tuple(args[i] for i in range(len(args)) if i not in self.static_argnums)
    args = jax.tree.flatten((args, kwargs, state_vals))[0]

    # calling the function
    closed_jaxpr = self.get_jaxpr(cache_key)
    out_treedef = self.get_out_treedef(cache_key)
    jaxpr_outs = jax.core.eval_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.consts, *args)

    # output processing
    out, new_state_vals = out_treedef.unflatten(jaxpr_outs)
    assert len(new_state_vals) == len(state_vals), 'State length mismatch.'
    # # No need to check, because the make_jaxpr() has been checked whether the value's tree is correct.
    # for val, st in zip(new_state_vals, states):  # check state's value tree structure
    #   st._check_value_tree(val)
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
    cache_key = self.get_arg_cache_key(*args, **kwargs)
    states = self.get_states(cache_key)
    state_vals, out = self.jaxpr_call([st.value for st in states], *args, **kwargs)
    for st, val in zip(states, state_vals):
      st.value = val
    return out


@set_module_as("brainstate.compile")
def make_jaxpr(
    fun: Callable,
    static_argnums: Union[int, Iterable[int]] = (),
    axis_env: Optional[Sequence[tuple[Hashable, int]]] = None,
    return_shape: bool = False,
    abstracted_axes: Optional[Any] = None,
    state_returns: Union[str, Tuple[str, ...]] = ('read', 'write')
) -> Callable[..., (Tuple[jax.core.ClosedJaxpr, Tuple[State, ...]] |
                    Tuple[jax.core.ClosedJaxpr, Tuple[State, ...], PyTree])]:
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

  stateful_fun = StatefulFunction(fun, static_argnums, axis_env, abstracted_axes, state_returns)

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
) -> Callable[..., (jax.core.ClosedJaxpr | tuple[jax.core.ClosedJaxpr, Any])]:
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
      return map(jax.api_util.shaped_abstractify, flat_args), in_tree, [True] * len(flat_args)
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
    debug_info = pe.debug_info(fun, in_tree, out_tree, True, 'make_jaxpr')
    with ExitStack() as stack:
      for axis_name, size in axis_env or []:
        stack.enter_context(jax.core.extend_axis_env(axis_name, size, None))
      jaxpr, out_type, consts = pe.trace_to_jaxpr_dynamic2(f, debug_info=debug_info)
    closed_jaxpr = jax.core.ClosedJaxpr(jaxpr, consts)
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
