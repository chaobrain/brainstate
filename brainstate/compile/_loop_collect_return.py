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

from __future__ import annotations

import math
from functools import wraps
from typing import Callable, Optional, TypeVar, Tuple, Any

import jax
import jax.numpy as jnp

from brainstate._utils import set_module_as
from brainstate.augment._unvmap import unvmap
from ._make_jaxpr import StatefulFunction, _assign_state_values
from ._progress_bar import ProgressBar

X = TypeVar('X')
Y = TypeVar('Y')
T = TypeVar('T')
Carry = TypeVar('Carry')

__all__ = [
  # for loop & scan
  'scan', 'checkpointed_scan',
  'for_loop', 'checkpointed_for_loop',
]


def _wrap_fun_with_pbar(fun, pbar_runner):
  @wraps(fun)
  def new_fun(new_carry, inputs):
    i, old_carry = new_carry
    old_carry, old_outputs = fun(old_carry, inputs)
    pbar_runner(unvmap(i, op='none'))
    return (i + 1, old_carry), old_outputs

  return new_fun


def _wrapped_scan_fun(stateful_fun: StatefulFunction, states):
  @wraps(stateful_fun.fun)
  def wrapped_fun(new_carry, inputs):
    state_vals, carry = new_carry
    assert len(states) == len(state_vals)
    for st, val in zip(states, state_vals):
      st.value = val
    carry, out = stateful_fun.jaxpr_call_auto(carry, inputs)
    return (tuple(st.value for st in states), carry), out

  return wrapped_fun


@set_module_as('brainstate.compile')
def scan(
    f: Callable[[Carry, X], Tuple[Carry, Y]],
    init: Carry,
    xs: X,
    length: int | None = None,
    reverse: bool = False,
    unroll: int | bool = 1,
    pbar: ProgressBar | None = None,
) -> Tuple[Carry, Y]:
  """
  Scan a function over leading array axes while carrying along state.

  The `Haskell-like type signature`_ in brief is

  .. code-block:: haskell

    scan :: (c -> a -> (c, b)) -> c -> [a] -> (c, [b])

  where for any array type specifier ``t``, ``[t]`` represents the type with an additional
  leading axis, and if ``t`` is a pytree (container) type with array leaves then ``[t]``
  represents the type with the same pytree structure and corresponding leaves
  each with an additional leading axis.

  When the type of ``xs`` (denoted `a` above) is an array type or None, and the type
  of ``ys`` (denoted `b` above) is an array type, the semantics of :func:`~scan` are
  given roughly by this Python implementation::

    def scan(f, init, xs, length=None):
      if xs is None:
        xs = [None] * length
      carry = init
      ys = []
      for x in xs:
        carry, y = f(carry, x)
        ys.append(y)
      return carry, np.stack(ys)

  Unlike that Python version, both ``xs`` and ``ys`` may be arbitrary pytree
  values, and so multiple arrays can be scanned over at once and produce multiple
  output arrays. ``None`` is actually a special case of this, as it represents an
  empty pytree.

  Also unlike that Python version, :func:`~scan` is a JAX primitive and is
  lowered to a single WhileOp. That makes it useful for reducing
  compilation times for JIT-compiled functions, since native Python
  loop constructs in an :func:`~jax.jit` function are unrolled, leading to large
  XLA computations.

  Finally, the loop-carried value ``carry`` must hold a fixed shape and dtype
  across all iterations (and not just be consistent up to NumPy rank/shape
  broadcasting and dtype promotion rules, for example). In other words, the type
  ``c`` in the type signature above represents an array with a fixed shape and
  dtype (or a nested tuple/list/dict container data structure with a fixed
  structure and arrays with fixed shape and dtype at the leaves).

  Args:
    f: a Python function to be scanned of type ``c -> a -> (c, b)``, meaning
      that ``f`` accepts two arguments where the first is a value of the loop
      carry and the second is a slice of ``xs`` along its leading axis, and that
      ``f`` returns a pair where the first element represents a new value for
      the loop carry and the second represents a slice of the output.
    init: an initial loop carry value of type ``c``, which can be a scalar,
      array, or any pytree (nested Python tuple/list/dict) thereof, representing
      the initial loop carry value. This value must have the same structure as
      the first element of the pair returned by ``f``.
    xs: the value of type ``[a]`` over which to scan along the leading axis,
      where ``[a]`` can be an array or any pytree (nested Python
      tuple/list/dict) thereof with consistent leading axis sizes.
    length: optional integer specifying the number of loop iterations, which
      must agree with the sizes of leading axes of the arrays in ``xs`` (but can
      be used to perform scans where no input ``xs`` are needed).
    reverse: optional boolean specifying whether to run the scan iteration
      forward (the default) or in reverse, equivalent to reversing the leading
      axes of the arrays in both ``xs`` and in ``ys``.
    unroll: optional positive int or bool specifying, in the underlying
      operation of the scan primitive, how many scan iterations to unroll within
      a single iteration of a loop. If an integer is provided, it determines how
      many unrolled loop iterations to run within a single rolled iteration of
      the loop. If a boolean is provided, it will determine if the loop is
      completely unrolled (i.e. `unroll=True`) or left completely unrolled (i.e.
      `unroll=False`).
    pbar: optional :class:`~.ProgressBar` instance to display the progress
      of the scan operation.

  Returns:
    A pair of type ``(c, [b])`` where the first element represents the final
    loop carry value and the second element represents the stacked outputs of
    the second output of ``f`` when scanned over the leading axis of the inputs.

  .. _Haskell-like type signature: https://wiki.haskell.org/Type_signature
  """
  # check "f"
  if not callable(f):
    raise TypeError("f argument should be a callable.")

  # check "xs"
  xs_flat, xs_tree = jax.tree.flatten(xs)
  try:
    lengths = [x.shape[0] for x in xs_flat]
  except AttributeError as err:
    raise ValueError("scan got value with no leading axis to scan over: "
                     "{}.".format(', '.join(str(x) for x in xs_flat if not hasattr(x, 'shape')))) from err
  if length is not None:
    length = int(length)
    if not all(length == l for l in lengths):
      raise ValueError(("scan got `length` argument of {} which disagrees with "
                        "leading axis sizes {}.").format(length, [x.shape[0] for x in xs_flat]))
  else:
    unique_lengths = set(lengths)
    if len(unique_lengths) > 1:
      msg = "scan got values with different leading axis sizes: {}."
      raise ValueError(msg.format(', '.join(str(x.shape[0]) for x in xs_flat)))
    elif len(unique_lengths) == 0:
      raise ValueError("scan got no values to scan over and `length` not provided.")
    else:
      length, = unique_lengths

  # function with progress bar
  has_pbar = False
  if pbar is not None:
    has_pbar = True
    f = _wrap_fun_with_pbar(f, pbar.init(length))
    init = (0, init) if pbar else init

  # not jit
  if jax.config.jax_disable_jit:
    if length == 0:
      raise ValueError("zero-length scan is not supported in disable_jit() mode because the output type is unknown.")
    carry = init
    ys = []
    maybe_reversed = reversed if reverse else lambda x: x
    for i in maybe_reversed(range(length)):
      xs_slice = [jax.lax.index_in_dim(x, i, keepdims=False) for x in xs_flat]
      carry, y = f(carry, jax.tree.unflatten(xs_tree, xs_slice))
      ys.append(y)
    stacked_y = jax.tree.map(lambda *elems: jnp.stack(elems), *maybe_reversed(ys))
    if has_pbar:
      return carry[1], stacked_y
    else:
      return carry, stacked_y

  # evaluate jaxpr, get all states #
  # ------------------------------ #
  xs_avals = [jax.core.raise_to_shaped(jax.core.get_aval(x)) for x in xs_flat]
  x_avals = [jax.core.mapped_aval(length, 0, aval) for aval in xs_avals]
  stateful_fun = StatefulFunction(f).make_jaxpr(init, xs_tree.unflatten(x_avals))
  all_states = stateful_fun.get_states()
  wrapped_f = _wrapped_scan_fun(stateful_fun, all_states)

  # scan
  init = (tuple(st.value for st in all_states), init)
  (state_vals, carry), ys = jax.lax.scan(wrapped_f, init, xs, length=length, reverse=reverse, unroll=unroll)
  _assign_state_values(all_states, state_vals)
  if has_pbar:
    carry = carry[1]
  return carry, ys


def checkpointed_scan(
    f: Callable[[Carry, X], Tuple[Carry, Y]],
    init: Carry,
    xs: X,
    length: Optional[int] = None,
    base: int = 16,
    pbar: Optional[ProgressBar] = None,
):
  """
  Scan a function over leading array axes while carrying along state.
  This function is similar to :func:`~scan` but with a checkpointed version.

  Args:
    f: a Python function to be scanned of type ``c -> a -> (c, b)``, meaning
      that ``f`` accepts two arguments where the first is a value of the loop
      carry and the second is a slice of ``xs`` along its leading axis, and that
      ``f`` returns a pair where the first element represents a new value for
      the loop carry and the second represents a slice of the output.
    init: an initial loop carry value of type ``c``, which can be a scalar,
      array, or any pytree (nested Python tuple/list/dict) thereof, representing
      the initial loop carry value. This value must have the same structure as
      the first element of the pair returned by ``f``.
    xs: the value of type ``[a]`` over which to scan along the leading axis,
      where ``[a]`` can be an array or any pytree (nested Python
      tuple/list/dict) thereof with consistent leading axis sizes.
    length: optional integer specifying the number of loop iterations, which
      must agree with the sizes of leading axes of the arrays in ``xs`` (but can
      be used to perform scans where no input ``xs`` are needed).
    base: optional integer specifying the base for the bounded scan loop.
    pbar: optional :class:`~.ProgressBar` instance to display the progress
      of the scan operation.

  Returns:
    A pair of type ``(c, [b])`` where the first element represents the final
    loop carry value and the second element represents the stacked outputs of
    the second output of ``f`` when scanned over the leading axis of the inputs.
  """
  # check "f"
  if not callable(f):
    raise TypeError("f argument should be a callable.")

  # check "xs"
  xs_flat, xs_tree = jax.tree.flatten(xs)
  try:
    lengths = [x.shape[0] for x in xs_flat]
  except AttributeError as err:
    raise ValueError("scan got value with no leading axis to scan over: "
                     "{}.".format(', '.join(str(x) for x in xs_flat if not hasattr(x, 'shape')))) from err
  if length is not None:
    length = int(length)
    if not all(length == l for l in lengths):
      raise ValueError(("scan got `length` argument of {} which disagrees with "
                        "leading axis sizes {}.").format(length, [x.shape[0] for x in xs_flat]))
  else:
    unique_lengths = set(lengths)
    if len(unique_lengths) > 1:
      msg = "scan got values with different leading axis sizes: {}."
      raise ValueError(msg.format(', '.join(str(x.shape[0]) for x in xs_flat)))
    elif len(unique_lengths) == 0:
      raise ValueError("scan got no values to scan over and `length` not provided.")
    else:
      length, = unique_lengths

  # function with progress bar
  if pbar is not None:
    pbar_runner = pbar.init(length)
  else:
    pbar_runner = None

  # evaluate jaxpr
  xs_avals = [jax.core.raise_to_shaped(jax.core.get_aval(x)) for x in xs_flat]
  x_avals = [jax.core.mapped_aval(length, 0, aval) for aval in xs_avals]
  stateful_fun = StatefulFunction(f).make_jaxpr(init, xs_tree.unflatten(x_avals))
  all_states = stateful_fun.get_states()
  out_info = stateful_fun.get_out_shapes()[0]

  # initialize the collected values/dataa
  assert len(out_info) == 2, "function in checkpointed_scan should return two data: carray and out."
  data2collection = jax.tree.map(lambda x: jnp.zeros((length,) + x.shape, x.dtype), out_info[1])
  del out_info

  def wrapped_cond_fun(inp):
    return inp[-1] < length

  def wrapped_body_fun(inp):
    (prev_states, carray), prev_collect, i = inp
    # progress bar
    if pbar_runner is not None:
      pbar_runner(unvmap(i, op='none'))
    # call the function
    new_states, (new_carray, out4updates) = stateful_fun.jaxpr_call(
      prev_states, carray, jax.tree.map(lambda x: x[i], xs))
    # out of bounds
    pred = i < length
    new_collect = jax.tree.map(
      lambda x, update: x.at[i].set(jax.lax.select(pred, update, x[i])),
      prev_collect,
      out4updates,
    )
    new_states = jax.tree.map(
      lambda ps, ns: jax.lax.select(pred, ns, ps),
      prev_states,
      new_states,
    )
    new_carray = jax.tree.map(
      lambda pc, nc: jax.lax.select(pred, nc, pc),
      carray,
      new_carray,
    )
    return (new_states, new_carray), new_collect, i + 1

  # while_loop
  rounded_max_steps = base ** int(math.ceil(math.log(length, base)))
  (state_vals, carry), data2collection, _ = _bounded_while_loop(
    wrapped_cond_fun,
    wrapped_body_fun,
    ((tuple(st.value for st in all_states), init), data2collection, 0),
    rounded_max_steps,
    base,
    pbar_runner
  )
  _assign_state_values(all_states, state_vals)
  del state_vals, all_states, stateful_fun
  return carry, data2collection


def _forloop_to_scan_fun(f: Callable):
  @wraps(f)
  def scan_fun(carry, x):
    return carry, f(*x)

  return scan_fun


@set_module_as('brainstate.compile')
def for_loop(
    f: Callable[[X], Y],
    *xs,
    length: Optional[int] = None,
    reverse: bool = False,
    unroll: int | bool = 1,
    pbar: Optional[ProgressBar] = None
):
  """
  ``for-loop`` control flow with :py:class:`~.State`.

  Args:
    f: a Python function to be scanned of type ``c -> a -> (c, b)``, meaning
      that ``f`` accepts two arguments where the first is a value of the loop
      carry and the second is a slice of ``xs`` along its leading axis, and that
      ``f`` returns a pair where the first element represents a new value for
      the loop carry and the second represents a slice of the output.
    xs: the value of type ``[a]`` over which to scan along the leading axis,
      where ``[a]`` can be an array or any pytree (nested Python
      tuple/list/dict) thereof with consistent leading axis sizes.
    length: optional integer specifying the number of loop iterations, which
      must agree with the sizes of leading axes of the arrays in ``xs`` (but can
      be used to perform scans where no input ``xs`` are needed).
    reverse: optional boolean specifying whether to run the scan iteration
      forward (the default) or in reverse, equivalent to reversing the leading
      axes of the arrays in both ``xs`` and in ``ys``.
    unroll: optional positive int or bool specifying, in the underlying
      operation of the scan primitive, how many scan iterations to unroll within
      a single iteration of a loop. If an integer is provided, it determines how
      many unrolled loop iterations to run within a single rolled iteration of
      the loop. If a boolean is provided, it will determine if the loop is
      completely unrolled (i.e. `unroll=True`) or left completely unrolled (i.e.
      `unroll=False`).
    pbar: optional :class:`~.ProgressBar` instance to display the progress
      of the scan operation.

  Returns:
    The return represents the stacked outputs of the second output of ``f`` 
    when scanned over the leading axis of the inputs.

  """
  _, ys = scan(
    _forloop_to_scan_fun(f),
    init=None,
    xs=xs,
    length=length,
    reverse=reverse,
    unroll=unroll,
    pbar=pbar
  )
  return ys


def checkpointed_for_loop(
    f: Callable[[X], Y],
    *xs: X,
    length: Optional[int] = None,
    base: int = 16,
    pbar: Optional[ProgressBar] = None,
):
  """
  ``for-loop`` control flow with :py:class:`~.State` with a checkpointed version, similar to :py:func:`for_loop`.

  Args:
    f: a Python function to be scanned of type ``c -> a -> (c, b)``, meaning
      that ``f`` accepts two arguments where the first is a value of the loop
      carry and the second is a slice of ``xs`` along its leading axis, and that
      ``f`` returns a pair where the first element represents a new value for
      the loop carry and the second represents a slice of the output.
    xs: the value of type ``[a]`` over which to scan along the leading axis,
      where ``[a]`` can be an array or any pytree (nested Python
      tuple/list/dict) thereof with consistent leading axis sizes.
    length: optional integer specifying the number of loop iterations, which
      must agree with the sizes of leading axes of the arrays in ``xs`` (but can
      be used to perform scans where no input ``xs`` are needed).
    base: optional integer specifying the base for the bounded scan loop.
    pbar: optional :class:`~.ProgressBar` instance to display the progress
      of the scan operation.

  Returns:
    The return represents the stacked outputs of the second output of ``f``
    when scanned over the leading axis of the inputs.
  """
  _, ys = checkpointed_scan(
    _forloop_to_scan_fun(f),
    init=None,
    xs=xs,
    length=length,
    base=base,
    pbar=pbar
  )
  return ys


# There's several tricks happening here to work around various limitations of JAX.
# (Also see https://github.com/google/jax/issues/2139#issuecomment-1039293633)
# 1. `unvmap_any` prior to using `lax.cond`. JAX has a problem in that vmap-of-cond
#    is converted to a `lax.select`, which executes both branches unconditionally.
#    Thus writing this naively, using a plain `lax.cond`, will mean the loop always
#    runs to `max_steps` when executing under vmap. Instead we run (only) until every
#    batch element has finished.
# 2. Treating in-place updates specially in the body_fun. Specifically we need to
#    `lax.select` the update-to-make, not the updated buffer. This is because the
#    latter instead results in XLA:CPU failing to determine that the buffer can be
#    updated in-place, and instead it makes a copy. c.f. JAX issue #8192.
#    This is done through the extra `inplace` argument provided to `body_fun`.
# 3. The use of the `@jax.checkpoint` decorator. Backpropagation through a
#    `bounded_while_loop` will otherwise run in θ(max_steps) time, rather than
#    θ(number of steps actually taken).
# 4. The use of `base`. In theory `base=2` is optimal at run time, as it implies the
#    fewest superfluous operations. In practice this implies quite deep recursion in
#    the construction of the bounded while loop, and this slows down the jaxpr
#    creation and the XLA compilation. We choose `base=16` as a reasonable-looking
#    compromise between compilation time and run time.

def _bounded_while_loop(
    cond_fun: Callable,
    body_fun: Callable,
    val: Any,
    max_steps: int,
    base: int,
    pbar_runner: Optional[Callable] = None
):
  if max_steps == 1:
    return body_fun(val)
  else:

    def true_call(val_):
      return _bounded_while_loop(cond_fun, body_fun, val_, max_steps // base, base, pbar_runner)

    def false_call(val_):
      if pbar_runner is not None:
        pbar_runner(unvmap(val_[-1] + max_steps, op='none'))
      return val_[:-1] + (val_[-1] + max_steps,)

    def scan_fn(val_, _):
      return jax.lax.cond(unvmap(cond_fun(val_), op='any'), true_call, false_call, val_), None

    # Don't put checkpointing on the lowest level
    if max_steps != base:
      scan_fn = jax.checkpoint(scan_fn, prevent_cse=False)  # pyright: ignore

    return jax.lax.scan(scan_fn, val, xs=None, length=base)[0]
