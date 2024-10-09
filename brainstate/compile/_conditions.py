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

import operator
from collections.abc import Callable, Sequence
from functools import wraps, reduce

import jax
import jax.numpy as jnp
import numpy as np

from brainstate._utils import set_module_as
from ._error_if import jit_error_if
from ._make_jaxpr import StatefulFunction, _assign_state_values

__all__ = [
  'cond', 'switch', 'ifelse',
]


def _wrapped_fun(stateful_fun: StatefulFunction, states, return_states=True):
  @wraps(stateful_fun.fun)
  def wrapped_branch(state_vals, *operands):
    assert len(states) == len(state_vals)
    for st, val in zip(states, state_vals):
      st.value = val
    out = stateful_fun.jaxpr_call_auto(*operands)
    if return_states:
      return tuple(st.value for st in states), out
    return out

  return wrapped_branch


@set_module_as('brainstate.compile')
def cond(pred, true_fun: Callable, false_fun: Callable, *operands):
  """
  Conditionally apply ``true_fun`` or ``false_fun``.

  Provided arguments are correctly typed, ``cond()`` has equivalent
  semantics to this Python implementation, where ``pred`` must be a
  scalar type::

    def cond(pred, true_fun, false_fun, *operands):
      if pred:
        return true_fun(*operands)
      else:
        return false_fun(*operands)


  In contrast with :func:`jax.lax.select`, using ``cond`` indicates that only one of
  the two branches is executed (up to compiler rewrites and optimizations).
  However, when transformed with :func:`~jax.vmap` to operate over a batch of
  predicates, ``cond`` is converted to :func:`~jax.lax.select`.

  Args:
    pred: Boolean scalar type, indicating which branch function to apply.
    true_fun: Function (A -> B), to be applied if ``pred`` is True.
    false_fun: Function (A -> B), to be applied if ``pred`` is False.
    operands: Operands (A) input to either branch depending on ``pred``. The
      type can be a scalar, array, or any pytree (nested Python tuple/list/dict)
      thereof.

  Returns:
    Value (B) of either ``true_fun(*operands)`` or ``false_fun(*operands)``,
    depending on the value of ``pred``. The type can be a scalar, array, or any
    pytree (nested Python tuple/list/dict) thereof.
  """
  if not (callable(true_fun) and callable(false_fun)):
    raise TypeError("true_fun and false_fun arguments should be callable.")

  if pred is None:
    raise TypeError("cond predicate is None")
  if isinstance(pred, Sequence) or np.ndim(pred) != 0:
    raise TypeError(f"Pred must be a scalar, got {pred} of " +
                    (f"type {type(pred)}" if isinstance(pred, Sequence)
                     else f"shape {np.shape(pred)}."))

  # check pred
  try:
    pred_dtype = jax.dtypes.result_type(pred)
  except TypeError as err:
    raise TypeError("Pred type must be either boolean or number, got {}.".format(pred)) from err
  if pred_dtype.kind != 'b':
    if pred_dtype.kind in 'iuf':
      pred = pred != 0
    else:
      raise TypeError("Pred type must be either boolean or number, got {}.".format(pred_dtype))

  # not jit
  if jax.config.jax_disable_jit and isinstance(jax.core.get_aval(pred), jax.core.ConcreteArray):
    if pred:
      return true_fun(*operands)
    else:
      return false_fun(*operands)

  # evaluate jaxpr
  true_fun_wrap = StatefulFunction(true_fun).make_jaxpr(*operands)
  false_fun_wrap = StatefulFunction(false_fun).make_jaxpr(*operands)

  # wrap the functions
  all_states = tuple(set(true_fun_wrap.get_states() + false_fun_wrap.get_states()))
  true_fun = _wrapped_fun(true_fun_wrap, all_states)
  false_fun = _wrapped_fun(false_fun_wrap, all_states)

  # operands
  operands = ([st.value for st in all_states],) + operands

  # cond
  state_vals, out = jax.lax.cond(pred, true_fun, false_fun, *operands)
  _assign_state_values(all_states, state_vals)
  return out

  # ops, ops_tree = jax.tree.flatten(operands)
  # linear_ops = [False] * len(ops)
  # ops_avals = tuple(jax.util.safe_map(_abstractify, ops))
  #
  # # true and false jaxprs
  # jaxprs, consts, out_trees = _initial_style_jaxprs_with_common_consts(
  #   (true_fun, false_fun), ops_tree, ops_avals, 'cond')
  # if any(isinstance(op_aval, state.AbstractRef) for op_aval in ops_avals):
  #   raise ValueError("Cannot pass `Ref`s into `cond`.")
  # true_jaxpr, false_jaxpr = jaxprs
  # out_tree, false_out_tree = out_trees
  # if any(isinstance(out_aval, state.AbstractRef) for out_aval in true_jaxpr.out_avals + false_jaxpr.out_avals):
  #   raise ValueError("Cannot return `Ref`s from `cond`.")
  #
  # _check_tree_and_avals("true_fun and false_fun output",
  #                       out_tree, true_jaxpr.out_avals,
  #                       false_out_tree, false_jaxpr.out_avals)
  # joined_effects = jax.core.join_effects(true_jaxpr.effects, false_jaxpr.effects)
  # disallowed_effects = effects.control_flow_allowed_effects.filter_not_in(joined_effects)
  # if disallowed_effects:
  #   raise NotImplementedError(f'Effects not supported in `cond`: {disallowed_effects}')
  #
  # # replace jaxpr effects
  # index = jax.lax.convert_element_type(pred, np.int32)
  # if joined_effects:
  #   # Raise index in case of effects to allow data-dependence-based discharging
  #   # of those effects (even if they don't have an explicit data dependence).
  #   index = jax.core.raise_as_much_as_possible(index)
  # false_jaxpr = _replace_jaxpr_effects(false_jaxpr, joined_effects)
  # true_jaxpr = _replace_jaxpr_effects(true_jaxpr, joined_effects)
  #
  # # bind
  # linear = [False] * len(consts) + linear_ops
  # cond_outs = jax.lax.cond_p.bind(index, *consts, *ops, branches=(false_jaxpr, true_jaxpr), linear=tuple(linear))
  #
  # # outputs
  # st_vals, out = jax.tree.unflatten(out_tree, cond_outs)
  # for st, val in zip(all_states, st_vals):
  #   st.value = val
  # return out


@set_module_as('brainstate.compile')
def switch(index, branches: Sequence[Callable], *operands):
  """
  Apply exactly one of ``branches`` given by ``index``.

  If ``index`` is out of bounds, it is clamped to within bounds.

  Has the semantics of the following Python::

    def switch(index, branches, *operands):
      index = clamp(0, index, len(branches) - 1)
      return branches[index](*operands)

  Internally this wraps XLA's `Conditional
  <https://www.tensorflow.org/xla/operation_semantics#conditional>`_
  operator. However, when transformed with :func:`~jax.vmap` to operate over a
  batch of predicates, ``cond`` is converted to :func:`~jax.lax.select`.

  Args:
    index: Integer scalar type, indicating which branch function to apply.
    branches: Sequence of functions (A -> B) to be applied based on ``index``.
    operands: Operands (A) input to whichever branch is applied.

  Returns:
    Value (B) of ``branch(*operands)`` for the branch that was selected based
    on ``index``.
  """
  # check branches
  if not all(callable(branch) for branch in branches):
    raise TypeError("branches argument should be a sequence of callables.")

  # check index
  if len(np.shape(index)) != 0:
    raise TypeError(f"Branch index must be scalar, got {index} of shape {np.shape(index)}.")
  try:
    index_dtype = jax.dtypes.result_type(index)
  except TypeError as err:
    msg = f"Index type must be an integer, got {index}."
    raise TypeError(msg) from err
  if index_dtype.kind not in 'iu':
    raise TypeError(f"Index type must be an integer, got {index} as {index_dtype}")

  # format branches
  branches = tuple(branches)
  if len(branches) == 0:
    raise ValueError("Empty branch sequence")
  elif len(branches) == 1:
    return branches[0](*operands)

  # format index
  index = jax.lax.convert_element_type(index, np.int32)
  lo = np.array(0, np.int32)
  hi = np.array(len(branches) - 1, np.int32)
  index = jax.lax.clamp(lo, index, hi)

  # not jit
  if jax.config.jax_disable_jit and isinstance(jax.core.core.get_aval(index), jax.core.ConcreteArray):
    return branches[int(index)](*operands)

  # evaluate jaxpr
  wrapped_branches = [StatefulFunction(branch) for branch in branches]
  for wrapped_branch in wrapped_branches:
    wrapped_branch.make_jaxpr(*operands)

  # wrap the functions
  all_states = tuple(set(reduce(operator.add, [wrapped_branch.get_states() for wrapped_branch in wrapped_branches])))
  branches = tuple(_wrapped_fun(wrapped_branch, all_states) for wrapped_branch in wrapped_branches)

  # operands
  operands = ([st.value for st in all_states],) + operands

  # switch
  state_vals, out = jax.lax.switch(index, branches, *operands)
  _assign_state_values(all_states, state_vals)
  return out

  # ops, ops_tree = jax.tree.flatten(operands)
  # ops_avals = tuple(jax.util.safe_map(_abstractify, ops))
  #
  # # true jaxprs
  # jaxprs, consts, out_trees = _initial_style_jaxprs_with_common_consts(
  #   branches, ops_tree, ops_avals, primitive_name='switch')
  # for i, (out_tree, jaxpr) in enumerate(zip(out_trees[1:], jaxprs[1:])):
  #   _check_tree_and_avals(f"branch 0 and {i + 1} outputs",
  #                         out_trees[0], jaxprs[0].out_avals,
  #                         out_tree, jaxpr.out_avals)
  # joined_effects = jax.core.join_effects(*(jaxpr.effects for jaxpr in jaxprs))
  # disallowed_effects = effects.control_flow_allowed_effects.filter_not_in(joined_effects)
  # if disallowed_effects:
  #   raise NotImplementedError(f'Effects not supported in `switch`: {disallowed_effects}')
  # if joined_effects:
  #   # Raise index in case of effects to allow data-dependence-based discharging
  #   # of those effects (even if they don't have an explicit data dependence).
  #   index = jax.core.raise_as_much_as_possible(index)
  #
  # # bind
  # linear = (False,) * (len(consts) + len(ops))
  # cond_outs = jax.lax.cond_p.bind(index, *consts, *ops, branches=tuple(jaxprs), linear=linear)
  #
  # # outputs
  # st_vals, out = jax.tree.unflatten(out_trees[0], cond_outs)
  # for st, val in zip(all_states, st_vals):
  #   st.value = val
  # return out


@set_module_as('brainstate.compile')
def ifelse(conditions, branches, *operands, check_cond: bool = True):
  """
  ``If-else`` control flows looks like native Pythonic programming.

  Examples
  --------

  >>> import brainstate as bst
  >>> def f(a):
  >>>    return bst.compile.ifelse(conditions=[a > 10, a > 5, a > 2, a > 0],
  >>>                               branches=[lambda: 1,
  >>>                                         lambda: 2,
  >>>                                         lambda: 3,
  >>>                                         lambda: 4,
  >>>                                         lambda: 5])
  >>> f(1)
  4
  >>> f(0)
  5

  Parameters
  ----------
  conditions: bool, sequence of bool, Array
    The boolean conditions.
  branches: Any
    The branches, at least has two elements. Elements can be functions,
    arrays, or numbers. The number of ``branches`` and ``conditions`` has
    the relationship of `len(branches) == len(conditions) + 1`.
    Each branch should receive one arguement for ``operands``.
  *operands: optional, Any
    The operands for each branch.
  check_cond: bool
    Whether to check the conditions. Default is True.

  Returns
  -------
  res: Any
    The results of the control flow.
  """
  # check branches
  if not all(callable(branch) for branch in branches):
    raise TypeError("branches argument should be a sequence of callables.")

  # format branches
  branches = tuple(branches)
  if len(branches) == 0:
    raise ValueError("Empty branch sequence")
  elif len(branches) == 1:
    return branches[0](*operands)
  if len(conditions) != len(branches):
    raise ValueError("The number of conditions should be equal to the number of branches.")

  # format index
  conditions = jnp.asarray(conditions, np.int32)
  if check_cond:
    jit_error_if(jnp.sum(conditions) != 1, "Only one condition can be True. But got {}.", err_arg=conditions)
  index = jnp.where(conditions, size=1, fill_value=len(conditions) - 1)[0][0]
  return switch(index, branches, *operands)
