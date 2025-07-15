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

from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from brainstate._compatible_import import to_concrete_aval, Tracer
from brainstate._utils import set_module_as
from ._error_if import jit_error_if
from ._make_jaxpr import StatefulFunction
from ._util import wrap_single_fun_in_multi_branches, write_back_state_values

__all__ = [
    'cond', 'switch', 'ifelse',
]


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
    if jax.config.jax_disable_jit and not isinstance(to_concrete_aval(pred), Tracer):
        if pred:
            return true_fun(*operands)
        else:
            return false_fun(*operands)

    # evaluate jaxpr
    stateful_true = StatefulFunction(true_fun, name='cond:true').make_jaxpr(*operands)
    stateful_false = StatefulFunction(false_fun, name='conda:false').make_jaxpr(*operands)

    # state trace and state values
    state_trace = stateful_true.get_state_trace() + stateful_false.get_state_trace()
    read_state_vals = state_trace.get_read_state_values(True)
    write_state_vals = state_trace.get_write_state_values(True)

    # wrap the functions
    true_fun = wrap_single_fun_in_multi_branches(stateful_true, state_trace, read_state_vals, True)
    false_fun = wrap_single_fun_in_multi_branches(stateful_false, state_trace, read_state_vals, True)

    # cond
    write_state_vals, out = jax.lax.cond(pred, true_fun, false_fun, write_state_vals, *operands)

    # assign the written state values and restore the read state values
    write_back_state_values(state_trace, read_state_vals, write_state_vals)
    return out


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
    wrapped_branches = [StatefulFunction(branch, name='switch') for branch in branches]
    for wrapped_branch in wrapped_branches:
        wrapped_branch.make_jaxpr(*operands)

    # wrap the functions
    state_trace = wrapped_branches[0].get_state_trace() + wrapped_branches[1].get_state_trace()
    state_trace.merge(*[wrapped_branch.get_state_trace() for wrapped_branch in wrapped_branches[2:]])
    read_state_vals = state_trace.get_read_state_values(True)
    write_state_vals = state_trace.get_write_state_values(True)
    branches = [
        wrap_single_fun_in_multi_branches(wrapped_branch, state_trace, read_state_vals, True)
        for wrapped_branch in wrapped_branches
    ]

    # switch
    write_state_vals, out = jax.lax.switch(index, branches, write_state_vals, *operands)

    # write back state values or restore them
    write_back_state_values(state_trace, read_state_vals, write_state_vals)
    return out


@set_module_as('brainstate.compile')
def ifelse(conditions, branches, *operands, check_cond: bool = True):
    """
    ``If-else`` control flows looks like native Pythonic programming.

    Examples
    --------

    >>> import brainstate
    >>> def f(a):
    >>>    return brainstate.compile.ifelse(conditions=[a > 10, a > 5, a > 2, a > 0],
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
