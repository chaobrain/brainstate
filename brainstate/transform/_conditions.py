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

from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from brainstate._compatible_import import to_concrete_aval, Tracer
from brainstate._utils import set_module_as
from ._error_if import jit_error_if
from ._make_jaxpr import StatefulFunction
from ._util import wrap_single_fun_in_multi_branches

__all__ = [
    'cond', 'switch', 'ifelse',
]


@set_module_as('brainstate.transform')
def cond(pred, true_fun: Callable, false_fun: Callable, *operands):
    """
    Conditionally apply ``true_fun`` or ``false_fun``.

    Parameters
    ----------
    pred : bool or array-like
        Boolean scalar selecting which branch to execute. Numeric inputs are
        treated as ``True`` when non-zero.
    true_fun : Callable
        Function that receives ``*operands`` when ``pred`` is ``True``.
    false_fun : Callable
        Function that receives ``*operands`` when ``pred`` is ``False``.
    *operands : Any
        Operands forwarded to either branch. May be any pytree of arrays,
        scalars, or nested containers thereof.

    Returns
    -------
    Any
        Value returned by the selected branch with the same pytree structure
        as produced by ``true_fun`` or ``false_fun``.

    Notes
    -----
    Provided the arguments are correctly typed, :func:`cond` has semantics
    that match the following Python implementation, where ``pred`` must be a
    scalar:

    .. code-block:: python

        >>> def cond(pred, true_fun, false_fun, *operands):
        ...     if pred:
        ...         return true_fun(*operands)
        ...     return false_fun(*operands)

    In contrast with :func:`jax.lax.select`, using :func:`cond` indicates that only
    one branch runs (subject to compiler rewrites and optimizations). When
    transformed with :func:`~jax.vmap` over a batch of predicates, :func:`cond` is
    converted to :func:`~jax.lax.select`.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>>
        >>> def branch_true(x):
        ...     return x + 1
        >>>
        >>> def branch_false(x):
        ...     return x - 1
        >>>
        >>> brainstate.transform.cond(True, branch_true, branch_false, 3)
    """
    if not (callable(true_fun) and callable(false_fun)):
        raise TypeError("true_fun and false_fun arguments should be callable.")

    if pred is None:
        raise TypeError("cond predicate is None")
    if isinstance(pred, Sequence) or np.ndim(pred) != 0:
        raise TypeError(f"Pred must be a scalar, got {pred} of " +
                        (f"type {type(pred)}" if isinstance(pred, Sequence) else f"shape {np.shape(pred)}."))

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
    state_trace = (stateful_true.get_state_trace(*operands) +
                   stateful_false.get_state_trace(*operands))
    read_state_vals = state_trace.get_read_state_values(True)
    write_state_vals = state_trace.get_write_state_values(True)

    # wrap the functions
    true_fun = wrap_single_fun_in_multi_branches(
        stateful_true, state_trace, read_state_vals, True, stateful_true.get_arg_cache_key(*operands)
    )
    false_fun = wrap_single_fun_in_multi_branches(
        stateful_false, state_trace, read_state_vals, True, stateful_false.get_arg_cache_key(*operands)
    )

    # cond
    write_state_vals, out = jax.lax.cond(pred, true_fun, false_fun, write_state_vals, *operands)

    # assign the written state values and restore the read state values
    state_trace.assign_state_vals_v2(read_state_vals, write_state_vals)
    return out


@set_module_as('brainstate.transform')
def switch(index, branches: Sequence[Callable], *operands):
    """
    Apply exactly one branch from ``branches`` based on ``index``.

    Parameters
    ----------
    index : int or array-like
        Scalar integer specifying which branch to execute.
    branches : Sequence[Callable]
        Sequence of callables; each receives ``*operands``.
    *operands : Any
        Operands forwarded to the selected branch. May be any pytree of arrays,
        scalars, or nested containers thereof.

    Returns
    -------
    Any
        Value returned by the selected branch with the same pytree structure
        as the selected callable.

    Notes
    -----
    If ``index`` is out of bounds, it is clamped to ``[0, len(branches) - 1]``.
    Conceptually, :func:`switch` behaves like:

    .. code-block:: python

        >>> def switch(index, branches, *operands):
        ...     safe_index = clamp(0, index, len(branches) - 1)
        ...     return branches[safe_index](*operands)

    Internally this wraps XLA's `Conditional <https://www.tensorflow.org/xla/operation_semantics#conditional>`_
    operator. When transformed with :func:`~jax.vmap` over a batch of predicates,
    :func:`switch` is converted to :func:`~jax.lax.select`.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>>
        >>> branches = (
        ...     lambda x: x - 1,
        ...     lambda x: x,
        ...     lambda x: x + 1,
        ... )
        >>>
        >>> brainstate.transform.switch(2, branches, 3)
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
    if jax.config.jax_disable_jit and not isinstance(to_concrete_aval(index), Tracer):
        return branches[int(index)](*operands)

    # evaluate jaxpr
    wrapped_branches = [StatefulFunction(branch, name='switch').make_jaxpr(*operands) for branch in branches]

    # wrap the functions
    state_trace = (wrapped_branches[0].get_state_trace(*operands) +
                   wrapped_branches[1].get_state_trace(*operands))
    state_trace.merge(*[wrapped_branch.get_state_trace(*operands)
                        for wrapped_branch in wrapped_branches[2:]])
    read_state_vals = state_trace.get_read_state_values(True)
    write_state_vals = state_trace.get_write_state_values(True)
    branches = [
        wrap_single_fun_in_multi_branches(
            wrapped_branch, state_trace, read_state_vals, True, wrapped_branch.get_arg_cache_key(*operands)
        )
        for wrapped_branch in wrapped_branches
    ]

    # switch
    write_state_vals, out = jax.lax.switch(index, branches, write_state_vals, *operands)

    # write back state values or restore them
    state_trace.assign_state_vals_v2(read_state_vals, write_state_vals)
    return out


@set_module_as('brainstate.transform')
def ifelse(conditions, branches, *operands, check_cond: bool = True):
    """
    Represent multi-way ``if``/``elif``/``else`` control flow.

    Parameters
    ----------
    conditions : Sequence[bool] or Array
        Sequence of mutually exclusive boolean predicates. When ``check_cond`` is
        ``True``, exactly one entry must evaluate to ``True``.
    branches : Sequence[Callable]
        Sequence of branch callables evaluated lazily. Must have the same length as
        ``conditions``, contain at least two callables, and each branch receives
        ``*operands`` when selected.
    *operands : Any
        Operands forwarded to the selected branch as positional arguments.
    check_cond : bool, default=True
        Whether to verify that exactly one condition evaluates to ``True``.

    Returns
    -------
    Any
        Value produced by the branch corresponding to the active condition.

    Notes
    -----
    When ``check_cond`` is ``True``, exactly one condition must evaluate to ``True``.
    A common pattern is to make the final condition ``True`` to encode a default
    branch.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>>
        >>> def describe(a):
        ...     return brainstate.transform.ifelse(
        ...         conditions=[a > 5, a > 0, True],
        ...         branches=[
        ...             lambda: "greater than five",
        ...             lambda: "positive",
        ...             lambda: "non-positive",
        ...         ],
        ...     )
        >>>
        >>> describe(7)
        >>> describe(-1)
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
        jit_error_if(jnp.sum(conditions) != 1, "Only one condition can be True. But got {c}.", c=conditions)
    index = jnp.where(conditions, size=1, fill_value=len(conditions) - 1)[0][0]
    return switch(index, branches, *operands)
