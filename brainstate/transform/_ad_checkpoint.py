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

import functools
from typing import Callable, Tuple, Union

import jax

from brainstate._utils import set_module_as
from brainstate.typing import Missing
from ._make_jaxpr import StatefulFunction, _ensure_index_tuple

__all__ = [
    'checkpoint',
    'remat'
]


@set_module_as('brainstate.transform')
def checkpoint(
    fun: Callable = Missing(),
    *,
    prevent_cse: bool = True,
    policy: Callable[..., bool] | None = None,
    static_argnums: int | Tuple[int, ...] = (),
) -> Union[Callable, Callable[[Callable], Callable]]:
    """Make ``fun`` recompute internal linearization points when differentiated.

    This decorator wraps :func:`jax.checkpoint` (also exposed as :func:`jax.remat`) to
    rematerialize intermediate values during reverse-mode automatic differentiation.
    It allows trading additional computation for reduced peak memory when evaluating
    functions with :func:`jax.grad`, :func:`jax.vjp`, or :func:`jax.linearize`.

    Parameters
    ----------
    fun : Callable, optional
        Function whose autodiff evaluation strategy should use rematerialization.
        Positional and keyword arguments may be arrays, scalars, or arbitrarily
        nested Python containers of those types.
    prevent_cse : bool, default True
        Whether to prevent common-subexpression-elimination (CSE) optimizations in
        the generated HLO. Disabling CSE is usually necessary under
        :func:`jax.jit`/:func:`jax.pmap` so that rematerialization is not optimized
        away. Set to ``False`` when decorating code inside control-flow primitives
        (for example, :func:`jax.lax.scan`) where CSE is already handled safely.
    policy : Callable[..., bool], optional
        Callable drawn from :mod:`jax.checkpoint_policies` that decides which
        primitive outputs may be saved as residuals instead of being recomputed. The
        callable receives type-level information about a primitive application and
        returns ``True`` when the corresponding value can be cached.
    static_argnums : int or tuple of int, optional
        Indices of arguments to treat as static during tracing. Marking arguments as
        static can avoid :class:`jax.errors.ConcretizationTypeError` at the expense
        of additional retracing when those arguments change.

    Returns
    -------
    callable
        A function with the same input/output behaviour as ``fun``. When
        differentiated, it rematerializes intermediate linearization points instead
        of storing them, reducing memory pressure at the cost of extra computation.

    Notes
    -----
    Reverse-mode autodiff normally stores all linearization points during the
    forward pass so that they can be reused during the backward pass. This storage
    can dominate memory usage, particularly on accelerators where memory accesses
    are expensive. Applying ``checkpoint`` causes those values to be recomputed on
    the backward pass from the saved inputs instead of being cached.

    The decorator can be composed recursively to express sophisticated
    rematerialization strategies. For functions with data-dependent Python control
    flow, specify ``static_argnums`` (and, if needed,
    :func:`jax.ensure_compile_time_eval`) so that branching conditions are evaluated
    at trace time.

    Examples
    --------
    Use :func:`jax.checkpoint` to trade computation for memory:

    .. code-block:: python

       >>> import brainstate
       >>> import jax.numpy as jnp

       >>> @brainstate.transform.checkpoint
       ... def g(x):
       ...     y = jnp.sin(x)
       ...     z = jnp.sin(y)
       ...     return z

       >>> value, grad = jax.value_and_grad(g)(2.0)

    Compose checkpoints recursively to control the rematerialization granularity:

    .. code-block:: python

       >>> import jax

       >>> def recursive_checkpoint(funs):
       ...     if len(funs) == 1:
       ...         return funs[0]
       ...     if len(funs) == 2:
       ...         f1, f2 = funs
       ...         return lambda x: f1(f2(x))
       ...     f1 = recursive_checkpoint(funs[: len(funs) // 2])
       ...     f2 = recursive_checkpoint(funs[len(funs) // 2 :])
       ...     return lambda x: f1(jax.checkpoint(f2)(x))

    When control flow depends on argument values, mark the relevant arguments as
    static:

    .. code-block:: python

       >>> from functools import partial
       >>> import jax
       >>> import brainstate

       >>> @brainstate.transform.checkpoint(static_argnums=(1,))
       ... def foo(x, is_training):
       ...     if is_training:
       ...         ...
       ...     else:
       ...         ...

       >>> @brainstate.transform.checkpoint(static_argnums=(1,))
       ... def foo_with_eval(x, y):
       ...     with jax.ensure_compile_time_eval():
       ...         y_pos = y > 0
       ...     if y_pos:
       ...         ...
       ...     else:
       ...         ...

    As an alternative to ``static_argnums``, compute values that drive control flow
    outside the decorated function and close over them in the JAX-traced callable.
    """
    if isinstance(fun, Missing):
        return lambda f: checkpoint(f, prevent_cse=prevent_cse, policy=policy, static_argnums=static_argnums)

    static_argnums = _ensure_index_tuple(tuple() if static_argnums is None else static_argnums)
    fun = StatefulFunction(fun, static_argnums=static_argnums, name='checkpoint')
    checkpointed_fun = jax.checkpoint(
        fun.jaxpr_call,
        prevent_cse=prevent_cse,
        policy=policy,
        static_argnums=tuple(i + 1 for i in static_argnums)
    )

    @functools.wraps(fun.fun)
    def remat_fun(*args, **params):
        # compile the function and get the state trace
        state_trace = fun.get_state_trace(*args, **params, compile_if_miss=True)
        read_state_vals = state_trace.get_read_state_values(True)
        # call the checkpointed function
        write_state_vals, outs = checkpointed_fun(state_trace.get_state_values(), *args, **params)
        # write the state values back to the states
        state_trace.assign_state_vals_v2(read_state_vals, write_state_vals)
        return outs

    return remat_fun


remat = checkpoint
