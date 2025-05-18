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

import functools
from typing import Callable, Tuple, Union

import jax

from brainstate.typing import Missing
from ._make_jaxpr import StatefulFunction, _ensure_index_tuple
from ._util import write_back_state_values

__all__ = [
    'checkpoint',
    'remat'
]


def checkpoint(
    fun: Callable = Missing(),
    *,
    prevent_cse: bool = True,
    policy: Callable[..., bool] | None = None,
    static_argnums: int | Tuple[int, ...] = (),
) -> Union[Callable, Callable[[Callable], Callable]]:
    """Make ``fun`` recompute internal linearization points when differentiated.

    The :func:`jax.checkpoint` decorator, aliased to :func:`jax.remat`, provides a
    way to trade off computation time and memory cost in the context of automatic
    differentiation, especially with reverse-mode autodiff like :func:`jax.grad`
    and :func:`jax.vjp` but also with :func:`jax.linearize`.

    When differentiating a function in reverse-mode, by default all the
    linearization points (e.g. inputs to elementwise nonlinear primitive
    operations) are stored when evaluating the forward pass so that they can be
    reused on the backward pass. This evaluation strategy can lead to a high
    memory cost, or even to poor performance on hardware accelerators where memory
    access is much more expensive than FLOPs.

    An alternative evaluation strategy is for some of the linearization points to
    be recomputed (i.e. rematerialized) rather than stored. This approach can
    reduce memory usage at the cost of increased computation.

    This function decorator produces a new version of ``fun`` which follows
    the rematerialization strategy rather than the default store-everything
    strategy. That is, it returns a new version of ``fun`` which, when
    differentiated, doesn't store any of its intermediate linearization points.
    Instead, these linearization points are recomputed from the function's saved
    inputs.

    See the examples below.

    Args:
      fun: Function for which the autodiff evaluation strategy is to be changed
        from the default of storing all intermediate linearization points to
        recomputing them. Its arguments and return value should be arrays,
        scalars, or (nested) standard Python containers (tuple/list/dict) thereof.
      prevent_cse: Optional, boolean keyword-only argument indicating whether to
        prevent common subexpression elimination (CSE) optimizations in the HLO
        generated from differentiation. This CSE prevention has costs because it
        can foil other optimizations, and because it can incur high overheads on
        some backends, especially GPU. The default is True because otherwise,
        under a :func:`~jax.jit` or :func:`~jax.pmap`, CSE can defeat the purpose
        of this decorator.
        But in some settings, like when used inside a :func:`~jax.lax.scan`, this
        CSE prevention mechanism is unnecessary, in which case ``prevent_cse`` can
        be set to False.
      static_argnums: Optional, int or sequence of ints, a keyword-only argument
        indicating which argument values on which to specialize for tracing and
        caching purposes. Specifying arguments as static can avoid
        ConcretizationTypeErrors when tracing, but at the cost of more retracing
        overheads. See the example below.
      policy: Optional, callable keyword-only argument. It should be one of the
        attributes of ``jax.checkpoint_policies``. The callable takes as input a
        type-level specification of a first-order primitive application and
        returns a boolean indicating whether the corresponding output value(s) can
        be saved as residuals (or instead must be recomputed in the (co)tangent
        computation if needed).

    Returns:
      A function (callable) with the same input/output behavior as ``fun`` but
      which, when differentiated using e.g. :func:`jax.grad`, :func:`jax.vjp`, or
      :func:`jax.linearize`, recomputes rather than stores intermediate
      linearization points, thus potentially saving memory at the cost of extra
      computation.

    Here is a simple example:

    >>> import jax
    >>> import jax.numpy as jnp

    >>> @jax.checkpoint
    ... def g(x):
    ...   y = jnp.sin(x)
    ...   z = jnp.sin(y)
    ...   return z
    ...
    >>> jax.value_and_grad(g)(2.0)
    (Array(0.78907233, dtype=float32, weak_type=True), Array(-0.2556391, dtype=float32, weak_type=True))

    Here, the same value is produced whether or not the :func:`jax.checkpoint`
    decorator is present. When the decorator is not present, the values
    ``jnp.cos(2.0)`` and ``jnp.cos(jnp.sin(2.0))`` are computed on the forward
    pass and are stored for use in the backward pass, because they are needed
    on the backward pass and depend only on the primal inputs. When using
    :func:`jax.checkpoint`, the forward pass will compute only the primal outputs
    and only the primal inputs (``2.0``) will be stored for the backward pass.
    At that time, the value ``jnp.sin(2.0)`` is recomputed, along with the values
    ``jnp.cos(2.0)`` and ``jnp.cos(jnp.sin(2.0))``.

    While :func:`jax.checkpoint` controls what values are stored from the
    forward-pass to be used on the backward pass, the total amount of memory
    required to evaluate a function or its VJP depends on many additional internal
    details of that function. Those details include which numerical primitives are
    used, how they're composed, where jit and control flow primitives like scan
    are used, and other factors.

    The :func:`jax.checkpoint` decorator can be applied recursively to express
    sophisticated autodiff rematerialization strategies. For example:

    >>> def recursive_checkpoint(funs):
    ...   if len(funs) == 1:
    ...     return funs[0]
    ...   elif len(funs) == 2:
    ...     f1, f2 = funs
    ...     return lambda x: f1(f2(x))
    ...   else:
    ...     f1 = recursive_checkpoint(funs[:len(funs)//2])
    ...     f2 = recursive_checkpoint(funs[len(funs)//2:])
    ...     return lambda x: f1(jax.checkpoint(f2)(x))
    ...

    If ``fun`` involves Python control flow that depends on argument values,
    it may be necessary to use the ``static_argnums`` parameter. For example,
    consider a boolean flag argument::

      from functools import partial

      @partial(jax.checkpoint, static_argnums=(1,))
      def foo(x, is_training):
        if is_training:
          ...
        else:
          ...

    Here, the use of ``static_argnums`` allows the ``if`` statement's condition
    to depends on the value of ``is_training``. The cost to using
    ``static_argnums`` is that it introduces re-tracing overheads across calls:
    in the example, ``foo`` is re-traced every time it is called with a new value
    of ``is_training``. In some situations, ``jax.ensure_compile_time_eval``
    is needed as well::

      @partial(jax.checkpoint, static_argnums=(1,))
      def foo(x, y):
        with jax.ensure_compile_time_eval():
          y_pos = y > 0
        if y_pos:
          ...
        else:
          ...

    As an alternative to using ``static_argnums`` (and
    ``jax.ensure_compile_time_eval``), it may be easier to compute some values
    outside the :func:`jax.checkpoint`-decorated function and then close over them.
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
        state_trace = fun.compile_function_and_get_state_trace(*args, **params, return_only_write=True)
        read_state_vals = state_trace.get_read_state_values()
        # call the checkpointed function
        write_state_vals, outs = checkpointed_fun(state_trace.get_state_values(), *args, **params)
        # write the state values back to the states
        write_back_state_values(state_trace, read_state_vals, write_state_vals)
        return outs

    return remat_fun


remat = checkpoint
