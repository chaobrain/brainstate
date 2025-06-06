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

import math
from typing import Any, Callable, TypeVar

import jax

from brainstate._utils import set_module_as
from ._loop_collect_return import _bounded_while_loop
from ._make_jaxpr import StatefulFunction
from ._util import wrap_single_fun_in_multi_branches_while_loop as wrap_fn
from ._util import write_back_state_values

X = TypeVar('X')
Y = TypeVar('Y')
T = TypeVar('T')
Carry = TypeVar('Carry')
BooleanNumeric = Any  # A bool, or a Boolean array.

__all__ = [
    'while_loop', 'bounded_while_loop',
]


@set_module_as('brainstate.compile')
def while_loop(
    cond_fun: Callable[[T], BooleanNumeric],
    body_fun: Callable[[T], T],
    init_val: T
) -> T:
    """
    Call ``body_fun`` repeatedly in a loop while ``cond_fun`` is True.

    The `Haskell-like type signature`_ in brief is

    .. code-block:: haskell

      while_loop :: (a -> Bool) -> (a -> a) -> a -> a

    The semantics of ``while_loop`` are given by this Python implementation::

      def while_loop(cond_fun, body_fun, init_val):
        val = init_val
        while cond_fun(val):
          val = body_fun(val)
        return val

    Unlike that Python version, ``while_loop`` is a JAX primitive and is lowered
    to a single WhileOp. That makes it useful for reducing compilation times
    for jit-compiled functions, since native Python loop constructs in an ``@jit``
    function are unrolled, leading to large XLA computations.

    Also unlike the Python analogue, the loop-carried value ``val`` must hold a
    fixed shape and dtype across all iterations (and not just be consistent up to
    NumPy rank/shape broadcasting and dtype promotion rules, for example). In
    other words, the type ``a`` in the type signature above represents an array
    with a fixed shape and dtype (or a nested tuple/list/dict container data
    structure with a fixed structure and arrays with fixed shape and dtype at the
    leaves).

    Another difference from using Python-native loop constructs is that
    ``while_loop`` is not reverse-mode differentiable because XLA computations
    require static bounds on memory requirements.

    Args:
      cond_fun: function of type ``a -> Bool``.
      body_fun: function of type ``a -> a``.
      init_val: value of type ``a``, a type that can be a scalar, array, or any
        pytree (nested Python tuple/list/dict) thereof, representing the initial
        loop carry value.

    Returns:
      The output from the final iteration of body_fun, of type ``a``.

    .. _Haskell-like type signature: https://wiki.haskell.org/Type_signature
    """
    if not (callable(body_fun) and callable(cond_fun)):
        raise TypeError("while_loop: body_fun and cond_fun arguments should be callable.")
    if jax.config.jax_disable_jit:
        try:
            val = init_val
            while cond_fun(val):
                val = body_fun(val)
            return val
        except jax.core.ConcretizationTypeError:
            # Can't run this while_loop in Python (e.g. because there's a vmap
            # transformation on it), so we fall back to the primitive version.
            pass

    # evaluate jaxpr
    stateful_cond = StatefulFunction(cond_fun, name='while:cond').make_jaxpr(init_val)
    stateful_body = StatefulFunction(body_fun, name='while:body').make_jaxpr(init_val)
    if len(stateful_cond.get_write_states()) != 0:
        raise ValueError("while_loop: cond_fun should not have any write states.")

    # state trace and state values
    state_trace = stateful_cond.get_state_trace() + stateful_body.get_state_trace()
    read_state_vals = state_trace.get_read_state_values(True)
    write_state_vals = state_trace.get_write_state_values(True)
    new_cond_fn = wrap_fn(stateful_cond, state_trace, read_state_vals, False)
    new_body_fn = wrap_fn(stateful_body, state_trace, read_state_vals, True)

    # while_loop
    state_vals, final_val = jax.lax.while_loop(new_cond_fn, new_body_fn, (write_state_vals, init_val))

    # write back state values or restore them
    write_back_state_values(state_trace, read_state_vals, state_vals)
    return final_val


def bounded_while_loop(
    cond_fun: Callable[[T], BooleanNumeric],
    body_fun: Callable[[T], T],
    init_val: T,
    *,
    max_steps: int,
    base: int = 16,
):
    """
    While loop with a bound on the maximum number of steps.

    This function is adapted from ``while_loop`` in `equinox <https://github.com/patrick-kidger/equinox/blob/main/equinox/internal/_loop/loop.py>`_.

    This function is useful when you want to ensure that a while loop terminates
    even if the condition function is never false. The function is implemented
    using a scan operation, so it is reverse-mode differentiable.

    Args:
      cond_fun: A function of type ``a -> Bool``.
      body_fun: A function of type ``a -> a``.
      init_val: The initial value of type ``a``.
      max_steps: A bound on the maximum number of steps, after which the loop
        terminates unconditionally.
      base: Run time will increase slightly as `base` increases. Compilation time will
        decrease substantially as `math.ceil(math.log(max_steps, base))` decreases.
        (Which happens as `base` increases.)

    Returns:
      The final value, as if computed by a `lax.while_loop`.
    """

    # checking
    if not isinstance(max_steps, int) or max_steps < 0:
        raise ValueError("max_steps must be a non-negative integer")
    init_val = jax.tree.map(jax.numpy.asarray, init_val)
    if max_steps == 0:
        return init_val

    # evaluate jaxpr
    stateful_cond = StatefulFunction(cond_fun, name='bounded_while:cond').make_jaxpr(init_val)
    stateful_body = StatefulFunction(body_fun, name='bounded_while:body').make_jaxpr(init_val)
    if len(stateful_cond.get_write_states()) != 0:
        raise ValueError("while_loop: cond_fun should not have any write states.")

    # state trace and state values
    state_trace = stateful_cond.get_state_trace() + stateful_body.get_state_trace()
    read_state_vals = state_trace.get_read_state_values(True)
    write_state_vals = state_trace.get_write_state_values(True)
    new_cond_fn = wrap_fn(stateful_cond, state_trace, read_state_vals, False)
    new_body_fn = wrap_fn(stateful_body, state_trace, read_state_vals, True)

    # initial value
    init_val = (write_state_vals, init_val)

    # while_loop
    rounded_max_steps = base ** int(math.ceil(math.log(max_steps, base)))
    state_vals, val = _bounded_while_loop(new_cond_fn, new_body_fn, init_val, rounded_max_steps, base, None)

    # write back state values or restore them
    write_back_state_values(state_trace, read_state_vals, state_vals)
    return val
