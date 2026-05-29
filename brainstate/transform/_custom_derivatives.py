# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
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

from typing import Callable, Optional

import jax

from brainstate._state import State
from brainstate._utils import set_module_as

__all__ = ['custom_vjp', 'custom_jvp']


def _normalize_states(grad_states):
    if grad_states is None:
        return []
    if isinstance(grad_states, State):
        return [grad_states]
    states = list(grad_states)
    if any(not isinstance(s, State) for s in states):
        raise TypeError("grad_states must be State instances.")
    return states


class CustomVJP:
    """Reverse-mode custom differentiation rule over the State system.

    Wraps :func:`jax.custom_vjp` so a function that reads ``State`` objects can
    have custom gradients defined with respect to its positional arguments and a
    set of ``grad_states``. The differentiated function must be read-only with
    respect to states (state writes in the custom path are not supported in this
    version).

    Use :meth:`def_fwd` and :meth:`def_bwd` to register the rules; call the
    instance like the original function.

    Parameters
    ----------
    fun : callable
        The primal function ``fun(*args)``; may read ``State`` objects.
    grad_states : State or sequence of State, optional
        States whose gradients the backward rule provides, in order.

    See Also
    --------
    custom_jvp, vjp, grad

    Notes
    -----
    The backward rule must return a 2-tuple ``(state_grads, arg_grads)``:
    ``state_grads`` is a sequence aligned with ``grad_states`` (use ``()`` when
    there are none); ``arg_grads`` is a tuple aligned with the positional
    arguments, or a single value for a one-argument function.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> import jax, jax.numpy as jnp
        >>>
        >>> w = brainstate.State(jnp.array(2.0))
        >>> cf = brainstate.transform.custom_vjp(lambda x: w.value * x, grad_states=[w])
        >>>
        >>> @cf.def_fwd
        ... def fwd(x):
        ...     return w.value * x, (x, w.value)
        >>>
        >>> @cf.def_bwd
        ... def bwd(res, ct):
        ...     x, wv = res
        ...     return [ct * x], ct * wv
        >>>
        >>> cf(jnp.array(3.0))
        Array(6., dtype=float32, weak_type=True)
    """

    def __init__(self, fun: Callable, grad_states=None):
        self.fun = fun
        self._grad_states = _normalize_states(grad_states)
        self._fwd = None
        self._bwd = None

        @jax.custom_vjp
        def prim(state_vals, args):
            for s, v in zip(self._grad_states, state_vals):
                s.restore_value(v)
            return self.fun(*args)

        def prim_fwd(state_vals, args):
            if self._fwd is None:
                raise RuntimeError(
                    "custom_vjp: register a forward rule with .def_fwd before differentiating."
                )
            for s, v in zip(self._grad_states, state_vals):
                s.restore_value(v)
            return self._fwd(*args)

        def prim_bwd(residual, cotangent):
            if self._bwd is None:
                raise RuntimeError(
                    "custom_vjp: register a backward rule with .def_bwd before differentiating."
                )
            state_grads, arg_grads = self._bwd(residual, cotangent)
            sg = tuple(state_grads)
            ag = arg_grads if isinstance(arg_grads, tuple) else (arg_grads,)
            return (sg, ag)

        prim.defvjp(prim_fwd, prim_bwd)
        self._prim = prim

    def def_fwd(self, fwd: Callable) -> Callable:
        """Register the forward rule ``fwd(*args) -> (output, residual)``."""
        self._fwd = fwd
        return fwd

    def def_bwd(self, bwd: Callable) -> Callable:
        """Register the backward rule ``bwd(residual, cotangent) -> (state_grads, arg_grads)``."""
        self._bwd = bwd
        return bwd

    def __call__(self, *args):
        state_vals = tuple(s.value for s in self._grad_states)
        return self._prim(state_vals, args)


@set_module_as("brainstate.transform")
def custom_vjp(fun: Optional[Callable] = None, *, grad_states=None):
    """Define a state-aware custom reverse-mode (VJP) differentiation rule.

    Returns a :class:`CustomVJP` wrapping ``fun``. Register the forward and
    backward rules with ``.def_fwd`` / ``.def_bwd``. Usable directly
    (``custom_vjp(fun, grad_states=...)``) or as a decorator
    (``@custom_vjp`` / ``@custom_vjp(grad_states=...)``).

    Parameters
    ----------
    fun : callable, optional
        The primal function. If ``None``, a decorator is returned.
    grad_states : State or sequence of State, optional
        States whose gradients the backward rule provides.

    Returns
    -------
    CustomVJP or callable
        A :class:`CustomVJP` instance, or a decorator when ``fun`` is ``None``.

    See Also
    --------
    custom_jvp, vjp, grad
    """
    if fun is None:
        return lambda f: CustomVJP(f, grad_states=grad_states)
    return CustomVJP(fun, grad_states=grad_states)


class CustomJVP:
    """Forward-mode custom differentiation rule over the State system.

    Wraps :func:`jax.custom_jvp`. In this version the rule differentiates with
    respect to the positional arguments; states read inside ``fun`` are treated
    as constants. Register the rule with :meth:`def_jvp`.

    Parameters
    ----------
    fun : callable
        The primal function ``fun(*args)``; may read ``State`` objects.

    See Also
    --------
    custom_vjp, jvp, jacfwd

    Notes
    -----
    The jvp rule has the signature ``rule(primals, tangents) -> (output,
    tangent_output)``, where ``primals`` is the tuple of positional arguments and
    ``tangents`` is the matching tuple of input tangents — the same convention as
    :func:`jax.custom_jvp`.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> import jax, jax.numpy as jnp
        >>>
        >>> @brainstate.transform.custom_jvp
        ... def square(x):
        ...     return x ** 2
        >>>
        >>> @square.def_jvp
        ... def square_jvp(primals, tangents):
        ...     (x,), (xt,) = primals, tangents
        ...     return x ** 2, 2.0 * x * xt
        >>>
        >>> square(jnp.array(4.0))
        Array(16., dtype=float32, weak_type=True)
    """

    def __init__(self, fun: Callable):
        self.fun = fun
        self._jvp = None

        @jax.custom_jvp
        def prim(args):
            return self.fun(*args)

        @prim.defjvp
        def prim_jvp(primals, tangents):
            if self._jvp is None:
                raise RuntimeError(
                    "custom_jvp: register a jvp rule with .def_jvp before differentiating."
                )
            (args,) = primals
            (arg_tangents,) = tangents
            return self._jvp(args, arg_tangents)

        self._prim = prim

    def def_jvp(self, jvp: Callable) -> Callable:
        """Register the jvp rule ``rule(primals, tangents) -> (output, tangent_output)``."""
        self._jvp = jvp
        return jvp

    def __call__(self, *args):
        return self._prim(args)


@set_module_as("brainstate.transform")
def custom_jvp(fun: Optional[Callable] = None):
    """Define a state-aware custom forward-mode (JVP) differentiation rule.

    Returns a :class:`CustomJVP` wrapping ``fun``. Register the rule with
    ``.def_jvp``. Usable directly (``custom_jvp(fun)``) or as a decorator
    (``@custom_jvp``).

    Parameters
    ----------
    fun : callable, optional
        The primal function. If ``None``, a decorator is returned.

    Returns
    -------
    CustomJVP or callable
        A :class:`CustomJVP` instance, or a decorator when ``fun`` is ``None``.

    See Also
    --------
    custom_vjp, jvp, jacfwd
    """
    if fun is None:
        return lambda f: CustomJVP(f)
    return CustomJVP(fun)
