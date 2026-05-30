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

from typing import Any, Callable, Dict, Optional, Sequence, Union

import jax

from brainstate._state import State
from brainstate._utils import set_module_as
from ._make_jaxpr import StatefulFunction

__all__ = ['vjp', 'jvp']


def _flatten_grad_states(grad_states):
    """Flatten a State / sequence / dict of States into (list_of_states, treedef)."""
    states, tree = jax.tree.flatten(grad_states, is_leaf=lambda x: isinstance(x, State))
    if any(not isinstance(s, State) for s in states):
        raise TypeError("All grad_states must be State instances.")
    return states, tree


@set_module_as("brainstate.transform")
def vjp(
    fun: Callable,
    *primals,
    grad_states: Optional[Union[State, Sequence[State], Dict[str, State]]] = None,
    argnums: Optional[Union[int, Sequence[int]]] = 0,
    has_aux: bool = False,
):
    """Compute a state-aware vector-Jacobian product (reverse-mode autodiff).

    Trace ``fun`` (which may read and write :class:`~brainstate.State` objects)
    into a pure function, apply :func:`jax.vjp`, re-thread any written states
    back into their ``State`` objects, and return the primal output together
    with a pullback (a.k.a. *cotangent map* or *backward function*).

    Calling the returned ``vjp_fn`` with a cotangent ``v`` (a tangent of the
    output) yields ``v @ J``, where ``J`` is the Jacobian of ``fun`` evaluated
    at ``primals``. This is the building block of reverse-mode autodiff: a
    single forward trace amortizes arbitrarily many backward passes, so it is
    the natural primitive for full Jacobians (one row per output) and for
    higher-order products such as Hessian-vector products.

    What ``vjp_fn`` returns depends on ``grad_states`` and ``argnums``,
    mirroring :func:`brainstate.transform.grad`:

    +----------------+--------------------------+----------------------------+
    | ``grad_states``| ``argnums``              | ``vjp_fn(v)`` returns      |
    +================+==========================+============================+
    | ``None``       | ``int`` / sequence       | ``arg_cotangents``         |
    +----------------+--------------------------+----------------------------+
    | provided       | ``int`` / sequence       | ``(state_cts, arg_cts)``   |
    +----------------+--------------------------+----------------------------+
    | provided       | ``None`` (or no primals) | ``state_cotangents``       |
    +----------------+--------------------------+----------------------------+

    Parameters
    ----------
    fun : callable
        Function to be differentiated. It may read and/or write ``State``
        objects. If ``has_aux`` is ``True`` it must return ``(output, aux)``.
    *primals
        Positional arguments at which to evaluate ``fun`` and its pullback.
        May be omitted entirely when differentiating only with respect to
        ``grad_states`` (e.g. a parameterized model whose inputs are closed
        over or supplied through states).
    grad_states : State, sequence of State, or dict of State, optional
        States to compute cotangents for. The returned state cotangents follow
        the structure of ``grad_states`` (an unwrapped array for a single
        ``State``, a list for a sequence, a matching ``dict`` for a mapping).
        When given, the pullback also returns argument cotangents unless
        ``argnums`` is ``None`` or no positional ``primals`` are supplied.
    argnums : int, sequence of int, or None, default 0
        Positional argument(s) to differentiate with respect to. A single
        ``int`` yields an unwrapped argument cotangent; a sequence yields a
        tuple, one entry per index. ``None`` disables argument differentiation
        so the pullback returns only state cotangents (requires
        ``grad_states``). If ``fun`` is called with no positional ``primals``,
        argument differentiation is disabled automatically.
    has_aux : bool, default False
        Whether ``fun`` returns ``(output, aux)``. The auxiliary data is not
        differentiated but is returned to the caller.

    Returns
    -------
    primal_out : PyTree
        The value ``fun(*primals)`` (the first element if ``has_aux``).
    vjp_fn : callable
        Pullback mapping a cotangent of ``primal_out`` to input cotangents.
        The cotangent passed to ``vjp_fn`` must have the same pytree structure,
        shape, and dtype as ``primal_out``. See the table above for the return
        structure.
    aux : PyTree
        Returned only when ``has_aux`` is ``True``.

    Raises
    ------
    TypeError
        If any entry of ``grad_states`` is not a :class:`~brainstate.State`.
    ValueError
        If a ``grad_state`` is never read by ``fun`` (so its cotangent is
        undefined), if ``argnums`` is out of range for the given ``primals``,
        or if there is nothing to differentiate (no ``primals`` and no
        ``grad_states``).

    See Also
    --------
    jvp, grad, jacrev

    Notes
    -----
    States that ``fun`` *writes* are re-threaded back into their ``State``
    objects after the forward trace (the same side effects ``fun`` would have
    produced if called directly). Differentiation is always taken with respect
    to the *input* value a state held on entry, so reading-then-writing a
    ``grad_state`` is well defined.

    Because ``vjp`` traces ``fun`` once and returns a reusable pullback, it is
    strictly more flexible than :func:`grad` when you need (a) multiple
    backward passes, (b) a non-scalar output, or (c) a custom cotangent ``v``.
    Evaluating ``vjp_fn(1.0)`` on a scalar-output ``fun`` reproduces
    :func:`grad` exactly.

    Examples
    --------
    **Plain reverse-mode autodiff (no states).** ``vjp`` matches
    :func:`jax.vjp` on a pure function; with a scalar ``int`` ``argnums`` the
    argument cotangent is returned unwrapped.

    .. code-block:: python

        >>> import brainstate
        >>> import jax.numpy as jnp
        >>> def f(x):
        ...     return jnp.sum(x ** 2)
        >>> x = jnp.array([1.0, 2.0, 3.0])
        >>> out, vjp_fn = brainstate.transform.vjp(f, x)
        >>> out
        Array(14., dtype=float32)
        >>> vjp_fn(1.0)            # d/dx sum(x**2) = 2x
        Array([2., 4., 6.], dtype=float32)

    **Gradients with respect to states.** Pass ``grad_states`` to also obtain
    state cotangents. The pullback then returns ``(state_cts, arg_cts)``.

    .. code-block:: python

        >>> w = brainstate.State(jnp.array([2.0, 3.0]))
        >>> def loss(x):
        ...     return jnp.sum(w.value * x)
        >>> x = jnp.array([5.0, 7.0])
        >>> out, vjp_fn = brainstate.transform.vjp(loss, x, grad_states=w)
        >>> state_ct, arg_ct = vjp_fn(1.0)
        >>> state_ct              # d/dw sum(w*x) = x
        Array([5., 7.], dtype=float32)
        >>> arg_ct                # d/dx sum(w*x) = w
        Array([2., 3.], dtype=float32)

    **State-only gradients (no differentiable argument).** This is the typical
    neural-network case: the loss closes over the trainable parameters, so the
    pullback returns just the state cotangents.

    .. code-block:: python

        >>> weight = brainstate.State(jnp.array([[1.0, 2.0], [3.0, 4.0]]))
        >>> bias = brainstate.State(jnp.array([0.0, 0.0]))
        >>> x = jnp.array([1.0, 1.0])
        >>> def predict_loss():
        ...     y = x @ weight.value + bias.value
        ...     return jnp.sum(y ** 2)
        >>> out, vjp_fn = brainstate.transform.vjp(predict_loss, grad_states=[weight, bias])
        >>> grads = vjp_fn(1.0)           # list of state cotangents, no arg cotangent
        >>> [g.shape for g in grads]
        [(2, 2), (2,)]

    **Auxiliary data and state write-back.** ``has_aux=True`` returns the side
    output untouched; states written inside ``fun`` keep their new values.

    .. code-block:: python

        >>> counter = brainstate.State(jnp.array(0.0))
        >>> def f(x):
        ...     counter.value = counter.value + 1.0
        ...     return jnp.sum(x ** 2), {'mean': jnp.mean(x)}
        >>> x = jnp.array([1.0, 2.0])
        >>> out, vjp_fn, aux = brainstate.transform.vjp(f, x, has_aux=True)
        >>> aux['mean']
        Array(1.5, dtype=float32)
        >>> vjp_fn(1.0)
        Array([2., 4.], dtype=float32)
        >>> float(counter.value)  # write re-threaded back into the State
        1.0

    **Full Jacobian by reusing the pullback.** One trace, many backward passes:
    map the pullback over the rows of the identity to build the Jacobian.

    .. code-block:: python

        >>> import jax
        >>> def f(x):
        ...     return jnp.array([jnp.sum(x), jnp.sum(x ** 2)])
        >>> x = jnp.array([1.0, 2.0, 3.0])
        >>> out, vjp_fn = brainstate.transform.vjp(f, x)
        >>> jac = jax.vmap(vjp_fn)(jnp.eye(2))
        >>> jac                   # rows are gradients of each output
        Array([[1., 1., 1.],
               [2., 4., 6.]], dtype=float32)

    **Multiple arguments.** A sequence ``argnums`` returns a tuple of
    cotangents, one per requested argument.

    .. code-block:: python

        >>> def f(x, y):
        ...     return jnp.sum(x * y)
        >>> x = jnp.array([1.0, 2.0])
        >>> y = jnp.array([3.0, 4.0])
        >>> out, vjp_fn = brainstate.transform.vjp(f, x, y, argnums=(0, 1))
        >>> gx, gy = vjp_fn(1.0)
        >>> gx, gy                # (d/dx, d/dy) sum(x*y) = (y, x)
        (Array([3., 4.], dtype=float32), Array([1., 2.], dtype=float32))
    """
    grad_state_list, grad_tree = _flatten_grad_states(grad_states)
    grad_ids = [id(s) for s in grad_state_list]
    grad_id_set = set(grad_ids)
    has_states = len(grad_state_list) > 0

    # Resolve which positional arguments (if any) to differentiate. Mirrors the
    # semantics of ``brainstate.transform.grad``:
    #   * ``argnums=None``      -> differentiate states only (no arg cotangent).
    #   * no positional primals -> differentiate states only (nothing else).
    #   * otherwise             -> differentiate the selected positional args.
    if argnums is None or len(primals) == 0:
        diff_args = False
        single_arg = False
        argnums_t: tuple = ()
    else:
        single_arg = isinstance(argnums, int)
        argnums_t = (argnums,) if single_arg else tuple(argnums)
        n_primals = len(primals)
        for i in argnums_t:
            if not -n_primals <= i < n_primals:
                raise ValueError(
                    f"argnums {i} is out of range for a function called with "
                    f"{n_primals} positional argument(s)."
                )
        diff_args = True

    if not diff_args and not has_states:
        raise ValueError(
            "vjp has nothing to differentiate: supply positional arguments to "
            "differentiate (controlled by `argnums`) and/or `grad_states`."
        )

    sf = StatefulFunction(fun, name='vjp', return_only_write=True)
    sf.make_jaxpr(*primals)
    cache = sf.get_arg_cache_key(*primals)
    state_trace = sf.get_state_trace_by_cache(cache)
    read_state_vals = state_trace.get_read_state_values(True)

    grad_vals: Dict[int, Any] = {}
    other_vals: Dict[int, Any] = {}
    for st in state_trace.states:
        if id(st) in grad_id_set:
            grad_vals[id(st)] = st.value
        else:
            other_vals[id(st)] = st.value

    missing = grad_id_set - set(grad_vals.keys())
    if missing:
        raise ValueError(
            "Some grad_states are not used by the function and cannot be "
            "differentiated. Ensure every grad_state is read inside `fun`."
        )

    diff_primals = tuple(primals[i] for i in argnums_t)

    def _merge(gvals: Dict[int, Any]):
        return [gvals[id(st)] if id(st) in gvals else other_vals[id(st)]
                for st in state_trace.states]

    def _pure(gvals, diff_arg_vals):
        full = list(primals)
        for pos, i in enumerate(argnums_t):
            full[i] = diff_arg_vals[pos]
        write_state_vals, raw_out = sf.jaxpr_call(_merge(gvals), *full)
        if has_aux:
            primal_out, aux = raw_out
        else:
            primal_out, aux = raw_out, None
        return primal_out, (write_state_vals, aux)

    primal_out, pull, (write_state_vals, aux) = jax.vjp(
        _pure, grad_vals, diff_primals, has_aux=True
    )

    # Re-thread written state values back into the State objects.
    state_trace.assign_state_vals_v2(read_state_vals, write_state_vals)

    def vjp_fn(cotangent):
        gval_ct, diff_arg_ct = pull(cotangent)
        if has_states:
            state_ct = grad_tree.unflatten([gval_ct[i] for i in grad_ids])
            if diff_args:
                arg_ct = diff_arg_ct[0] if single_arg else diff_arg_ct
                return state_ct, arg_ct
            return state_ct
        return diff_arg_ct[0] if single_arg else diff_arg_ct

    if has_aux:
        return primal_out, vjp_fn, aux
    return primal_out, vjp_fn


@set_module_as("brainstate.transform")
def jvp(
    fun: Callable,
    primals: Sequence,
    tangents: Sequence,
    *,
    has_aux: bool = False,
):
    """Compute a state-aware Jacobian-vector product (forward-mode autodiff).

    Trace ``fun`` (which may read and write :class:`~brainstate.State` objects)
    into a pure function, apply :func:`jax.jvp` with respect to the positional
    arguments, and re-thread any written states back into their ``State``
    objects.

    States are treated as constants for the forward pass (zero tangent); states
    written inside ``fun`` are still updated. Differentiating with respect to
    state values (state tangents) is a future enhancement.

    Parameters
    ----------
    fun : callable
        Function to be differentiated. It may read/write ``State`` objects. If
        ``has_aux`` is ``True`` it must return ``(output, aux)``.
    primals : sequence
        The positional arguments at which to evaluate ``fun``, as a tuple/list
        matching ``fun``'s signature.
    tangents : sequence
        Tangent vectors, a tuple/list with the same structure as ``primals``.
    has_aux : bool, default False
        Whether ``fun`` returns ``(output, aux)``. The auxiliary data is not
        differentiated.

    Returns
    -------
    primal_out : PyTree
        The value ``fun(*primals)`` (the first element if ``has_aux``).
    tangent_out : PyTree
        The directional derivative of ``fun`` along ``tangents``.
    aux : PyTree
        Returned only when ``has_aux`` is ``True``.

    Raises
    ------
    TypeError
        If ``primals`` or ``tangents`` is not a tuple/list.

    See Also
    --------
    vjp, jacfwd, fwd_grad

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> import jax.numpy as jnp
        >>> def f(x):
        ...     return jnp.sum(x ** 2)
        >>> out, tangent = brainstate.transform.jvp(f, (jnp.array([1.0, 2.0]),),
        ...                                          (jnp.array([1.0, 1.0]),))
    """
    if not isinstance(primals, (tuple, list)):
        raise TypeError("`primals` must be a tuple or list of positional arguments.")
    if not isinstance(tangents, (tuple, list)):
        raise TypeError("`tangents` must be a tuple or list matching `primals`.")
    primals = tuple(primals)
    tangents = tuple(tangents)

    sf = StatefulFunction(fun, name='jvp', return_only_write=True)
    sf.make_jaxpr(*primals)
    cache = sf.get_arg_cache_key(*primals)
    state_trace = sf.get_state_trace_by_cache(cache)
    read_state_vals = state_trace.get_read_state_values(True)

    # States captured as constants (not differentiated in forward mode).
    state_vals = [st.value for st in state_trace.states]

    def _pure(prim_args):
        write_state_vals, raw_out = sf.jaxpr_call(list(state_vals), *prim_args)
        if has_aux:
            primal_out, aux = raw_out
        else:
            primal_out, aux = raw_out, None
        return primal_out, (write_state_vals, aux)

    primal_out, tangent_out, (write_state_vals, aux) = jax.jvp(
        _pure, (primals,), (tangents,), has_aux=True
    )

    # Re-thread written state values back into the State objects.
    state_trace.assign_state_vals_v2(read_state_vals, write_state_vals)

    if has_aux:
        return primal_out, tangent_out, aux
    return primal_out, tangent_out
