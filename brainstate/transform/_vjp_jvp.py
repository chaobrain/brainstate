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
    argnums: Union[int, Sequence[int]] = 0,
    has_aux: bool = False,
):
    """Compute a state-aware vector-Jacobian product (reverse-mode autodiff).

    Trace ``fun`` (which may read and write :class:`~brainstate.State` objects)
    into a pure function, apply :func:`jax.vjp`, re-thread any written states
    back into their ``State`` objects, and return the primal output together
    with a pullback function.

    Parameters
    ----------
    fun : callable
        Function to be differentiated. It may read/write ``State`` objects. If
        ``has_aux`` is ``True`` it must return ``(output, aux)``.
    *primals
        Positional arguments at which to evaluate ``fun`` and its pullback.
    grad_states : State, sequence of State, or dict of State, optional
        States to compute cotangents for. When given, the pullback returns
        ``(state_cotangents, arg_cotangents)``.
    argnums : int or sequence of int, default 0
        Positional argument(s) to differentiate with respect to. A single
        ``int`` yields an unwrapped argument cotangent; a sequence yields a
        tuple.
    has_aux : bool, default False
        Whether ``fun`` returns ``(output, aux)``. The auxiliary data is not
        differentiated.

    Returns
    -------
    primal_out : PyTree
        The value ``fun(*primals)`` (the first element if ``has_aux``).
    vjp_fn : callable
        Pullback mapping a cotangent of ``primal_out`` to input cotangents.
        Returns ``arg_cotangents`` when ``grad_states`` is ``None``, else
        ``(state_cotangents, arg_cotangents)``.
    aux : PyTree
        Returned only when ``has_aux`` is ``True``.

    See Also
    --------
    jvp, grad, jacrev

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> import jax.numpy as jnp
        >>> w = brainstate.State(jnp.array([2.0, 3.0]))
        >>> def f(x):
        ...     return jnp.sum(w.value * x)
        >>> out, vjp_fn = brainstate.transform.vjp(f, jnp.array([5.0, 7.0]), grad_states=w)
        >>> state_ct, arg_ct = vjp_fn(1.0)
    """
    single_arg = isinstance(argnums, int)
    argnums_t = (argnums,) if single_arg else tuple(argnums)

    grad_state_list, grad_tree = _flatten_grad_states(grad_states)
    grad_ids = [id(s) for s in grad_state_list]
    grad_id_set = set(grad_ids)
    has_states = len(grad_state_list) > 0

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

    def _pure(gvals, diff_args):
        full = list(primals)
        for pos, i in enumerate(argnums_t):
            full[i] = diff_args[pos]
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
        arg_ct = diff_arg_ct[0] if single_arg else diff_arg_ct
        if has_states:
            state_ct = grad_tree.unflatten([gval_ct[i] for i in grad_ids])
            return state_ct, arg_ct
        return arg_ct

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
