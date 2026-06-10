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

from __future__ import annotations

from functools import wraps
from typing import Any, Callable

from jax.experimental import checkify as _cfy

from brainstate._state import StateTraceStack
from brainstate._utils import set_module_as
from brainstate.typing import ArrayLike
from ._make_jaxpr import StatefulFunction

__all__ = [
    'checkify', 'check', 'check_error',
    'all_checks', 'user_checks', 'nan_checks', 'div_checks',
    'index_checks', 'float_checks', 'automatic_checks',
]

# Error-category sets re-exported from JAX for convenience, so callers can
# select what to check without importing ``jax.experimental.checkify`` directly.
all_checks = _cfy.all_checks
user_checks = _cfy.user_checks
nan_checks = _cfy.nan_checks
div_checks = _cfy.div_checks
index_checks = _cfy.index_checks
float_checks = _cfy.float_checks
automatic_checks = _cfy.automatic_checks


@set_module_as('brainstate.transform')
def check(pred: ArrayLike, msg: str, *fmt_args, debug: bool = False, **fmt_kwargs) -> None:
    """Assert a runtime condition inside a :func:`checkify`-wrapped function.

    A state-transparent re-export of :func:`jax.experimental.checkify.check`.
    When the wrapping function is transformed by :func:`checkify`, a false
    ``pred`` is functionalized into the threaded ``Error`` rather than raising
    immediately, so the check survives ``jit``/``vmap``/``scan``.

    Parameters
    ----------
    pred : bool or Array
        The condition that must hold. ``False`` records an error.
    msg : str
        The error message. May contain ``{}``/``{name}`` fields filled from
        ``fmt_args``/``fmt_kwargs`` (traced values are allowed).
    *fmt_args
        Positional format arguments for ``msg``.
    debug : bool, default False
        If ``True``, the check is treated as a debug-only check.
    **fmt_kwargs
        Keyword format arguments for ``msg``.

    See Also
    --------
    checkify, check_error
    """
    _cfy.check(pred, msg, *fmt_args, debug=debug, **fmt_kwargs)


@set_module_as('brainstate.transform')
def check_error(error: _cfy.Error) -> None:
    """Re-raise a previously captured :class:`Error` inside a checkified context.

    A re-export of :func:`jax.experimental.checkify.check_error`. Use it to
    propagate an ``Error`` returned by one :func:`checkify`-wrapped function into
    the error state of an enclosing one.

    Parameters
    ----------
    error : jax.experimental.checkify.Error
        The error value to propagate.

    See Also
    --------
    checkify, check
    """
    _cfy.check_error(error)


@set_module_as('brainstate.transform')
def checkify(fun: Callable, errors: Any = user_checks) -> Callable:
    """Functionalize runtime error checks in a stateful function.

    A state-aware wrapper over :func:`jax.experimental.checkify.checkify`. The
    returned function runs ``fun`` with its error checks (NaN, division-by-zero,
    out-of-bounds, and user :func:`check` assertions, as selected by ``errors``)
    threaded into an :class:`~jax.experimental.checkify.Error` value instead of
    raising on the host. This survives ``jit``/``vmap``/``scan`` without host
    callbacks. ``State`` reads and writes performed by ``fun`` are handled
    transparently: writes are applied after the call, reads are left unchanged.

    Unlike :func:`jit_error_if` (a fire-and-forget host ``debug.callback``),
    ``checkify`` is *functional* and *composable*: the caller receives the
    ``Error`` and decides when to inspect (``err.get()``) or raise (``err.throw()``).

    Parameters
    ----------
    fun : callable
        The function to check. May read and write ``State`` objects and call
        :func:`check`.
    errors : frozenset, default :data:`user_checks`
        The set of error categories to enable. Use the re-exported sets
        :data:`user_checks`, :data:`nan_checks`, :data:`div_checks`,
        :data:`index_checks`, :data:`float_checks`, :data:`automatic_checks`, or
        :data:`all_checks`.

    Returns
    -------
    callable
        A function with the same signature as ``fun`` that returns a tuple
        ``(error, out)``, where ``error`` is a
        :class:`~jax.experimental.checkify.Error` and ``out`` is ``fun``'s
        original output.

    See Also
    --------
    check, check_error, jit_error_if

    Notes
    -----
    ``fun`` is *re-run* under :func:`jax.experimental.checkify.checkify` (rather
    than replaying a pre-traced jaxpr) so that ``check`` primitives are emitted
    directly into checkify's trace and functionalized natively. The states
    ``fun`` touches are discovered once via :class:`StatefulFunction`; after the
    call every state is restored (writes to their new values, reads to their
    originals) so no tracer leaks into the global ``State`` objects.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> import jax.numpy as jnp
        >>>
        >>> def safe_double(x):
        ...     brainstate.transform.check(x > 0, 'x must be positive')
        ...     return x * 2.0
        >>>
        >>> checked = brainstate.transform.checkify(safe_double)
        >>> err, out = checked(jnp.array(3.0))
        >>> err.throw()   # no error: does nothing
        >>> out
        Array(6., dtype=float32, weak_type=True)
    """

    @wraps(fun)
    def wrapped(*args, **kwargs):
        # 1. Discover the states the function touches.
        sf = StatefulFunction(fun, name='checkify')
        sf.make_jaxpr(*args, **kwargs)
        cache = sf.get_arg_cache_key(*args, **kwargs)
        trace = sf.get_state_trace_by_cache(cache)
        all_states = tuple(trace.states)
        write_states = tuple(sf.get_write_states(*args, **kwargs))
        orig_vals = tuple(st.value for st in all_states)

        # 2. Re-runnable pure function: inject states, run fun, read writes out.
        def pure(state_vals, p_args, p_kwargs):
            for st, v in zip(all_states, state_vals):
                st.restore_value(v)
            # plain watcher (no new_arg): the user function runs under the
            # raw jax checkify trace here, and state writes of its tracers
            # are legitimate — an active StateTraceStack keeps the
            # tracer-write guard quiet
            with StateTraceStack(name='checkify:run'):
                out = fun(*p_args, **p_kwargs)
                new_write_vals = tuple(st.value for st in write_states)
            return out, new_write_vals

        # 3. Functionalize the checks; the Error is threaded out.
        checked = _cfy.checkify(pure, errors)
        try:
            err, (out, write_vals) = checked(orig_vals, args, kwargs)
        except Exception:
            # a failure mid-trace must not leave tracers in the states
            for st, ov in zip(all_states, orig_vals):
                st.restore_value(ov)
            raise

        # 4. Restore ALL states: writes -> new values, reads -> originals
        #    (prevents tracers leaking into the global State objects).
        wv_by_id = {id(st): v for st, v in zip(write_states, write_vals)}
        for st, ov in zip(all_states, orig_vals):
            st.restore_value(wv_by_id[id(st)] if id(st) in wv_by_id else ov)

        return err, out

    return wrapped
