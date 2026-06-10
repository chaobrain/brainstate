# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
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

"""
NaN / Inf debugging utilities, built on :mod:`jax.experimental.checkify`.

Why checkify
------------
An earlier version of this module re-implemented a jaxpr interpreter that
replayed every primitive and recursed by hand into ``jit`` / ``cond`` /
``while`` / ``scan``.  That approach was fragile: it broke on the modern
``Primitive.get_bind_params`` contract, mishandled ``while`` loop constants,
and silently failed on higher-order primitives such as ``custom_jvp_call``,
``custom_vjp_call`` and ``remat`` — exactly the primitives produced by
gradients, ``vmap`` and neural-network activations.

``checkify`` is a JAX transform, so it composes *natively* with ``grad``,
``vmap``, ``jit``, ``scan``, ``while_loop``, ``cond`` and custom-derivative
primitives.  Re-basing the detector on it removes the entire hand-written
interpreter and the whole class of bugs that came with it.

Two-layer detection
--------------------
``checkify`` finds *where* a NaN (or division-by-zero) is produced.  But a
per-primitive checker also flags NaN that is computed inside a branch / lane
that is subsequently *masked away* (``jnp.where`` "safe" patterns, a
``vmap``-ed ``cond`` lowered to ``select_n``).  Those NaN never reach the
result and should not be reported.

To avoid such false positives we add an **output-contamination gate**: after
running the function we only surface an error when NaN/Inf actually reaches
the function's observable outputs (its return value *and* any updated state).
``checkify`` supplies the source primitive and location; the gate decides
whether to raise.

Inf localization
----------------
``checkify`` localizes NaN and division-by-zero to the producing primitive
with an IDE-clickable source line.  It does **not** localize other Infs
(e.g. ``exp`` overflow or ``log(0) == -inf``).  Such Infs are still *detected*
by the output gate and reported with per-output diagnostics, but without a
primitive-level source.
"""

from __future__ import annotations

import functools
import traceback as tb_module
from typing import Any, Callable, List, Tuple

import jax
import jax.numpy as jnp
from jax._src import traceback_util as _traceback_util
from jax.experimental import checkify
from jax.extend import source_info_util

from brainstate._compatible_import import Tracer
from brainstate.typing import ArrayLike
from ._conditions import cond
from ._make_jaxpr import StatefulFunction
from ._unvmap import unvmap

__all__ = [
    'breakpoint_if',
    'debug_nan',
    'debug_nan_if',
]

# checkify error categories used for detection: NaN generation and
# division-by-zero (which is the most common source of Inf).
_FLOAT_CHECKS = checkify.float_checks


# ---------------------------------------------------------------------------
# Output-gate helpers: does an array (or list of arrays) carry NaN / Inf?
# ---------------------------------------------------------------------------

def _is_inexact_array(x) -> bool:
    """Return True for floating *and* complex arrays (``jnp.inexact``)."""
    return hasattr(x, 'dtype') and jnp.issubdtype(x.dtype, jnp.inexact)


def _bad_elements(x) -> jax.Array:
    """Elementwise boolean mask of NaN-or-Inf entries (works for complex)."""
    return jnp.isnan(x) | jnp.isinf(x)


def _has_nan_flag(vals) -> jax.Array:
    """
    Return a scalar boolean JAX array: True iff any inexact value is NaN or Inf.

    Integer / boolean arrays are ignored.  Complex arrays are scanned on both
    their real and imaginary parts (``jnp.isnan`` / ``jnp.isinf`` already do
    this elementwise).
    """
    flags = [jnp.any(_bad_elements(v)) for v in vals if _is_inexact_array(v)]
    if not flags:
        return jnp.array(False)
    return jnp.any(jnp.stack(flags))


# ---------------------------------------------------------------------------
# Source info: IDE-clickable location strings
# ---------------------------------------------------------------------------

def _extract_user_source(obj) -> str:
    """
    Extract a human-readable, IDE-clickable source location.

    Accepts either a jaxpr equation's ``source_info`` (which carries a
    ``traceback`` attribute) or a raw ``jaxlib`` ``Traceback`` object (such as
    the one attached to a :mod:`checkify` error).

    Uses JAX's ``filter_traceback`` — which respects ``register_exclusion``
    registrations from brainstate and other libraries — then further strips any
    remaining ``site-packages`` frames so only genuine user frames are shown.
    Returns the innermost user frame in standard Python traceback format
    ``File "path", line N, in func_name`` which VSCode / PyCharm hyperlink.
    """
    if obj is None:
        return "<unknown source>"
    # A source_info exposes ``.traceback``; a raw Traceback exposes
    # ``.as_python_traceback`` directly.
    tb = getattr(obj, 'traceback', None)
    if tb is None and hasattr(obj, 'as_python_traceback'):
        tb = obj
    if tb is None:
        return "<unknown source>"

    def _innermost_user_frame(py_tb) -> str:
        """Return the innermost frame string that is not a ``site-packages`` frame."""
        if py_tb is None:
            return ""
        lines = tb_module.format_tb(py_tb)
        lines = [l for l in lines if '/site-packages/' not in l and '\\site-packages\\' not in l]
        return lines[-1].strip() if lines else ""

    try:
        py_tb = tb.as_python_traceback()
    except Exception:
        py_tb = None

    if py_tb is not None:
        # 1. JAX-aware filtering: respects ``register_exclusion`` registrations
        #    (brainstate registers its whole package), so this hides library
        #    internals and surfaces genuine user code.
        try:
            filtered_tb = _traceback_util.filter_traceback(py_tb)
        except Exception:
            filtered_tb = None
        src = _innermost_user_frame(filtered_tb)
        if src:
            return src
        # 2. Fallback: every frame was excluded (e.g. the NaN was produced
        #    entirely inside registered/library code).  Surface the innermost
        #    non-``site-packages`` frame so the report still points somewhere
        #    instead of giving up.
        src = _innermost_user_frame(py_tb)
        if src:
            return src

    try:
        return source_info_util.summarize(obj)
    except Exception:
        return "<unknown source>"


# ---------------------------------------------------------------------------
# checkify error introspection + message formatting
# ---------------------------------------------------------------------------

def _error_info(err) -> Tuple[Any, Any, Any]:
    """
    Return ``(primitive_name, traceback, detail)`` from a checkify ``Error``.

    Any element may be ``None``:

    * ``primitive_name`` is set for :class:`NaNError` (e.g. ``'log'``) and
      ``None`` for :class:`DivisionByZeroError` and Inf-only contamination.
    * ``traceback`` is the raw JAX ``Traceback`` recorded at the failing
      primitive (present for NaN and division errors).
    * ``detail`` is checkify's own human message (e.g.
      ``'nan generated by primitive: log.'``).
    """
    try:
        exc = err.get_exception()
    except Exception:
        exc = None
    if exc is None:
        return None, None, None
    prim = getattr(exc, 'prim', None)
    tb = getattr(exc, 'traceback_info', None)
    try:
        detail = err.get()
    except Exception:
        detail = None
    return prim, tb, detail


def _diagnose(out_leaves) -> List[str]:
    """Per-output diagnostics for every leaf that carries NaN / Inf."""
    lines: List[str] = []
    for i, v in enumerate(out_leaves):
        if not _is_inexact_array(v):
            continue
        try:
            n_nan = int(jnp.sum(jnp.isnan(v)))
            n_inf = int(jnp.sum(jnp.isinf(v)))
        except Exception:
            continue
        if n_nan == 0 and n_inf == 0:
            continue
        desc = (
            f"    output[{i}]: shape={tuple(jnp.shape(v))} dtype={v.dtype} "
            f"NaNs={n_nan} Infs={n_inf}"
        )
        if jnp.issubdtype(v.dtype, jnp.floating):
            try:
                desc += f" min={float(jnp.nanmin(v)):.4g} max={float(jnp.nanmax(v)):.4g}"
            except Exception:
                pass
        lines.append(desc)
    return lines


def _build_message(err, out_leaves, phase: str) -> Tuple[str, Any]:
    """Build the user-facing error message and return ``(message, traceback)``."""
    prim, tb, detail = _error_info(err)
    phase_tag = f" [{phase}]" if phase else ""
    lines = [f"NaN/Inf detected{phase_tag} in the output of the checked function!"]
    if prim is not None:
        lines.append(f"  Introduced by primitive : `{prim}`")
    lines.append("  Source location         :")
    lines.append(f"    {_extract_user_source(tb)}")
    if detail:
        lines.append(f"  Checkify report         : {detail}")
    diag = _diagnose(out_leaves)
    if diag:
        lines.append("  Contaminated outputs    :")
        lines.extend(diag)
    if prim is None and not diag:
        lines.append("  (NaN/Inf reached the output but could not be localized.)")
    return '\n'.join(lines), tb


def _raise_report(err, out_leaves, phase: str) -> None:
    """Raise a clean ``RuntimeError`` (eager) pointing the traceback at user code."""
    msg, tb = _build_message(err, out_leaves, phase)
    if tb is not None:
        try:
            ctx = source_info_util.user_context(tb)
        except Exception:
            ctx = None
        if ctx is not None:
            with ctx:
                raise RuntimeError(msg)
    raise RuntimeError(msg)


def _raise_report_cb(err, out_tree, phase: str) -> None:
    """Host callback used inside JIT: raise (JAX wraps it in JaxRuntimeError)."""
    msg, _ = _build_message(err, jax.tree.leaves(out_tree), phase)
    raise RuntimeError(msg)


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class DebugNan:
    """
    JIT-compatible NaN / Inf debugging utility.

    The (possibly stateful) target function is functionalised through
    :class:`StatefulFunction` and run under :func:`jax.experimental.checkify`.
    An error is raised only when NaN/Inf reaches the function's observable
    outputs (return value or updated state), so NaN that is computed but then
    masked away (e.g. ``jnp.where`` guards, ``vmap``-ed ``cond``) does not
    trigger a false alarm.

    Parameters
    ----------
    fn : Callable
        The function to debug.  May read and write :class:`~brainstate.State`
        objects.
    *args
        Example arguments used to trace the function.  Shapes and dtypes matter;
        the values are used for the actual debug run.
    phase : str, optional
        Label prepended to the error message (e.g. ``"forward"``).

    See Also
    --------
    debug_nan : Functional wrapper that runs the check unconditionally.
    debug_nan_if : Functional wrapper that runs the check conditionally.

    Notes
    -----
    ``checkify`` localizes NaN and division-by-zero to the producing primitive
    with an IDE-clickable source location.  Other Infs (overflow, ``log(0)``)
    are detected at the output and reported with diagnostics but without a
    primitive-level source.

    The function is executed once for tracing and once for the instrumented
    run, so functions with side effects (including global RNG that is not
    routed through :mod:`brainstate.random`) may behave differently from a
    single plain call.
    """

    def __init__(self, fn: Callable, *args, phase: str = ''):
        self.fn = fn
        self.phase = phase
        self._args = args

        self._stateful_fn = StatefulFunction(fn)
        # Compile once to capture the jaxpr and the accessed State objects.
        self._cache_key = self._stateful_fn.get_arg_cache_key(*args, compile_if_miss=True)
        self._states = self._stateful_fn.get_states_by_cache(self._cache_key)
        # checkify the functional form once; reused by check() / check_if().
        self._checked = checkify.checkify(self._pure, errors=_FLOAT_CHECKS)

    def _pure(self, state_vals, *args):
        """Pure ``(state_vals, *args) -> (out, new_state_vals)`` form of ``fn``."""
        new_state_vals, out = self._stateful_fn.jaxpr_call(state_vals, *args)
        return out, new_state_vals

    def _state_vals(self) -> list:
        """Current values of the accessed states (re-read fresh each call)."""
        return [s.value for s in self._states]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self):
        """
        Run the instrumented function and raise if NaN/Inf reaches an output.

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            If NaN or Inf contaminates the function's outputs or updated state.
            Under an enclosing trace (e.g. ``jit``) the error is raised from an
            ordered ``jax.debug.callback`` at run time instead.
        """
        err, out_tree = self._checked(self._state_vals(), *self._args)
        out_leaves = jax.tree.leaves(out_tree)
        flag = unvmap(_has_nan_flag(out_leaves), op='any')
        if isinstance(flag, Tracer):
            # inside an enclosing trace (e.g. jit) the flag cannot be
            # concretized -- raise from an ordered callback at run time instead
            jax.lax.cond(
                flag,
                lambda: jax.debug.callback(
                    functools.partial(_raise_report_cb, phase=self.phase),
                    err, out_tree, ordered=True,
                ),
                lambda: None,
            )
            return None
        if bool(flag):
            _raise_report(err, out_leaves, self.phase)
        return None

    def check_if(self, has_nan):
        """
        Conditionally run the instrumented function only when *has_nan* is True.

        Parameters
        ----------
        has_nan : bool or jax.Array
            Scalar boolean condition.  Batched / vmapped arrays are collapsed
            with ``unvmap(..., op='any')``.

        Returns
        -------
        None

        Notes
        -----
        * **Eager mode** (concrete predicate): runs the clean store-free raise
          so the ``RuntimeError`` points directly at user code.
        * **JIT-traced mode** (predicate is a :class:`Tracer`): uses
          ``jax.lax.cond`` and raises inside an ordered ``jax.debug.callback``;
          the error surfaces as a ``JaxRuntimeError`` wrapping the message.
        """
        pred = unvmap(has_nan, op='any')

        if isinstance(pred, Tracer):
            def _do_check():
                err, out_tree = self._checked(self._state_vals(), *self._args)
                outputs_bad = _has_nan_flag(jax.tree.leaves(out_tree))
                jax.lax.cond(
                    outputs_bad,
                    lambda: jax.debug.callback(
                        functools.partial(_raise_report_cb, phase=self.phase),
                        err, out_tree, ordered=True,
                    ),
                    lambda: None,
                )
                return None

            jax.lax.cond(pred, _do_check, lambda: None)
        else:
            if bool(pred):
                self.check()
        return None


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def debug_nan(fn: Callable, *args, phase: str = '') -> None:
    """
    Run *fn* with NaN / Inf detection (JIT-compatible).

    An error is raised only when NaN/Inf reaches *fn*'s observable outputs, so
    NaN that is computed but masked away does not trigger a false alarm.

    Parameters
    ----------
    fn : Callable
        The function to debug.
    *args
        Arguments to pass to the function.
    phase : str, optional
        Label prepended to the error message.

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        If NaN or Inf contaminates *fn*'s outputs or updated state.

    See Also
    --------
    debug_nan_if : Conditional variant.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import brainstate
        >>> brainstate.transform.debug_nan(lambda x: jnp.log(x), jnp.array([-1.0, 1.0]))
        Traceback (most recent call last):
            ...
        RuntimeError: NaN/Inf detected ...
    """
    DebugNan(fn, *args, phase=phase).check()


def debug_nan_if(has_nan: ArrayLike, fn: Callable, *args, phase: str = '') -> None:
    """
    Conditionally run *fn* with NaN / Inf detection.

    Equivalent to::

        if has_nan:
            debug_nan(fn, *args, phase=phase)

    but JIT-compatible via ``jax.lax.cond``.

    Parameters
    ----------
    has_nan : bool or jax.Array
        Condition to trigger debugging.  Batched arrays are collapsed with
        ``unvmap(..., op='any')``.
    fn : Callable
        The function to debug.
    *args
        Arguments to pass to the function.
    phase : str, optional
        Label prepended to the error message.

    Returns
    -------
    None

    See Also
    --------
    debug_nan : Unconditional variant.
    """
    DebugNan(fn, *args, phase=phase).check_if(has_nan)


def breakpoint_if(pred: ArrayLike, **breakpoint_kwargs) -> Any:
    """
    As ``jax.debug.breakpoint``, but only triggers if *pred* is True.

    Parameters
    ----------
    pred : bool or jax.Array
        Predicate for whether to trigger the breakpoint.
    **breakpoint_kwargs
        Forwarded to ``jax.debug.breakpoint``.
    """
    token = breakpoint_kwargs.get("token", None)
    return cond(
        unvmap(pred, op='any'),
        lambda: jax.debug.breakpoint(**breakpoint_kwargs),
        lambda: token,
    )
