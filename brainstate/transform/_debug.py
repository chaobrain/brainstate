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

import threading
import traceback as tb_module
from typing import Callable, Dict, List

import jax
import jax.numpy as jnp
from jax.extend import source_info_util

from brainstate._compatible_import import DropVar, Literal, ClosedJaxpr, is_jit_primitive
from ._conditions import cond
from ._make_jaxpr import StatefulFunction
from ._unvmap import unvmap

# ---------------------------------------------------------------------------
# Thread-local NaN detection store
#
# Callbacks must NOT raise inside jax.debug.callback because JAX wraps any
# exception in JaxRuntimeError("INTERNAL: CpuCallback error …"), burying the
# user-facing message.  Instead the callback stores (message, raw_source_info)
# here; _interpret_jaxpr_with_nan_check checks after every run and raises via
# source_info_util.user_context — the same mechanism used by
# State.raise_error_with_source_info — so the exception traceback points
# directly at the user code that introduced NaN.
# ---------------------------------------------------------------------------

_nan_store = threading.local()


def _nan_store_get() -> List[tuple]:
    """Each entry is (formatted_message: str, raw_source_info | None)."""
    if not hasattr(_nan_store, 'records'):
        _nan_store.records = []
    return _nan_store.records

__all__ = [
    'breakpoint_if',
    'debug_nan',
    'debug_nan_if',
]

# ---------------------------------------------------------------------------
# Source info: IDE-clickable location strings
# ---------------------------------------------------------------------------

def _extract_user_source(source_info) -> str:
    """
    Extract a human-readable, IDE-clickable source location from a jaxpr
    equation's source_info.

    Filters out JAX internal frames and returns the innermost user frame in
    standard Python traceback format:  ``File "path", line N, in func_name``
    which VSCode / PyCharm recognise as a hyperlink.
    """
    if source_info is None:
        return "<unknown source>"
    tb = getattr(source_info, 'traceback', None)
    if tb is None:
        return "<unknown source>"
    try:
        py_tb = tb.as_python_traceback()
        lines = tb_module.format_tb(py_tb)
        user_lines = [
            line for line in lines
            if '/site-packages/' not in line and 'jax/_src/' not in line
        ]
        if user_lines:
            # Last user frame is the innermost call site — format it cleanly.
            return user_lines[-1].strip()
    except Exception:
        pass
    # Fallback to JAX's own summarise helper
    try:
        return source_info_util.summarize(source_info)
    except Exception:
        return "<unknown source>"


# ---------------------------------------------------------------------------
# Jaxpr variable formatting (shared helpers)
# ---------------------------------------------------------------------------

def _build_var_names(jaxpr) -> Dict[int, str]:
    var_names: Dict[int, str] = {}
    counter = 0
    for var in jaxpr.invars:
        var_names[id(var)] = f"v{counter}"
        counter += 1
    for eqn in jaxpr.eqns:
        for var in eqn.outvars:
            if not isinstance(var, DropVar):
                var_names[id(var)] = f"v{counter}"
                counter += 1
    return var_names


def _format_var(var, var_names: Dict[int, str]) -> str:
    if isinstance(var, Literal):
        val = var.val
        if hasattr(val, 'item') and val.ndim == 0:
            val = val.item()
        return str(val)
    name = var_names.get(id(var))
    if name is None:
        return f"const:{var.aval.str_short()}"
    return f"{name}:{var.aval.str_short()}"


def _format_eqn(eqn, var_names: Dict[int, str]) -> str:
    outvars = ' '.join(_format_var(v, var_names) for v in eqn.outvars)
    invars = ' '.join(_format_var(v, var_names) for v in eqn.invars)
    return f"{outvars} = {eqn.primitive.name} {invars}"


# ---------------------------------------------------------------------------
# On-device NaN / Inf detection helpers (run inside JIT on the device)
# ---------------------------------------------------------------------------

def _is_float_array(x) -> bool:
    return hasattr(x, 'dtype') and jnp.issubdtype(x.dtype, jnp.floating)


def _has_nan_flag(vals) -> jax.Array:
    """Return a scalar bool JAX array: True iff any float value has NaN or Inf."""
    flags = [
        jnp.any(jnp.isnan(v) | jnp.isinf(v))
        for v in vals if _is_float_array(v)
    ]
    if not flags:
        return jnp.array(False)
    return jnp.any(jnp.stack(flags))


# ---------------------------------------------------------------------------
# NaN callback factory
# ---------------------------------------------------------------------------

def _format_nan_message(eqn_idx: int, total_eqns: int, prim_name: str,
                        eqn_str: str, source_loc: str, phase: str,
                        float_vals) -> str:
    phase_tag = f" [{phase}]" if phase else ""
    lines = [
        f"NaN/Inf detected{phase_tag}!",
        f"  Introduced by primitive : `{prim_name}`",
        f"  Source location         :",
        f"    {source_loc}",
        f"  Equation [{eqn_idx + 1}/{total_eqns}]: {eqn_str}",
    ]
    if float_vals:
        lines.append("  Float input values:")
        for i, v in enumerate(float_vals):
            if hasattr(v, 'size') and v.size <= 8:
                lines.append(f"    input[{i}] = {v}")
            elif hasattr(v, 'shape'):
                nan_cnt = int(jnp.sum(jnp.isnan(v)))
                inf_cnt = int(jnp.sum(jnp.isinf(v)))
                lines.append(
                    f"    input[{i}]: shape={v.shape}  "
                    f"min={float(jnp.min(v)):.4g}  max={float(jnp.max(v)):.4g}  "
                    f"NaNs={nan_cnt}  Infs={inf_cnt}"
                )
    return '\n'.join(lines)


def _make_nan_callback(eqn_idx: int, total_eqns: int, prim_name: str,
                       eqn_str: str, source_loc: str, raw_source_info,
                       phase: str, raise_in_callback: bool = False):
    """
    Return a host callback that reports a NaN detection.

    Two modes:
    - ``raise_in_callback=False`` (default, used outside JIT):
      Stores ``(message, raw_source_info)`` in thread-local storage.
      ``_raise_if_nan_detected`` then raises a **clean** ``RuntimeError``
      via ``source_info_util.user_context``, pointing the traceback at the
      user code — same mechanism as ``State.raise_error_with_source_info``.

    - ``raise_in_callback=True`` (used inside JIT / ``jax.lax.cond`` with
      traced predicate):
      Raises ``RuntimeError`` directly so JAX can propagate it.  The error
      will be wrapped in ``JaxRuntimeError`` but the full message (with source
      info) is still visible in ``str(exc)``.
    """
    def _report(*float_vals):
        msg = _format_nan_message(
            eqn_idx, total_eqns, prim_name, eqn_str, source_loc, phase, float_vals
        )
        if raise_in_callback:
            raise RuntimeError(msg)
        else:
            _nan_store_get().append((msg, raw_source_info))

    return _report


# ---------------------------------------------------------------------------
# Core: instrumented jaxpr interpreter (all ops stay on device)
# ---------------------------------------------------------------------------

def _get_invals(eqn, env):
    return [env[v] if not isinstance(v, Literal) else v.val for v in eqn.invars]


def _store_outvals(eqn, outvals, env):
    for var, val in zip(eqn.outvars, outvals):
        if not isinstance(var, DropVar):
            env[var] = val


def _raise_if_nan_detected(store_snapshot: int) -> None:
    """
    After an instrumented run, check the thread-local store for new NaN records.

    Uses ``source_info_util.user_context`` — the same mechanism as
    ``State.raise_error_with_source_info`` — so that JAX's traceback filtering
    shows the *user code* that introduced the NaN rather than library internals.
    """
    records = _nan_store_get()
    new = records[store_snapshot:]
    if not new:
        return
    msg, raw_src = new[0]
    if raw_src is not None:
        tb = getattr(raw_src, 'traceback', None)
        name_stack = (
            source_info_util.current_name_stack()
            + getattr(raw_src, 'name_stack', source_info_util.NameStack())
        )
        with source_info_util.user_context(tb, name_stack=name_stack):
            raise RuntimeError(msg)
    raise RuntimeError(msg)


def _interpret_jaxpr_with_nan_check(
    jaxpr, consts, *flat_args, phase: str = '', raise_in_callback: bool = False
) -> list:
    """
    Replay *jaxpr* equation-by-equation using real JAX primitives (JIT-compatible).

    After each equation an on-device NaN flag is computed.  When a primitive
    *introduces* NaN (output has NaN but input did not), ``jax.debug.callback``
    appends a report to thread-local storage.  After the full run,
    ``_raise_if_nan_detected`` checks for new reports and raises a clean
    ``RuntimeError`` via ``source_info_util.user_context``, so the traceback
    points to the user's code — not to library internals.
    """
    var_names = _build_var_names(jaxpr)
    total_eqns = len(jaxpr.eqns)

    # Snapshot store length before this call so nested calls don't interfere
    store_snapshot = len(_nan_store_get())

    # Build per-equation static metadata at Python / compile time
    eqn_meta: List[dict] = []
    for i, eqn in enumerate(jaxpr.eqns):
        raw_src = getattr(eqn, 'source_info', None)
        eqn_meta.append({
            'idx': i,
            'total': total_eqns,
            'prim': eqn.primitive.name,
            'eqn_str': _format_eqn(eqn, var_names),
            'source_loc': _extract_user_source(raw_src),
            'raw_src': raw_src,
        })

    # Set up environment
    env: dict = {}
    for var, val in zip(jaxpr.constvars, consts):
        env[var] = val
    for var, val in zip(jaxpr.invars, flat_args):
        env[var] = val

    for eqn, meta in zip(jaxpr.eqns, eqn_meta):
        invals = _get_invals(eqn, env)

        # On-device: does the input already carry NaN?
        input_has_nan = _has_nan_flag(invals)

        # Execute the primitive (with recursive instrumentation for nested ones)
        outvals = _execute_eqn(eqn, invals, phase, raise_in_callback)

        # On-device: did this op introduce new NaN?
        output_has_nan = _has_nan_flag(outvals)
        nan_introduced = output_has_nan & ~input_has_nan

        # Collect float inputs to pass to the callback
        float_invals = [v for v in invals if _is_float_array(v)]

        # Build callback closed over static metadata
        cb = _make_nan_callback(
            meta['idx'], meta['total'], meta['prim'],
            meta['eqn_str'], meta['source_loc'], meta['raw_src'], phase,
            raise_in_callback=raise_in_callback,
        )

        # Conditionally fire the callback (both branches return None)
        if float_invals:
            captured = tuple(float_invals)

            def _do_report(vals=captured, fn=cb):
                jax.debug.callback(fn, *vals, ordered=True)

            jax.lax.cond(nan_introduced, _do_report, lambda: None)
        else:
            jax.lax.cond(
                nan_introduced,
                lambda fn=cb: jax.debug.callback(fn, ordered=True),
                lambda: None,
            )

        _store_outvals(eqn, outvals, env)

    return [env[v] if not isinstance(v, Literal) else v.val for v in jaxpr.outvars]


# ---------------------------------------------------------------------------
# Primitive dispatch helpers
# ---------------------------------------------------------------------------

def _execute_eqn(eqn, invals, phase: str, raise_in_callback: bool = False) -> list:
    """Dispatch to specialised handlers for nested high-level primitives."""
    if is_jit_primitive(eqn):
        return _execute_jit_eqn(eqn, invals, phase, raise_in_callback)
    name = eqn.primitive.name
    if name == 'cond':
        return _execute_cond_eqn(eqn, invals, phase, raise_in_callback)
    if name == 'while':
        return _execute_while_eqn(eqn, invals, phase, raise_in_callback)
    if name == 'scan':
        return _execute_scan_eqn(eqn, invals, phase, raise_in_callback)
    # Plain primitive: execute normally
    subfuns, bind_params = eqn.primitive.get_bind_params(eqn.params)
    outvals = eqn.primitive.bind(*subfuns, *invals, **bind_params)
    if not eqn.primitive.multiple_results:
        outvals = [outvals]
    return outvals


def _extract_closed_jaxpr(obj):
    """Return (inner_jaxpr, inner_consts) from a ClosedJaxpr or bare Jaxpr."""
    if isinstance(obj, ClosedJaxpr):
        return obj.jaxpr, obj.consts
    return obj, ()


def _execute_jit_eqn(eqn, invals, phase: str, raise_in_callback: bool = False) -> list:
    """Replace a pjit/jit call with its recursively-instrumented inner jaxpr."""
    call_jaxpr = eqn.params.get('jaxpr') or eqn.params.get('call_jaxpr')
    if call_jaxpr is None:
        subfuns, bind_params = eqn.primitive.get_bind_params(eqn.params)
        outvals = eqn.primitive.bind(*subfuns, *invals, **bind_params)
        return [outvals] if not eqn.primitive.multiple_results else outvals
    inner_jaxpr, inner_consts = _extract_closed_jaxpr(call_jaxpr)
    return _interpret_jaxpr_with_nan_check(
        inner_jaxpr, inner_consts, *invals, phase=phase, raise_in_callback=raise_in_callback
    )


def _execute_cond_eqn(eqn, invals, phase: str, raise_in_callback: bool = False) -> list:
    """
    Replace a ``cond`` call with instrumented branches.

    JAX's cond jaxpr stores a sequence of ``ClosedJaxpr`` branches in
    ``eqn.params['branches']``.  The first inval is the integer index and
    the rest are the shared operands passed to every branch.
    """
    branches = eqn.params.get('branches')
    if branches is None:
        subfuns, bind_params = eqn.primitive.get_bind_params(eqn.params)
        outvals = eqn.primitive.bind(*subfuns, *invals, **bind_params)
        return [outvals] if not eqn.primitive.multiple_results else outvals

    index = invals[0]
    operands = tuple(invals[1:])

    def make_branch_fn(closed_jaxpr):
        b_jaxpr, b_consts = _extract_closed_jaxpr(closed_jaxpr)

        def branch_fn(*ops):
            result = _interpret_jaxpr_with_nan_check(
                b_jaxpr, b_consts, *ops, phase=phase, raise_in_callback=raise_in_callback
            )
            return tuple(result)

        return branch_fn

    branch_fns = [make_branch_fn(b) for b in branches]
    result = jax.lax.switch(index, branch_fns, *operands)
    return list(result) if isinstance(result, tuple) else [result]


def _execute_while_eqn(eqn, invals, phase: str, raise_in_callback: bool = False) -> list:
    """
    Replace a ``while`` call with an instrumented while loop.

    The body and condition jaxprs are instrumented so NaN checking occurs on
    every iteration without any CPU fallback.
    """
    cond_jaxpr_param = eqn.params.get('cond_jaxpr')
    body_jaxpr_param = eqn.params.get('body_jaxpr')
    if cond_jaxpr_param is None or body_jaxpr_param is None:
        subfuns, bind_params = eqn.primitive.get_bind_params(eqn.params)
        outvals = eqn.primitive.bind(*subfuns, *invals, **bind_params)
        return [outvals] if not eqn.primitive.multiple_results else outvals

    cond_jaxpr, cond_consts = _extract_closed_jaxpr(cond_jaxpr_param)
    body_jaxpr, body_consts = _extract_closed_jaxpr(body_jaxpr_param)

    def cond_fn(carry):
        result = _interpret_jaxpr_with_nan_check(
            cond_jaxpr, cond_consts, *carry, phase=phase, raise_in_callback=raise_in_callback
        )
        return result[0]  # single bool output

    def body_fn(carry):
        result = _interpret_jaxpr_with_nan_check(
            body_jaxpr, body_consts, *carry, phase=phase, raise_in_callback=raise_in_callback
        )
        return tuple(result)

    init_carry = tuple(invals)
    final_carry = jax.lax.while_loop(cond_fn, body_fn, init_carry)
    return list(final_carry) if isinstance(final_carry, tuple) else [final_carry]


def _execute_scan_eqn(eqn, invals, phase: str, raise_in_callback: bool = False) -> list:
    """
    Replace a ``scan`` call with an instrumented scan.

    The scan body jaxpr is instrumented so each iteration's NaN introduction
    is detected on the device.  The overall scan still runs natively via
    ``jax.lax.scan`` — no manual Python-level unrolling needed.
    """
    scan_jaxpr_param = eqn.params.get('jaxpr')
    if scan_jaxpr_param is None:
        subfuns, bind_params = eqn.primitive.get_bind_params(eqn.params)
        outvals = eqn.primitive.bind(*subfuns, *invals, **bind_params)
        return [outvals] if not eqn.primitive.multiple_results else outvals

    num_consts = eqn.params.get('num_consts', 0)
    num_carry = eqn.params.get('num_carry', 0)
    length = eqn.params.get('length')
    reverse = eqn.params.get('reverse', False)
    unroll = eqn.params.get('unroll', 1)

    inner_jaxpr, inner_consts_param = _extract_closed_jaxpr(scan_jaxpr_param)

    # Split outer invals: consts_from_outer | carry | xs
    outer_consts = tuple(invals[:num_consts])
    carry_init = tuple(invals[num_consts:num_consts + num_carry])
    xs_list = invals[num_consts + num_carry:]

    def scan_body(carry, x):
        # Reconstruct full argument list for the inner jaxpr:
        # inner_consts_param (closed) + outer_consts + carry + x_slices
        if len(xs_list) == 0:
            x_slices = ()
        elif len(xs_list) == 1:
            x_slices = (x,)
        else:
            x_slices = tuple(x)  # JAX passes a tuple when multiple xs

        all_inputs = list(outer_consts) + list(carry) + list(x_slices)
        outputs = _interpret_jaxpr_with_nan_check(
            inner_jaxpr, inner_consts_param, *all_inputs,
            phase=phase, raise_in_callback=raise_in_callback
        )
        new_carry = tuple(outputs[:num_carry])
        ys = tuple(outputs[num_carry:])
        return new_carry, ys

    if len(xs_list) == 0:
        xs_arg = None
    elif len(xs_list) == 1:
        xs_arg = xs_list[0]
    else:
        xs_arg = tuple(xs_list)

    final_carry, stacked_ys = jax.lax.scan(
        scan_body, carry_init, xs_arg,
        length=length, reverse=reverse, unroll=unroll,
    )

    carry_out = list(final_carry) if isinstance(final_carry, tuple) else [final_carry]

    num_ys = len(inner_jaxpr.outvars) - num_carry
    if num_ys == 0:
        ys_out = []
    elif num_ys == 1:
        ys_out = [stacked_ys]
    else:
        ys_out = list(stacked_ys) if isinstance(stacked_ys, tuple) else [stacked_ys]

    return carry_out + ys_out


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class DebugNan:
    """
    JIT-compatible NaN / Inf debugging utility.

    Instead of moving computation to the CPU when NaN is detected, this class
    instruments the function's jaxpr so that NaN checks run **on the device**
    (inside JIT) alongside the regular computation.  A lightweight
    ``jax.debug.callback`` fires only when NaN is actually introduced,
    reporting:

    * The primitive that introduced the NaN.
    * The **IDE-clickable** source location (``File "...", line N``).
    * The equation string from the jaxpr.
    * The float input values (or their statistics for large arrays).

    Parameters
    ----------
    fn : Callable
        The stateful function to debug.
    *args
        Example arguments used to trace the jaxpr (shapes / dtypes matter;
        values are used for the actual debug run).
    phase : str, optional
        Label prepended to the error message (e.g. ``"forward"``).
    """

    def __init__(self, fn: Callable, *args, phase: str = ''):
        self.fn = fn
        self.phase = phase

        self._stateful_fn = StatefulFunction(fn)
        # Compile once to get the jaxpr and the list of State objects accessed.
        cache_key = self._stateful_fn.get_arg_cache_key(*args, compile_if_miss=True)
        closed_jaxpr: ClosedJaxpr = self._stateful_fn.get_jaxpr_by_cache(cache_key)
        self._jaxpr = closed_jaxpr.jaxpr
        self._consts = closed_jaxpr.consts
        # User-provided args flattened (no state values yet)
        self._flat_user_args: list = jax.tree.leaves(args)
        # Keep the State objects so we can read their *current* values at call
        # time — state may have been updated between __init__ and check().
        self._states = self._stateful_fn.get_states_by_cache(cache_key)

    def _build_flat_args(self) -> list:
        """Concatenate flat user args with current state values."""
        flat_state_vals: list = jax.tree.leaves([s.value for s in self._states])
        return self._flat_user_args + flat_state_vals

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self):
        """
        Unconditionally run the instrumented function (JIT-compatible).

        All NaN checks execute on the device; the host callback fires only
        when NaN is detected, and the RuntimeError is raised cleanly after the
        instrumented run completes (no JAX INTERNAL wrapping).
        """
        flat_args = self._build_flat_args()
        store_snapshot = len(_nan_store_get())
        _interpret_jaxpr_with_nan_check(
            self._jaxpr, self._consts, *flat_args, phase=self.phase
        )
        # Callbacks are synchronous in eager mode; raise in Python space so
        # the user sees a clean RuntimeError, not "INTERNAL: CpuCallback …".
        _raise_if_nan_detected(store_snapshot)
        return None

    def check_if(self, has_nan):
        """
        Conditionally run the instrumented function only when *has_nan* is True.

        Parameters
        ----------
        has_nan : bool or jax.Array
            Scalar boolean condition.  Supports vmapped / batched arrays via
            ``unvmap(..., op='any')``.

        Notes
        -----
        Two modes based on whether we are inside a JIT-compiled function:

        * **Eager mode** (pred is concrete): uses the clean store-based raise
          so the RuntimeError points directly at user code.
        * **JIT-traced mode** (pred is a JAX Tracer): uses ``jax.lax.cond``
          with ``raise_in_callback=True`` so the callback raises inside XLA;
          the error surfaces as a JaxRuntimeError wrapping our message.
        """
        flat_args = self._build_flat_args()
        pred = unvmap(has_nan, op='any')

        if isinstance(pred, jax.core.Tracer):
            # Inside JIT: Python code won't re-run at execution time, so we
            # must raise inside the device callback.
            def _do_check():
                _interpret_jaxpr_with_nan_check(
                    self._jaxpr, self._consts, *flat_args,
                    phase=self.phase, raise_in_callback=True,
                )
                return None

            jax.lax.cond(pred, _do_check, lambda: None)
        else:
            # Eager mode: cleaner error — raise after callbacks complete.
            if bool(pred):
                store_snapshot = len(_nan_store_get())
                _interpret_jaxpr_with_nan_check(
                    self._jaxpr, self._consts, *flat_args, phase=self.phase
                )
                _raise_if_nan_detected(store_snapshot)


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def debug_nan(fn: Callable, *args, phase: str = ''):
    """
    Run *fn* with on-device NaN / Inf detection (JIT-compatible).

    Parameters
    ----------
    fn : Callable
        The function to debug.
    *args
        Arguments to pass to the function.
    phase : str, optional
        Label prepended to the error message.

    Notes
    -----
    This function is fully JIT-compatible.  All NaN checks run on the device;
    no data is moved to the CPU unless NaN is actually detected.
    """
    DebugNan(fn, *args, phase=phase).check()


def debug_nan_if(has_nan, fn: Callable, *args, phase: str = ''):
    """
    Conditionally run *fn* with on-device NaN / Inf detection.

    Equivalent to::

        if has_nan:
            debug_nan(fn, *args, phase=phase)

    but JIT-compatible via ``jax.lax.cond``.

    Parameters
    ----------
    has_nan : bool or jax.Array
        Condition to trigger debugging.
    fn : Callable
        The function to debug.
    *args
        Arguments to pass to the function.
    phase : str, optional
        Label prepended to the error message.
    """
    DebugNan(fn, *args, phase=phase).check_if(has_nan)


def breakpoint_if(pred, **breakpoint_kwargs):
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
