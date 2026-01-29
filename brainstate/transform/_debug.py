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

import functools
from typing import Callable, Dict, Optional, Any, Tuple, List

import jax
import jax.numpy as jnp

from brainstate._compatible_import import DropVar, Literal, ClosedJaxpr, is_jit_primitive
from ._conditions import cond
from ._make_jaxpr import StatefulFunction
from ._unvmap import unvmap
from ._unvmap import unvmap_any

__all__ = [
    'breakpoint',
    'debug_nan',
    'debug_nan_if',
]


def breakpoint(pred, **breakpoint_kwargs):
    """As `jax.debug.breakpoint`, but only triggers if `pred` is True.

    **Arguments:**

    - `pred`: the predicate for whether to trigger the breakpoint.
    - `**breakpoint_kwargs`: any other keyword arguments to forward to `jax.debug.breakpoint`.

    """

    # We can't just write `jax.debug.breakpoint` for the second branch. For some reason
    # it needs as lambda wrapper.

    token = breakpoint_kwargs.get("token", None)
    return cond(
        unvmap_any(pred),
        lambda: jax.debug.breakpoint(**breakpoint_kwargs),
        lambda: token,
    )


def _check_for_nan(x) -> Tuple[bool, int, Optional[Any]]:
    """
    Check if an array contains NaN or Inf values.

    Parameters
    ----------
    x : array-like
        The array to check for NaN/Inf values.

    Returns
    -------
    tuple
        A tuple of (has_bad, bad_count, bad_indices) where:
        - has_bad: bool indicating if any NaN/Inf values exist
        - bad_count: number of NaN/Inf values found
        - bad_indices: indices of NaN/Inf values (None if none found)
    """
    if not hasattr(x, 'dtype'):
        return False, 0, None
    if not jnp.issubdtype(x.dtype, jnp.floating):
        return False, 0, None
    # Check for both NaN and Inf
    bad_mask = jnp.isnan(x) | jnp.isinf(x)
    has_bad = bool(jnp.any(bad_mask))
    if has_bad:
        bad_count = int(jnp.sum(bad_mask))
        # Handle scalar arrays (0d arrays)
        if x.ndim == 0:
            bad_indices = ()
        else:
            bad_indices = jnp.where(bad_mask)
        return True, bad_count, bad_indices
    return False, 0, None


def _check_pytree_for_nan(pytree, name: str = "") -> Tuple[bool, List[Dict]]:
    """
    Check an entire pytree for NaN values.

    Parameters
    ----------
    pytree : PyTree
        The pytree to check for NaN values.
    name : str, optional
        A name to identify the pytree in reports.

    Returns
    -------
    tuple
        A tuple of (has_nan, results) where:
        - has_nan: bool indicating if any NaN values exist in the pytree
        - results: list of dicts with details about each leaf containing NaN
    """
    results = []
    leaves = jax.tree.leaves(pytree)
    for i, leaf in enumerate(leaves):
        has_nan, count, indices = _check_for_nan(leaf)
        if has_nan:
            results.append(
                {
                    'leaf_index': i,
                    'nan_count': count,
                    'indices': indices,
                    'shape': getattr(leaf, 'shape', None),
                    'dtype': getattr(leaf, 'dtype', None),
                }
            )
    return len(results) > 0, results


def _is_expandable_primitive(eqn) -> bool:
    """
    Check if a primitive contains inner jaxpr(s) that should be expanded for NaN checking.

    Parameters
    ----------
    eqn : JaxprEqn
        The equation to check.

    Returns
    -------
    bool
        True if the primitive contains inner jaxpr(s) that can be expanded.
    """
    if is_jit_primitive(eqn):
        return True
    if eqn.primitive.name in ['cond', 'while', 'scan']:
        return True
    return False


def _eval_jit_primitive(eqn, invals) -> Tuple[List, List[Dict], List[str]]:
    """
    Evaluate a JIT primitive by recursively evaluating its inner jaxpr.

    Parameters
    ----------
    eqn : JaxprEqn
        The JIT equation to evaluate.
    invals : list
        Input values for the equation.

    Returns
    -------
    tuple
        A tuple of (outputs, nan_report, eqn_strs).
    """
    # Try different parameter names for the inner jaxpr
    # JAX uses 'jaxpr' for pjit, 'call_jaxpr' for some older primitives
    call_jaxpr = eqn.params.get('jaxpr') or eqn.params.get('call_jaxpr')
    if call_jaxpr is None:
        # Fallback: evaluate normally
        subfuns, bind_params = eqn.primitive.get_bind_params(eqn.params)
        outvals = eqn.primitive.bind(*subfuns, *invals, **bind_params)
        if not eqn.primitive.multiple_results:
            outvals = [outvals]
        return outvals, [], []

    # Extract jaxpr and consts
    if isinstance(call_jaxpr, ClosedJaxpr):
        inner_jaxpr = call_jaxpr.jaxpr
        inner_consts = call_jaxpr.consts
    else:
        inner_jaxpr = call_jaxpr
        inner_consts = ()

    # Recursively evaluate the inner jaxpr
    outputs, nan_report, eqn_strs = _eval_jaxpr_with_nan_check(
        inner_jaxpr, inner_consts, *invals
    )

    # Mark reports as coming from JIT
    for report in nan_report:
        report['inside_jit'] = True

    return outputs, nan_report, eqn_strs


def _eval_cond_primitive(eqn, invals) -> Tuple[List, List[Dict], List[str]]:
    """
    Evaluate a cond primitive by evaluating the taken branch.

    Parameters
    ----------
    eqn : JaxprEqn
        The cond equation to evaluate.
    invals : list
        Input values for the equation. First value is the predicate.

    Returns
    -------
    tuple
        A tuple of (outputs, nan_report, eqn_strs).
    """
    branches = eqn.params.get('branches')
    if branches is None:
        # Fallback: evaluate normally
        subfuns, bind_params = eqn.primitive.get_bind_params(eqn.params)
        outvals = eqn.primitive.bind(*subfuns, *invals, **bind_params)
        if not eqn.primitive.multiple_results:
            outvals = [outvals]
        return outvals, [], []

    # First input is the predicate (index), rest are operands
    pred_idx = int(invals[0])
    operands = invals[1:]

    # Select the branch based on predicate
    branch_jaxpr = branches[pred_idx]
    if isinstance(branch_jaxpr, ClosedJaxpr):
        inner_jaxpr = branch_jaxpr.jaxpr
        inner_consts = branch_jaxpr.consts
    else:
        inner_jaxpr = branch_jaxpr
        inner_consts = ()

    # Recursively evaluate the selected branch
    outputs, nan_report, eqn_strs = _eval_jaxpr_with_nan_check(
        inner_jaxpr, inner_consts, *operands
    )

    # Mark reports as coming from cond
    for report in nan_report:
        report['inside_cond'] = True
        report['branch_index'] = pred_idx

    return outputs, nan_report, eqn_strs


def _eval_expanded_primitive(eqn, invals) -> Tuple[List, List[Dict], List[str]]:
    """
    Evaluate a high-level primitive by recursively evaluating its inner jaxpr.

    Parameters
    ----------
    eqn : JaxprEqn
        The equation to evaluate.
    invals : list
        Input values for the equation.

    Returns
    -------
    tuple
        A tuple of (outputs, nan_report, eqn_strs).
    """
    if is_jit_primitive(eqn):
        return _eval_jit_primitive(eqn, invals)
    elif eqn.primitive.name == 'cond':
        return _eval_cond_primitive(eqn, invals)
    # For while and scan, fall back to normal evaluation for now
    # (they require more complex handling with iteration)
    else:
        subfuns, bind_params = eqn.primitive.get_bind_params(eqn.params)
        outvals = eqn.primitive.bind(*subfuns, *invals, **bind_params)
        if not eqn.primitive.multiple_results:
            outvals = [outvals]
        return outvals, [], []


def _eval_jaxpr_with_nan_check(jaxpr, consts, *args) -> Tuple[List, List[Dict], List[str]]:
    """
    Evaluate a jaxpr equation by equation, checking for NaN after each operation.

    This function implements a custom jaxpr interpreter that evaluates each
    primitive operation and checks if NaN values are introduced in the outputs.

    Parameters
    ----------
    jaxpr : Jaxpr
        The jaxpr to evaluate.
    consts : sequence
        The constant values for the jaxpr.
    *args
        The input arguments for the jaxpr.

    Returns
    -------
    tuple
        A tuple of (outputs, nan_report, all_eqn_strs) where:
        - outputs: list of output values from the jaxpr evaluation
        - nan_report: list of dicts with NaN detection info for each equation
          that first introduced NaN values
        - all_eqn_strs: list of equation strings for all equations
    """
    env = {}
    nan_report = []
    all_eqn_strs = []  # Collect all equation strings for context display

    # Bind constants to their variables
    for var, val in zip(jaxpr.constvars, consts):
        env[var] = val

    # Bind input arguments to their variables
    for var, val in zip(jaxpr.invars, args):
        env[var] = val

    # Evaluate each equation
    for eqn_idx, eqn in enumerate(jaxpr.eqns):
        all_eqn_strs.append(str(eqn))  # Store equation string for context
        # Get input values for this equation
        invals = [env[v] if not isinstance(v, Literal) else v.val for v in eqn.invars]

        # Check inputs for NaN (to track propagation vs. introduction)
        input_has_nan, _ = _check_pytree_for_nan(invals, f"eqn_{eqn_idx}_inputs")

        # Check if this is an expandable primitive (jit, cond, etc.)
        if _is_expandable_primitive(eqn):
            # Recursively evaluate inner jaxpr
            outvals, inner_nan_report, inner_eqn_strs = _eval_expanded_primitive(eqn, invals)

            # Add inner NaN reports with adjusted indices
            for report in inner_nan_report:
                report['outer_eqn_index'] = eqn_idx
                report['outer_primitive'] = eqn.primitive.name
            nan_report.extend(inner_nan_report)

            # Add inner equation strings for context (indented)
            all_eqn_strs.extend([f"  [inner] {s}" for s in inner_eqn_strs])
        else:
            # Evaluate the primitive normally
            subfuns, bind_params = eqn.primitive.get_bind_params(eqn.params)
            outvals = eqn.primitive.bind(*subfuns, *invals, **bind_params)
            if not eqn.primitive.multiple_results:
                outvals = [outvals]

        # Check outputs for NaN
        output_has_nan, output_nan_details = _check_pytree_for_nan(outvals, f"eqn_{eqn_idx}_outputs")

        # If NaN appeared in output but wasn't in input, record it
        # (Skip for expandable primitives as NaN is already reported from inner)
        if output_has_nan and not input_has_nan and not _is_expandable_primitive(eqn):
            nan_report.append({
                'eqn_index': eqn_idx,
                'primitive': eqn.primitive.name,
                'input_shapes': [getattr(v, 'shape', None) for v in invals],
                'output_shapes': [getattr(v, 'shape', None) for v in outvals],
                'input_values': invals,  # Include actual input values for debugging
                'nan_details': output_nan_details,
                'equation_str': str(eqn),
                'source_info': getattr(eqn, 'source_info', None),
            })

        # Store outputs in environment
        for var, val in zip(eqn.outvars, outvals):
            if not isinstance(var, DropVar):
                env[var] = val

    # Get final outputs
    outputs = [env[v] if not isinstance(v, Literal) else v.val for v in jaxpr.outvars]
    return outputs, nan_report, all_eqn_strs


def _format_nan_report(
    nan_report: List[Dict],
    total_eqns: int,
    all_eqn_strs: Optional[List[str]] = None,
    context_window: int = 5
) -> str:
    """
    Format a NaN/Inf report into a human-readable string with context window.

    Parameters
    ----------
    nan_report : list
        List of dicts containing NaN/Inf detection information.
    total_eqns : int
        Total number of equations that were evaluated.
    all_eqn_strs : list of str, optional
        List of all equation strings for context display.
    context_window : int, default 5
        Number of equations to show before and after each NaN/Inf source.

    Returns
    -------
    str
        A formatted string describing where NaN/Inf values were detected.
    """
    if not nan_report:
        return f"No NaN/Inf detected in {total_eqns} equations."

    lines = [f"NaN/Inf detected! Found in {len(nan_report)} equation(s) out of {total_eqns}:"]

    for info in nan_report:
        eqn_idx = info['eqn_index']
        phase = info.get('phase', 'unknown')

        # Build context header with nesting info
        header_parts = [f"[{phase.upper()}]"]
        if 'outer_primitive' in info:
            header_parts.append(f"inside {info['outer_primitive']}")
        if 'inside_jit' in info:
            header_parts.append("(JIT expanded)")
        if 'inside_cond' in info:
            header_parts.append(f"(cond branch {info.get('branch_index', '?')})")

        header = " ".join(header_parts)
        lines.append(f"\n=== {header} Context around Equation {eqn_idx} ({info['primitive']}) ===")

        # Show context window if equation strings are available
        if all_eqn_strs:
            start_idx = max(0, eqn_idx - context_window)
            end_idx = min(total_eqns, eqn_idx + context_window + 1)

            for i in range(start_idx, end_idx):
                marker = "  <-- NaN/Inf introduced here" if i == eqn_idx else ""
                lines.append(f"  Equation {i}: {all_eqn_strs[i]}{marker}")

        # Show details about the NaN/Inf source
        lines.append(f"\n  Primitive: {info['primitive']}")
        lines.append(f"  Input shapes: {info['input_shapes']}")
        lines.append(f"  Output shapes: {info['output_shapes']}")

        # Format source info nicely if available
        source_info = info.get('source_info')
        if source_info is not None:
            try:
                # Try to get traceback from source_info
                if hasattr(source_info, 'traceback') and source_info.traceback:
                    tb = source_info.traceback()
                    if tb:
                        lines.append(f"  Source: {tb[-1] if tb else 'unknown'}")
            except Exception:
                pass  # Skip if source info extraction fails

        # Show input values that led to NaN/Inf (truncated for large arrays)
        for i, (shape, val) in enumerate(zip(info['input_shapes'], info['input_values'])):
            if hasattr(val, 'size') and val.size <= 10:
                lines.append(f"  Input {i} value: {val}")
            elif hasattr(val, 'size'):
                lines.append(
                    f"  Input {i} value (truncated): "
                    f"shape={shape}, "
                    f"min={float(jnp.min(val)):.4g}, "
                    f"max={float(jnp.max(val)):.4g}"
                )

    return '\n'.join(lines)


# =============================================================================
# JIT-compatible NaN/Inf detection functions
# =============================================================================


def _nan_error_callback(name, jaxpr_info):
    """
    Callback to raise error with detailed NaN/Inf analysis.

    This function is called via jax.debug.callback when NaN/Inf is detected.
    If jaxpr_info is provided, it performs detailed equation-by-equation analysis.
    Otherwise, it reports basic information about which leaves have NaN/Inf.

    Parameters
    ----------
    name : str
        A descriptive name for the computation (e.g., "gradient computation").
    jaxpr_info : dict, optional
        Contains {'jaxpr': jaxpr, 'consts': consts, 'flat_args': flat_args}
        for detailed equation-by-equation analysis.
    """
    # Detailed jaxpr analysis
    jaxpr = jaxpr_info['jaxpr']
    consts = jaxpr_info['consts']
    flat_args = jaxpr_info['flat_args']

    outputs, nan_report, all_eqn_strs = _eval_jaxpr_with_nan_check(jaxpr, consts, *flat_args)

    if nan_report:
        for item in nan_report:
            item['phase'] = 'gradient'
        report = _format_nan_report(nan_report, len(jaxpr.eqns), all_eqn_strs)
        raise RuntimeError(f"NaN/Inf detected in {name}:\n"
                           f"{report}")
    raise RuntimeError(
        f'NaN/Inf detected in {name}, but detailed analysis found no specific source. '
        'This may indicate NaN/Inf existed in inputs or was introduced in a way '
        'not captured by equation-by-equation analysis.'
    )


def debug_nan(fn: Callable, *args):
    stateful_fn = StatefulFunction(fn)
    grad_jaxpr = stateful_fn.get_jaxpr(*args, compile_if_miss=True)

    # Create callback that receives flat_args as concrete values
    def error_callback(flat_args_concrete, consts):
        jaxpr_info = {
            'jaxpr': grad_jaxpr.jaxpr,  # Static, captured in closure
            'consts': consts,  # Concrete values from callback
            'flat_args': flat_args_concrete  # Concrete values from callback
        }
        _nan_error_callback("gradient computation", jaxpr_info)

    # Pack grads and flat_args together as operand
    # Flatten args for passing through the callback
    flat_args, _ = jax.tree.flatten(args)
    flat_args_unvmapped = jax.tree.map(functools.partial(unvmap, op='none'), flat_args)

    # Use jax.lax.cond - pass flat_args through so they become concrete in callback
    # This is compatible with jax.jit
    jax.debug.callback(error_callback, flat_args_unvmapped, grad_jaxpr.consts)


def debug_nan_if(has_nan: bool | jax.Array, fn: Callable, *args):
    jax.lax.cond(
        unvmap(has_nan, op='any'),
        lambda *args_: debug_nan(fn, *args_),
        lambda operand: None,
        *args,
    )
