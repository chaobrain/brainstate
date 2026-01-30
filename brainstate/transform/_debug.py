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


def _get_primitive_display_info(eqn, report: Dict = None) -> Dict:
    """
    Extract display information from a primitive for tree visualization.

    Parameters
    ----------
    eqn : JaxprEqn
        The equation to extract info from.
    report : dict, optional
        The NaN report dict containing additional metadata like iteration index.

    Returns
    -------
    dict
        A dict with 'type' (primitive type) and 'name' (display name).
    """
    prim_name = eqn.primitive.name
    info = {'type': prim_name, 'name': prim_name}

    if is_jit_primitive(eqn):
        # Extract function name from jit/pjit params
        func_name = eqn.params.get('name', 'jit')
        info['type'] = 'jit'
        info['name'] = func_name

    elif prim_name == 'scan':
        info['type'] = 'scan'
        if report and 'iteration_index' in report:
            info['name'] = f"iteration {report['iteration_index']}"
        else:
            info['name'] = 'body'

    elif prim_name == 'while':
        info['type'] = 'while'
        if report:
            part = report.get('while_part', 'body')
            iter_idx = report.get('iteration_index', '?')
            info['name'] = f"{part} (iter {iter_idx})"
        else:
            info['name'] = 'body'

    elif prim_name == 'cond':
        info['type'] = 'cond'
        if report and 'branch_index' in report:
            info['name'] = f"branch {report['branch_index']}"
        else:
            info['name'] = 'branch'

    return info


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


def _eval_while_primitive(eqn, invals, max_iterations: int = 100) -> Tuple[List, List[Dict], List[str]]:
    """
    Evaluate a while primitive by iterating and checking for NaN at each step.

    Parameters
    ----------
    eqn : JaxprEqn
        The while equation to evaluate.
    invals : list
        Input values for the equation (initial carry values).
    max_iterations : int
        Maximum number of iterations to evaluate (to avoid infinite loops).

    Returns
    -------
    tuple
        A tuple of (outputs, nan_report, eqn_strs).
    """
    cond_jaxpr = eqn.params.get('cond_jaxpr')
    body_jaxpr = eqn.params.get('body_jaxpr')

    if cond_jaxpr is None or body_jaxpr is None:
        # Fallback: evaluate normally
        subfuns, bind_params = eqn.primitive.get_bind_params(eqn.params)
        outvals = eqn.primitive.bind(*subfuns, *invals, **bind_params)
        if not eqn.primitive.multiple_results:
            outvals = [outvals]
        return outvals, [], []

    # Extract jaxprs and consts
    if isinstance(cond_jaxpr, ClosedJaxpr):
        cond_inner_jaxpr = cond_jaxpr.jaxpr
        cond_consts = cond_jaxpr.consts
    else:
        cond_inner_jaxpr = cond_jaxpr
        cond_consts = ()

    if isinstance(body_jaxpr, ClosedJaxpr):
        body_inner_jaxpr = body_jaxpr.jaxpr
        body_consts = body_jaxpr.consts
    else:
        body_inner_jaxpr = body_jaxpr
        body_consts = ()

    # Iterate and check for NaN
    all_nan_reports = []
    all_eqn_strs = []
    carry = list(invals)

    for iteration in range(max_iterations):
        # Evaluate condition
        cond_outputs, cond_nan_report, cond_eqn_strs = _eval_jaxpr_with_nan_check(
            cond_inner_jaxpr, cond_consts, *carry
        )

        # Tag cond reports
        for report in cond_nan_report:
            report['inside_while'] = True
            report['while_part'] = 'cond'
            report['iteration_index'] = iteration
        all_nan_reports.extend(cond_nan_report)
        all_eqn_strs.extend([f"  [while iter {iteration} cond] {s}" for s in cond_eqn_strs])

        # Check if we should continue
        cond_result = cond_outputs[0] if cond_outputs else False
        if not bool(cond_result):
            break

        # Evaluate body
        body_outputs, body_nan_report, body_eqn_strs = _eval_jaxpr_with_nan_check(
            body_inner_jaxpr, body_consts, *carry
        )

        # Tag body reports
        for report in body_nan_report:
            report['inside_while'] = True
            report['while_part'] = 'body'
            report['iteration_index'] = iteration
        all_nan_reports.extend(body_nan_report)
        all_eqn_strs.extend([f"  [while iter {iteration} body] {s}" for s in body_eqn_strs])

        # Update carry for next iteration
        carry = body_outputs

        # If NaN was detected, we can stop early
        if all_nan_reports:
            break

    return carry, all_nan_reports, all_eqn_strs


def _eval_scan_primitive(eqn, invals, max_iterations: int = 100) -> Tuple[List, List[Dict], List[str]]:
    """
    Evaluate a scan primitive by iterating over the axis and checking for NaN.

    Parameters
    ----------
    eqn : JaxprEqn
        The scan equation to evaluate.
    invals : list
        Input values for the equation (consts + carry + xs).
    max_iterations : int
        Maximum number of iterations to check (to limit debugging time).

    Returns
    -------
    tuple
        A tuple of (outputs, nan_report, eqn_strs).
    """
    scan_jaxpr = eqn.params.get('jaxpr')
    num_consts = eqn.params.get('num_consts', 0)
    num_carry = eqn.params.get('num_carry', 0)
    reverse = eqn.params.get('reverse', False)
    length = eqn.params.get('length')

    if scan_jaxpr is None:
        # Fallback: evaluate normally
        subfuns, bind_params = eqn.primitive.get_bind_params(eqn.params)
        outvals = eqn.primitive.bind(*subfuns, *invals, **bind_params)
        if not eqn.primitive.multiple_results:
            outvals = [outvals]
        return outvals, [], []

    # Extract jaxpr and consts
    if isinstance(scan_jaxpr, ClosedJaxpr):
        inner_jaxpr = scan_jaxpr.jaxpr
        inner_consts = scan_jaxpr.consts
    else:
        inner_jaxpr = scan_jaxpr
        inner_consts = ()

    # Split inputs: consts, carry, xs
    consts = list(invals[:num_consts])
    carry = list(invals[num_consts:num_consts + num_carry])
    xs = list(invals[num_consts + num_carry:])

    # Determine iteration length
    if length is None and xs:
        # Infer from xs shape
        length = xs[0].shape[0] if hasattr(xs[0], 'shape') and xs[0].ndim > 0 else 1
    elif length is None:
        length = 0

    # Limit iterations for debugging
    num_iters = min(length, max_iterations)

    # Iterate and check for NaN
    all_nan_reports = []
    all_eqn_strs = []
    num_ys = len(inner_jaxpr.outvars) - num_carry
    all_ys = [[] for _ in range(num_ys)]

    indices = range(num_iters)
    if reverse:
        indices = reversed(list(indices))

    for iteration in indices:
        # Slice xs for this iteration
        x_slices = [x[iteration] if hasattr(x, '__getitem__') and x.ndim > 0 else x for x in xs]

        # Build inputs: consts + carry + x_slices
        iter_inputs = consts + carry + x_slices

        # Evaluate the scan function
        outputs, nan_report, eqn_strs = _eval_jaxpr_with_nan_check(
            inner_jaxpr, inner_consts, *iter_inputs
        )

        # Tag reports
        for report in nan_report:
            report['inside_scan'] = True
            report['iteration_index'] = iteration
        all_nan_reports.extend(nan_report)
        all_eqn_strs.extend([f"  [scan iter {iteration}] {s}" for s in eqn_strs])

        # Split outputs into new carry and ys
        new_carry = outputs[:num_carry]
        ys = outputs[num_carry:]

        # Collect outputs
        for i, y in enumerate(ys):
            all_ys[i].append(y)

        # Update carry
        carry = new_carry

        # If NaN was detected, we can stop early
        if all_nan_reports:
            break

    # Stack ys along axis 0 (for iterations we've done)
    stacked_ys = []
    for y_list in all_ys:
        if y_list:
            stacked_ys.append(jnp.stack(y_list, axis=0))
        else:
            stacked_ys.append(jnp.array([]))

    # Final outputs: carry + stacked_ys
    final_outputs = carry + stacked_ys

    return final_outputs, all_nan_reports, all_eqn_strs


def _eval_primitive(eqn, invals) -> Tuple[List, List[Dict], List[str]]:
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
    elif eqn.primitive.name == 'while':
        return _eval_while_primitive(eqn, invals)
    elif eqn.primitive.name == 'scan':
        return _eval_scan_primitive(eqn, invals)
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

    # Collect all equation strings upfront for context display
    all_eqn_strs = [str(eqn) for eqn in jaxpr.eqns]

    # Bind constants to their variables
    for var, val in zip(jaxpr.constvars, consts):
        env[var] = val

    # Bind input arguments to their variables
    for var, val in zip(jaxpr.invars, args):
        env[var] = val

    # Evaluate each equation
    for eqn_idx, eqn in enumerate(jaxpr.eqns):
        # Get input values for this equation
        invals = [env[v] if not isinstance(v, Literal) else v.val for v in eqn.invars]

        # Check inputs for NaN (to track propagation vs. introduction)
        input_has_nan, _ = _check_pytree_for_nan(invals, f"eqn_{eqn_idx}_inputs")

        # Check if this is an expandable primitive (jit, cond, etc.)
        # Recursively evaluate inner jaxpr
        outvals, inner_nan_report, inner_eqn_strs = _eval_primitive(eqn, invals)

        if _is_expandable_primitive(eqn):
            # Add inner NaN reports with adjusted indices and nesting path
            for report in inner_nan_report:
                report['outer_eqn_index'] = eqn_idx
                report['outer_primitive'] = eqn.primitive.name
                # Get display info for this primitive
                display_info = _get_primitive_display_info(eqn, report)
                # Prepend this equation to the nesting path
                outer_entry = {
                    'eqn_index': eqn_idx,
                    'primitive': eqn.primitive.name,
                    'eqn_str': str(eqn),
                    'display_type': display_info['type'],
                    'display_name': display_info['name'],
                    'all_eqn_strs': all_eqn_strs.copy(),  # Store outer jaxpr equations
                    'total_eqns': len(jaxpr.eqns),
                }
                if 'nesting_path' not in report:
                    report['nesting_path'] = []
                report['nesting_path'].insert(0, outer_entry)
            nan_report.extend(inner_nan_report)

            # Add inner equation strings for context (indented)
            all_eqn_strs.extend([f"  [inner] {s}" for s in inner_eqn_strs])

        # Check outputs for NaN
        output_has_nan, output_nan_details = _check_pytree_for_nan(outvals, f"eqn_{eqn_idx}_outputs")

        # If NaN appeared in output but wasn't in input, record it
        # (Skip for expandable primitives as NaN is already reported from inner)
        if output_has_nan and not input_has_nan and not _is_expandable_primitive(eqn):
            nan_report.append(
                {
                    'eqn_index': eqn_idx,
                    'primitive': eqn.primitive.name,
                    'input_shapes': [getattr(v, 'shape', None) for v in invals],
                    'output_shapes': [getattr(v, 'shape', None) for v in outvals],
                    'input_values': invals,  # Include actual input values for debugging
                    'nan_details': output_nan_details,
                    'equation_str': str(eqn),
                    'source_info': getattr(eqn, 'source_info', None),
                    # Initialize nesting_path with this equation as the innermost
                    'nesting_path': [{
                        'eqn_index': eqn_idx,
                        'primitive': eqn.primitive.name,
                        'eqn_str': str(eqn),
                        'all_eqn_strs': all_eqn_strs.copy(),
                        'total_eqns': len(jaxpr.eqns),
                        'display_type': eqn.primitive.name,
                        'display_name': eqn.primitive.name,
                        'is_nan_source': True,  # Mark this as the actual NaN source
                    }],
                }
            )

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
    context: int = 5,
    depth: int = 1
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
    context : int, default 5
        Number of equations to show before and after each NaN/Inf source.
        Use -1 to show all equations.
    depth : int, default 1
        Number of nesting levels to display fully.
        - depth=1: Show only innermost jaxpr
        - depth=2: Show innermost + one outer jaxpr
        - depth=-1: Show all nesting levels

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
        if 'inside_while' in info:
            iter_idx = info.get('iteration_index', '?')
            while_part = info.get('while_part', 'body')
            header_parts.append(f"(while {while_part}, iteration {iter_idx})")
        if 'inside_scan' in info:
            iter_idx = info.get('iteration_index', '?')
            header_parts.append(f"(scan iteration {iter_idx})")

        header = " ".join(header_parts)
        lines.append(f"\n=== {header} Context around Equation {eqn_idx} ({info['primitive']}) ===")

        # Show unified primitive display based on depth parameter
        nesting_path = info.get('nesting_path', [])
        if nesting_path:
            lines.append("\n  Primitives:")

            total_levels = len(nesting_path)

            def get_first_line(eqn_str: str) -> str:
                """Get first line of equation, adding ... if truncated."""
                first_line = eqn_str.split('\n')[0]
                if '\n' in eqn_str:
                    # Multi-line equation, indicate truncation
                    return first_line + " ..."
                return first_line

            def is_level_within_depth(level_idx: int) -> bool:
                """Check if a level should show full equations based on depth."""
                if depth == -1:
                    return True
                # level_idx is 0-based from outermost
                # depth=1 means only innermost (level total_levels-1)
                # depth=2 means innermost + one outer (levels total_levels-2, total_levels-1)
                return (total_levels - level_idx) <= depth

            # Pass 1: Opening lines (outermost to innermost)
            # For each non-innermost level, show the nested call equation
            for level_idx in range(total_levels - 1):
                path_entry = nesting_path[level_idx]
                indent = "    " + "  " * level_idx
                nested_eqn_idx = path_entry['eqn_index']
                eqn_str = path_entry['eqn_str']

                # Show the nested call equation (opening)
                lines.append(f"{indent}[{nested_eqn_idx}] {get_first_line(eqn_str)}")

            # Pass 2: Innermost level content (context window)
            innermost_entry = nesting_path[total_levels - 1]
            innermost_indent = "    " + "  " * (total_levels - 1)
            inner_eqn_strs = innermost_entry.get('all_eqn_strs', [])
            inner_total = innermost_entry.get('total_eqns', len(inner_eqn_strs))
            inner_eqn_idx = innermost_entry['eqn_index']

            if inner_eqn_strs:
                # Determine range based on context parameter
                if context == -1:
                    ctx_start, ctx_end = 0, inner_total
                else:
                    ctx_start = max(0, inner_eqn_idx - context)
                    ctx_end = min(inner_total, inner_eqn_idx + context + 1)

                for i in range(ctx_start, ctx_end):
                    if i < len(inner_eqn_strs):
                        marker = "  <-- NaN/Inf introduced here" if i == inner_eqn_idx else ""
                        lines.append(f"{innermost_indent}[{i}] {get_first_line(inner_eqn_strs[i])}{marker}")

                # Show ellipsis if more equations after context
                if ctx_end < inner_total:
                    lines.append(f"{innermost_indent}...")
            else:
                # Fallback if no equation strings available
                eqn_str = innermost_entry['eqn_str']
                lines.append(
                    f"{innermost_indent}[{inner_eqn_idx}] {get_first_line(eqn_str)}  <-- NaN/Inf introduced here")

            # Close innermost level - use parent's indent (aligns with the opening)
            if total_levels > 1:
                parent_indent = "    " + "  " * (total_levels - 2)
                lines.append(f"{parent_indent}]")

            # Pass 3: Closing lines (innermost to outermost)
            # For each non-innermost level, show remaining equations and close bracket
            for level_idx in range(total_levels - 2, -1, -1):
                path_entry = nesting_path[level_idx]
                indent = "    " + "  " * level_idx
                nested_eqn_idx = path_entry['eqn_index']
                outer_eqn_strs = path_entry.get('all_eqn_strs', [])
                outer_total = path_entry.get('total_eqns', len(outer_eqn_strs))

                if is_level_within_depth(level_idx):
                    # Show remaining equations after the nested call
                    for i in range(nested_eqn_idx + 1, outer_total):
                        if i < len(outer_eqn_strs):
                            lines.append(f"{indent}[{i}] {get_first_line(outer_eqn_strs[i])}")
                    # Close with ]
                    lines.append(f"{indent}]")
                else:
                    # Just show closing with ellipsis
                    lines.append(f"{indent}...]")

        # Show context window if equation strings are available (fallback for non-nested)
        elif all_eqn_strs:
            lines.append("\n  Primitives:")
            if context == -1:
                start_idx, end_idx = 0, total_eqns
            else:
                start_idx = max(0, eqn_idx - context)
                end_idx = min(total_eqns, eqn_idx + context + 1)

            for i in range(start_idx, end_idx):
                marker = "  <-- NaN/Inf introduced here" if i == eqn_idx else ""
                lines.append(f"    [{i}] {all_eqn_strs[i]}{marker}")

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


class DebugNan:
    """
    JIT-compatible NaN/Inf debugging utility.

    Uses jax.debug.callback for host-side analysis and jax.lax.cond for
    conditional execution, making it fully compatible with jax.jit.

    Parameters
    ----------
    fn : Callable
        The function to debug.
    *args
        Arguments to pass to the function.
    phase : str, optional
        Phase name for the error message.
    depth : int, default 1
        Number of nesting levels to display.

        - depth=1: Show only innermost jaxpr
        - depth=2: Show innermost + one outer jaxpr
        - depth=-1: Show all nesting levels
    context : int, default 5
        Number of equations before/after NaN to show.
        Use -1 to show all equations.
    """

    def __init__(
        self,
        fn: Callable,
        *args,
        phase: str = '',
        depth: int = 1,
        context: int = 5
    ):
        self.fn = fn
        self.args = args
        self.phase = phase
        self.depth = depth
        self.context = context

        # Build jaxpr once during initialization
        self.stateful_fn = StatefulFunction(fn)
        self.jaxpr_info = self.stateful_fn.get_jaxpr(*args, compile_if_miss=True)
        self.flat_args, _ = jax.tree.flatten(self.args)

    def _do_nan_analysis(self, flat_args, cnsts):
        """Host callback that performs detailed NaN analysis."""
        outputs, nan_report, all_eqn_strs = _eval_jaxpr_with_nan_check(self.jaxpr_info.jaxpr, cnsts, flat_args)

        if self.phase:
            for item in nan_report:
                item['phase'] = self.phase

        if nan_report:
            report = _format_nan_report(
                nan_report,
                len(self.jaxpr_info.jaxpr.eqns),
                all_eqn_strs,
                context=self.context,
                depth=self.depth
            )
            raise RuntimeError(f"NaN/Inf detected:\n{report}")
        raise ValueError('NaN/Inf is not found during detailed analysis, unexpected state.')

    def check(self):
        """
        Unconditionally run NaN analysis (JIT-compatible via callback).
        """

        jax.debug.callback(self._do_nan_analysis, self.flat_args, self.jaxpr_info.consts)

    def check_if(self, has_nan):
        """
        Conditionally run NaN analysis only if has_nan is True (JIT-compatible).

        Parameters
        ----------
        has_nan : bool or jax.Array
            Condition to trigger debugging.
        """
        def _do_check():
            jax.debug.callback(self._do_nan_analysis, self.flat_args, self.jaxpr_info.consts)

        def _no_op():
            pass

        jax.lax.cond(
            unvmap(has_nan, op='any'),
            _do_check,
            _no_op,
        )


def debug_nan(
    fn: Callable,
    *args,
    phase: str = '',
    depth: int = 1,
    context: int = 5,
):
    """
    Debug NaN/Inf in a function by analyzing its jaxpr.

    This function is JIT-compatible via jax.debug.callback.

    Parameters
    ----------
    fn : Callable
        The function to debug.
    *args
        Arguments to pass to the function.
    phase : str, optional
        Phase name for the error message.
    depth : int, default 1
        Number of nesting levels to display.

        - depth=1: Show only innermost jaxpr
        - depth=2: Show innermost + one outer jaxpr
        - depth=-1: Show all nesting levels
    context : int, default 5
        Number of equations before/after NaN to show.
        Use -1 to show all equations.
    """
    DebugNan(fn, *args, phase=phase, depth=depth, context=context).check()


def debug_nan_if(
    has_nan: bool | jax.Array,
    fn: Callable,
    *args,
    phase: str = '',
    depth: int = 1,
    context: int = 5
):
    """
    Conditionally debug NaN/Inf in a function.

    This function is JIT-compatible via jax.lax.cond and jax.debug.callback.

    Parameters
    ----------
    has_nan : bool or jax.Array
        Condition to trigger debugging.
    fn : Callable
        The function to debug.
    *args
        Arguments to pass to the function.
    phase : str, optional
        Phase name for the error message.
    depth : int, default 1
        Number of nesting levels to display.

        - depth=1: Show only innermost jaxpr
        - depth=2: Show innermost + one outer jaxpr
        - depth=-1: Show all nesting levels
    context : int, default 5
        Number of equations before/after NaN to show.
        Use -1 to show all equations.
    """
    DebugNan(fn, *args, phase=phase, depth=depth, context=context).check_if(has_nan)
