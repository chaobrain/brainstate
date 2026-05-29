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

from typing import Union, Sequence

import jax
import numpy as np
from jax import lax
from jax._src.core import JaxprEqnContext
from jax.extend import source_info_util

from brainstate._compatible_import import (Literal, Var, Jaxpr, ClosedJaxpr, JaxprEqn)
# Shared, hardened IR internals. ``IdentitySet`` is re-exported here so that
# existing ``from brainstate.transform._ir_optim import IdentitySet`` keeps working.
from brainstate.transform._ir_utils import (
    IdentitySet,
    IRValidationError,
    partial_eval_jaxpr,
    literal_with_dtype,
    CONSTANT_FOLD_BLACKLIST,
)

__all__ = [
    'constant_fold',
    'dead_code_elimination',
    'common_subexpression_elimination',
    'copy_propagation',
    'algebraic_simplification',
    'optimize_jaxpr',
]


def _fallback_source_info(eqns: Sequence[JaxprEqn]) -> source_info_util.SourceInfo:
    if len(eqns) > 0:
        source_info = eqns[-1].source_info
        if source_info is not None:
            return source_info
    return source_info_util.new_source_info()


def _default_eqn_ctx() -> JaxprEqnContext:
    """Build a default ``JaxprEqnContext`` robustly across JAX versions.

    Newer JAX builds it from the current config with no positional arguments;
    older versions accepted ``(compute_on, threefry_partitionable)``.
    """
    try:
        return JaxprEqnContext()
    except TypeError:  # pragma: no cover - older JAX positional form
        return JaxprEqnContext(None, True)


def _assign_literal(
    literal: Literal,
    outvar: Var,
    source_info: source_info_util.SourceInfo
) -> JaxprEqn:
    eqn = JaxprEqn(
        [literal],
        [outvar],
        lax.convert_element_type_p,
        {'new_dtype': outvar.aval.dtype, 'weak_type': False, 'sharding': None},
        set(),
        source_info,
        _default_eqn_ctx(),
    )
    return eqn


def _preserve_invars_outvars(result: Jaxpr, jaxpr: Jaxpr):
    eqns = list(result.eqns)
    for v1, v2 in zip(result.outvars, jaxpr.outvars):
        if isinstance(v1, Literal) and isinstance(v2, Var):
            eqns.append(_assign_literal(v1, v2, _fallback_source_info(eqns)))
    # Ensure invars and outvars are preserved
    return result.replace(eqns=eqns, invars=jaxpr.invars, outvars=jaxpr.outvars)


def _canonical_param(value):
    """Return a hashable, value-stable representation of an equation param.

    Equation parameters may contain values that are unhashable (e.g. numpy
    arrays) or that compare/sort poorly (e.g. jaxprs). This produces a stable
    key so CSE can compare params without raising ``TypeError``.
    """
    if isinstance(value, (str, bytes, int, float, bool, type(None))):
        return value
    if isinstance(value, (tuple, list)):
        return tuple(_canonical_param(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((k, _canonical_param(v)) for k, v in value.items()))
    if isinstance(value, np.ndarray):
        return ('ndarray', value.shape, str(value.dtype), value.tobytes())
    if hasattr(value, 'shape') and hasattr(value, 'dtype'):
        try:
            arr = np.asarray(value)
            return ('array', arr.shape, str(arr.dtype), arr.tobytes())
        except Exception:
            return ('id', id(value))
    # Jaxprs, ClosedJaxprs and any other complex param: conservative identity.
    return ('id', id(value))


def constant_fold(jaxpr: Jaxpr) -> Jaxpr:
    """
    Perform constant folding optimization on a Jaxpr.

    This optimization evaluates all operations with constant inputs at
    compile time, replacing them with their computed constant values.
    This reduces runtime computation and can enable further optimizations.

    Parameters
    ----------
    jaxpr : Jaxpr
        The input Jaxpr to optimize.

    Returns
    -------
    Jaxpr
        A new Jaxpr with constant expressions evaluated. The input and
        output variables are preserved.

    Notes
    -----
    This optimization preserves the input and output variables of the jaxpr,
    only modifying the internal computation. Some primitives like
    'broadcast_in_dim' and 'broadcast' are blacklisted and won't be folded.

    Examples
    --------
    >>> # Given a jaxpr that computes: y = x + (2 + 3)
    >>> # After constant folding: y = x + 5
    >>> optimized_jaxpr = constant_fold(original_jaxpr)
    """
    result = partial_eval_jaxpr(jaxpr, {})
    return _preserve_invars_outvars(result, jaxpr)


def dead_code_elimination(jaxpr: Jaxpr) -> Jaxpr:
    """
    Remove equations whose outputs are not used (dead code elimination).

    This optimization performs a backward pass to identify which variables are
    actually used, then removes equations that produce unused outputs. This
    reduces the number of computations and can improve performance.

    Parameters
    ----------
    jaxpr : Jaxpr
        The input Jaxpr to optimize.

    Returns
    -------
    Jaxpr
        A new Jaxpr with dead code removed. All input and output variables
        are preserved.

    Notes
    -----
    This optimization preserves all input and output variables to maintain
    the function interface. Only internal dead computations are eliminated.

    The algorithm uses a two-phase approach:
    1. Backward pass: Mark all variables that are transitively used
    2. Forward pass: Keep only equations that produce marked variables

    Examples
    --------
    >>> # Given a jaxpr with unused intermediate computations
    >>> # Before: a = x + 1; b = x * 2; y = x + 2  (a and b unused)
    >>> # After:  y = x + 2
    >>> optimized_jaxpr = dead_code_elimination(original_jaxpr)
    """
    # Mark all variables that are used (starting from outputs and ALL inputs)
    # We must keep all invars even if they appear unused, as they define the interface
    used_vars = IdentitySet(jaxpr.outvars)
    used_vars.update(jaxpr.invars)

    # Backward pass: mark variables as used if they're inputs to used equations
    # We need to iterate until convergence
    changed = True
    while changed:
        changed = False
        for eqn in reversed(jaxpr.eqns):
            # If any output is used, all inputs must be kept
            if any(outvar in used_vars for outvar in eqn.outvars):
                for invar in eqn.invars:
                    if invar not in used_vars and not isinstance(invar, Literal):
                        used_vars.add(invar)
                        changed = True

    # Forward pass: keep only equations that produce used outputs
    new_eqns = []
    for eqn in jaxpr.eqns:
        if any(outvar in used_vars for outvar in eqn.outvars):
            new_eqns.append(eqn)

    # Keep all input and output variables unchanged
    return jaxpr.replace(eqns=new_eqns, invars=jaxpr.invars, outvars=jaxpr.outvars)


def common_subexpression_elimination(jaxpr: Jaxpr) -> Jaxpr:
    """
    Eliminate redundant computations by reusing results (CSE).

    Common Subexpression Elimination identifies equations that perform the
    same operation with identical inputs and reuses the result instead of
    recomputing. This reduces redundant computations and memory usage.

    Parameters
    ----------
    jaxpr : Jaxpr
        The input Jaxpr to optimize.

    Returns
    -------
    Jaxpr
        A new Jaxpr with common subexpressions eliminated. All input and
        output variables are preserved.

    Notes
    -----
    This optimization preserves all input and output variables. When output
    variables are mapped to other variables due to CSE, identity equations
    (using ``convert_element_type`` with the same dtype) are added to maintain
    the correct interface.

    Two equations are considered identical if they have:
    - The same primitive operation
    - The same input variables (by identity)
    - The same parameters

    Examples
    --------
    >>> # Given a jaxpr with duplicate computations
    >>> # Before: a = x + y; b = x * 2; c = x + y  (c duplicates a)
    >>> # After:  a = x + y; b = x * 2; c = a
    >>> optimized_jaxpr = common_subexpression_elimination(original_jaxpr)
    """
    # Map from (primitive, invars, params) to output variables
    expr_cache = {}
    # Map from old variables to their replacements
    var_map = {}

    def get_var(var):
        """Get the canonical variable, following replacements."""
        if isinstance(var, Literal):
            return var
        return var_map.get(var, var)

    def make_key(eqn):
        """Create a hashable key for an equation, or ``None`` if not possible.

        Returning ``None`` opts the equation out of CSE (kept verbatim) rather
        than crashing on params that cannot be canonicalized.
        """
        invars_ids = tuple(id(get_var(v)) for v in eqn.invars)
        try:
            param_sig = tuple(sorted(
                (k, _canonical_param(v)) for k, v in eqn.params.items()
            ))
        except Exception:
            return None
        return (eqn.primitive.name, invars_ids, param_sig)

    new_eqns = []

    for eqn in jaxpr.eqns:
        # Update invars to use canonical variables
        canonical_invars = tuple(get_var(v) for v in eqn.invars)
        eqn = eqn.replace(invars=canonical_invars)

        # Check if we've seen this computation before
        key = make_key(eqn)

        if key is not None and key in expr_cache and len(eqn.outvars) == len(expr_cache[key]):
            # Reuse previous result
            prev_outvars = expr_cache[key]
            for old_var, new_var in zip(eqn.outvars, prev_outvars):
                var_map[old_var] = new_var
        else:
            # This is a new computation, keep it
            new_eqns.append(eqn)
            if key is not None:
                expr_cache[key] = eqn.outvars

    # For outvars that have been replaced, add identity equations to preserve the interface
    final_eqns = new_eqns[:]
    outvars_need_identity = []
    for outvar in jaxpr.outvars:
        canonical = get_var(outvar)
        if id(canonical) != id(outvar):
            outvars_need_identity.append((outvar, canonical))

    # Add identity equations if needed
    if outvars_need_identity:
        default_ctx = _default_eqn_ctx()
        for outvar, canonical in outvars_need_identity:
            # Create an identity equation: outvar = identity(canonical)
            # Use convert_element_type as identity (same type)
            eqn = JaxprEqn([canonical],
                           [outvar],
                           lax.convert_element_type_p,
                           {'new_dtype': outvar.aval.dtype, 'weak_type': False, 'sharding': None},
                           set(),
                           _fallback_source_info(new_eqns),
                           default_ctx)
            final_eqns.append(eqn)

    # Keep original outvars and invars
    return jaxpr.replace(eqns=final_eqns, outvars=jaxpr.outvars, invars=jaxpr.invars, debug_info=None)


def copy_propagation(jaxpr: Jaxpr) -> Jaxpr:
    """
    Eliminate unnecessary copy operations by propagating original variables.

    When a variable is simply copied or renamed via identity operations
    (copy, device_put, or redundant convert_element_type), this optimization
    propagates the original variable forward, eliminating the copy operation.

    Parameters
    ----------
    jaxpr : Jaxpr
        The input Jaxpr to optimize.

    Returns
    -------
    Jaxpr
        A new Jaxpr with copies propagated. All input and output variables
        are preserved.

    Notes
    -----
    This optimization preserves all input and output variables. Copy operations
    that produce output variables are kept to maintain the correct interface.

    The following operations are considered identity operations:
    - ``copy``: Always an identity
    - ``device_put``: Always an identity
    - ``convert_element_type``: Only when the input and output dtypes match

    Examples
    --------
    >>> # Given a jaxpr with unnecessary copies
    >>> # Before: a = copy(x); b = a + 1; c = copy(b)
    >>> # After:  b = x + 1; c = copy(b)
    >>> optimized_jaxpr = copy_propagation(original_jaxpr)
    """
    # Map from variables to their canonical representatives
    var_map = {}
    # Track which outvars are identity operations that can be safely removed
    identity_outvars = set()

    def get_canonical(var):
        """Follow the chain of copies to find the canonical variable."""
        if isinstance(var, Literal):
            return var
        original = var
        seen = set()
        while var in var_map and id(var) not in seen:
            seen.add(id(var))
            var = var_map[var]
        return var

    new_eqns = []

    for eqn in jaxpr.eqns:
        # Replace input variables with their canonical versions
        new_invars = tuple(get_canonical(v) for v in eqn.invars)

        # Check for identity/copy operations
        is_identity = False
        if eqn.primitive.name in ('copy', 'device_put', 'convert_element_type'):
            # These are potential identity operations
            if len(new_invars) == 1 and len(eqn.outvars) == 1:
                invar = new_invars[0]
                outvar = eqn.outvars[0]

                # For convert_element_type, check if types match
                if eqn.primitive.name == 'convert_element_type':
                    if hasattr(invar, 'aval') and hasattr(outvar, 'aval'):
                        if invar.aval.dtype == eqn.params.get('new_dtype'):
                            is_identity = True
                else:
                    is_identity = True

                if is_identity:
                    # Only eliminate if outvar is not in the original outvars
                    if outvar not in jaxpr.outvars:
                        var_map[outvar] = invar
                    else:
                        # Keep the identity equation if it's an output variable
                        is_identity = False

        if not is_identity:
            # Keep the equation with updated invars
            eqn = eqn.replace(invars=new_invars)
            new_eqns.append(eqn)

    # Update outvars, but keep them as-is since we preserved identity ops for them
    # Apply canonical mapping only to internal references
    new_outvars = jaxpr.outvars

    # Keep all input and output variables unchanged
    return jaxpr.replace(eqns=new_eqns, invars=jaxpr.invars, outvars=new_outvars)


def algebraic_simplification(jaxpr: Jaxpr) -> Jaxpr:
    """
    Apply algebraic identities to simplify arithmetic operations.

    This optimization recognizes and applies common algebraic identities
    to simplify operations, reducing computational complexity and enabling
    further optimizations.

    Parameters
    ----------
    jaxpr : Jaxpr
        The input Jaxpr to optimize.

    Returns
    -------
    Jaxpr
        A new Jaxpr with algebraic simplifications applied. All input and
        output variables are preserved.

    Notes
    -----
    This optimization preserves all input and output variables. When output
    variables are simplified, identity equations are added to maintain the
    correct interface.

    The following algebraic identities are recognized:

    Addition:
        - ``0 + x = x``
        - ``x + 0 = x``

    Subtraction:
        - ``x - 0 = x``
        - ``x - x = 0``

    Multiplication:
        - ``0 * x = 0``
        - ``x * 0 = 0``
        - ``1 * x = x``
        - ``x * 1 = x``

    Division:
        - ``x / 1 = x``
        - ``0 / x = 0`` (assuming x != 0)

    Examples
    --------
    >>> # Given a jaxpr with algebraic simplifications
    >>> # Before: a = x + 0; b = a * 1; c = b - 0
    >>> # After:  a = x; b = a; c = b
    >>> optimized_jaxpr = algebraic_simplification(original_jaxpr)
    """

    # Map from variables to their replacements (for eliminated operations)
    var_map = {}

    def get_var(var):
        """Get the canonical variable."""
        if isinstance(var, Literal):
            return var
        return var_map.get(var, var)

    def is_constant_value(var, value):
        """Check if a variable is a literal with a specific value."""
        if not isinstance(var, Literal):
            return False
        val = var.val
        try:
            # Handle scalar and array constants
            if isinstance(val, (int, float, complex)):
                return val == value
            elif hasattr(val, '__array__'):
                arr = np.asarray(val)
                return arr.shape == () and arr.item() == value
        except:
            pass
        return False

    def is_zero(var):
        return is_constant_value(var, 0)

    def is_one(var):
        return is_constant_value(var, 1)

    def make_literal(value, aval):
        """Create a literal whose value matches the aval's dtype."""
        return literal_with_dtype(value, aval)

    new_eqns = []

    for eqn in jaxpr.eqns:
        # Update invars to use canonical variables
        canonical_invars = tuple(get_var(v) for v in eqn.invars)
        simplified = False

        if len(canonical_invars) >= 2 and len(eqn.outvars) == 1:
            lhs, rhs = canonical_invars[0], canonical_invars[1]
            outvar = eqn.outvars[0]

            # Addition simplifications
            if eqn.primitive.name == 'add':
                if is_zero(lhs):  # 0 + x = x
                    var_map[outvar] = rhs
                    simplified = True
                elif is_zero(rhs):  # x + 0 = x
                    var_map[outvar] = lhs
                    simplified = True

            # Subtraction simplifications
            elif eqn.primitive.name == 'sub':
                if is_zero(rhs):  # x - 0 = x
                    var_map[outvar] = lhs
                    simplified = True
                elif id(lhs) == id(rhs):  # x - x = 0
                    var_map[outvar] = make_literal(0, outvar.aval)
                    simplified = True

            # Multiplication simplifications
            elif eqn.primitive.name == 'mul':
                # ``0 * x = 0`` is only IEEE-safe for integer dtypes: for floats
                # ``0 * inf`` and ``0 * nan`` are ``nan``, not ``0``.
                out_is_integer = np.issubdtype(np.dtype(outvar.aval.dtype), np.integer)
                if (is_zero(lhs) or is_zero(rhs)) and out_is_integer:
                    var_map[outvar] = make_literal(0, outvar.aval)
                    simplified = True
                elif is_one(lhs):  # 1 * x = x
                    var_map[outvar] = rhs
                    simplified = True
                elif is_one(rhs):  # x * 1 = x
                    var_map[outvar] = lhs
                    simplified = True

            # Division simplifications
            elif eqn.primitive.name == 'div':
                # ``0 / x = 0`` is unsafe (``0 / 0 = nan``) for every dtype, so
                # only ``x / 1 = x`` is applied.
                if is_one(rhs):  # x / 1 = x
                    var_map[outvar] = lhs
                    simplified = True

        if not simplified:
            # Keep the equation with updated invars
            eqn = eqn.replace(invars=canonical_invars)
            new_eqns.append(eqn)

    # For outvars that have been replaced, add identity equations to preserve the interface
    final_eqns = new_eqns[:]
    outvars_need_identity = []
    for outvar in jaxpr.outvars:
        canonical = get_var(outvar)
        if id(canonical) != id(outvar):
            outvars_need_identity.append((outvar, canonical))

    # Add identity equations if needed
    if outvars_need_identity:
        for outvar, canonical in outvars_need_identity:
            # Create an identity equation: outvar = identity(canonical)
            final_eqns.append(_assign_literal(canonical, outvar, _fallback_source_info(final_eqns)))

    # Keep original outvars and invars
    return jaxpr.replace(eqns=final_eqns, outvars=jaxpr.outvars, invars=jaxpr.invars)


def optimize_jaxpr(
    jaxpr: Jaxpr | ClosedJaxpr,
    max_iterations: int = 3,
    optimizations: Sequence[str] | None = None,
    verbose: bool = False,
) -> Jaxpr | ClosedJaxpr:
    """
    Apply multiple optimization passes to a Jaxpr.

    This function applies a sequence of optimizations in multiple iterations
    until convergence or the maximum number of iterations is reached. The
    optimizations work together to simplify the computation graph while
    preserving the function's semantics and interface.

    Parameters
    ----------
    jaxpr : Jaxpr or ClosedJaxpr
        The input Jaxpr or ClosedJaxpr to optimize.
    max_iterations : int, optional
        Maximum number of optimization passes. Default is 3.
    optimizations : sequence of str, optional
        List of optimization names to apply in order. If None, applies all
        optimizations in the recommended order: constant_fold, algebraic_simplification,
        copy_propagation, cse, dce. Use a custom list to control which optimizations
        run and in what order.
    verbose : bool, optional
        If True, print detailed optimization progress information including
        equation counts and reduction statistics. Default is False.

    Returns
    -------
    Jaxpr or ClosedJaxpr
        An optimized Jaxpr or ClosedJaxpr (same type as input) with reduced
        equation count and improved efficiency.

    Raises
    ------
    TypeError
        If the input is not a Jaxpr or ClosedJaxpr.
    ValueError
        If any optimization name in ``optimizations`` is invalid.
    RuntimeError
        If the input or output variables change during optimization (indicates
        a bug in the optimization passes).

    Notes
    -----
    Available optimizations:

    - **constant_fold**: Evaluate constant expressions at compile time
    - **algebraic_simplification**: Apply algebraic identities (x+0=x, x*1=x, etc.)
    - **copy_propagation**: Eliminate unnecessary copy operations
    - **cse**: Common subexpression elimination (reuse identical computations)
    - **dce**: Dead code elimination (remove unused equations)

    The optimization process iterates until:

    1. No more equations can be eliminated (convergence), or
    2. The maximum number of iterations is reached

    All optimizations preserve the function interface (input and output variables)
    while optimizing the internal computation graph.

    Examples
    --------
    Apply all default optimizations:

    .. code-block:: python

        >>> optimized = optimize_jaxpr(jaxpr)

    Use more iterations for aggressive optimization:

    .. code-block:: python

        >>> optimized = optimize_jaxpr(jaxpr, max_iterations=5)

    Run only specific optimizations:

    .. code-block:: python

        >>> optimized = optimize_jaxpr(jaxpr, optimizations=['constant_fold', 'dce'])

    Enable verbose output to see optimization progress:

    .. code-block:: python

        >>> optimized = optimize_jaxpr(jaxpr, verbose=True)
        Starting optimization with 50 equations
        Optimization sequence: constant_fold -> algebraic_simplification -> ...
        Max iterations: 3
        ------------------------------------------------------------

        Iteration 1:
          constant_fold: 50 -> 45 equations (-5)
          algebraic_simplification: 45 -> 42 equations (-3)
          dce: 42 -> 38 equations (-4)

        Converged after 2 iteration(s)
        ------------------------------------------------------------
        Optimization complete:
          Initial equations: 50
          Final equations:   38
          Reduction:         12 (24.0%)

    Custom optimization pipeline:

    .. code-block:: python

        >>> # First fold constants, then eliminate dead code
        >>> stage1 = optimize_jaxpr(jaxpr, optimizations=['constant_fold', 'dce'])
        >>> # Then apply CSE and more DCE
        >>> stage2 = optimize_jaxpr(stage1, optimizations=['cse', 'dce'])
    """
    _DEFAULT_PIPELINE = [
        'constant_fold',
        'algebraic_simplification',
        'copy_propagation',
        'cse',
        'dce',
    ]
    if optimizations is None:
        optimizations = list(_DEFAULT_PIPELINE)
    elif isinstance(optimizations, str):
        optimizations = list(_DEFAULT_PIPELINE) if optimizations == 'all' else [optimizations]
    else:
        optimizations = list(optimizations)

    if not isinstance(max_iterations, int) or isinstance(max_iterations, bool) or max_iterations < 1:
        raise IRValidationError(
            f"max_iterations must be a positive integer, got {max_iterations!r}."
        )

    # Parse input
    if isinstance(jaxpr, Jaxpr):
        closed_jaxpr = None
    elif isinstance(jaxpr, ClosedJaxpr):
        closed_jaxpr = jaxpr
        jaxpr = jaxpr.jaxpr
    else:
        raise TypeError(f'Expected Jaxpr or ClosedJaxpr, got {type(jaxpr)}')

    # Store original interface
    invars_before = tuple(jaxpr.invars)
    outvars_before = tuple(jaxpr.outvars)
    initial_eqns = len(jaxpr.eqns)

    # Define available optimizations
    _OPTIMIZATION_MAP = {
        'constant_fold': constant_fold,
        'algebraic_simplification': algebraic_simplification,
        'copy_propagation': copy_propagation,
        'cse': common_subexpression_elimination,
        'dce': dead_code_elimination,
    }

    # Validate optimization names
    invalid_opts = set(optimizations) - set(_OPTIMIZATION_MAP.keys())
    if invalid_opts:
        available = ', '.join(sorted(_OPTIMIZATION_MAP.keys()))
        raise ValueError(
            f"Invalid optimization(s): {', '.join(invalid_opts)}. "
            f"Available optimizations: {available}"
        )

    if verbose:
        print(f"Starting optimization with {initial_eqns} equations")
        print(f"Optimization sequence: {' -> '.join(optimizations)}")
        print(f"Max iterations: {max_iterations}")
        print("-" * 60)

    # Apply optimization iterations
    for iteration in range(max_iterations):
        prev_num_eqns = len(jaxpr.eqns)

        if verbose:
            print(f"\nIteration {iteration + 1}:")

        # Apply each optimization in sequence
        for opt_name in optimizations:
            opt_func = _OPTIMIZATION_MAP[opt_name]
            prev_eqns = len(jaxpr.eqns)
            jaxpr = opt_func(jaxpr)
            current_eqns = len(jaxpr.eqns)

            if verbose and current_eqns != prev_eqns:
                reduction = prev_eqns - current_eqns
                print(f"  {opt_name}: {prev_eqns} -> {current_eqns} equations "
                      f"({reduction:+d})")

        # Check for convergence
        if len(jaxpr.eqns) == prev_num_eqns:
            if verbose:
                print(f"\nConverged after {iteration + 1} iteration(s)")
            break
    else:
        if verbose:
            print(f"\nReached max iterations ({max_iterations})")

    # Final statistics
    final_eqns = len(jaxpr.eqns)
    if verbose:
        print("-" * 60)
        print(f"Optimization complete:")
        print(f"  Initial equations: {initial_eqns}")
        print(f"  Final equations:   {final_eqns}")
        print(f"  Reduction:         {initial_eqns - final_eqns} "
              f"({100 * (initial_eqns - final_eqns) / initial_eqns:.1f}%)")

    # Validate that interface is preserved
    invars_after = tuple(jaxpr.invars)
    outvars_after = tuple(jaxpr.outvars)
    if invars_before != invars_after:
        raise RuntimeError(
            f'Input variables changed during optimization. '
            f'Before: {len(invars_before)}, After: {len(invars_after)}'
        )
    if outvars_before != outvars_after:
        raise RuntimeError(
            f'Output variables changed during optimization. '
            f'Before: {len(outvars_before)}, After: {len(outvars_after)}'
        )

    # Restore ClosedJaxpr if needed
    if closed_jaxpr is not None:
        jaxpr = ClosedJaxpr(jaxpr, closed_jaxpr.consts)

    return jaxpr
