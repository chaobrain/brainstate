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

from collections.abc import MutableSet
from typing import Union

import jax

from brainstate._compatible_import import (Literal, Var, Jaxpr, ClosedJaxpr)

__all__ = [
    'constant_fold',
    'dead_code_elimination',
    'common_subexpression_elimination',
    'copy_propagation',
    'algebraic_simplification',
    'optimize_jaxpr',
]


class IdentitySet(MutableSet):
    """Set that compares objects by identity.

    This is a set that compares objects by identity instead of equality. It is
    useful for storing objects that are not hashable or that should be compared
    by identity.

    This is a mutable set, but it does not support the ``__hash__`` method and
    therefore cannot be used as a dictionary key or as an element of another set.
    """

    def __init__(self, iterable=None):
        self._data = {}
        if iterable is not None:
            self.update(iterable)

    def __contains__(self, value):
        return id(value) in self._data

    def __iter__(self):
        return iter(self._data.values())

    def __len__(self):
        return len(self._data)

    def add(self, value):
        self._data[id(value)] = value

    def discard(self, value):
        self._data.pop(id(value), None)

    def __repr__(self):
        return f"IdentitySet({list(repr(x) for x in self._data.values())})"

    def __str__(self):
        return f"IdentitySet({list(str(x) for x in self._data.values())})"


def constant_fold(jaxpr: Jaxpr):
    """
    Given a jaxpr, return a new jaxpr with all constant folding done.
    """
    return _partial_eval_jaxpr(jaxpr, {})


_constant_fold_blacklist = {'broadcast_in_dim', 'broadcast'}


def _partial_eval_jaxpr(jaxpr, env):
    env = env.copy()
    new_eqns = []

    def read(var):
        if isinstance(var, Literal):
            return var.val
        else:
            return env.get(var, None)

    def read_or_self(var):
        out = read(var)
        if out is None:
            return var
        elif isinstance(out, Var):
            return out
        elif isinstance(out, Literal):
            return Literal(out.val, var.aval)
        else:
            assert not isinstance(out, Jaxpr)
            return Literal(out, var.aval)

    for eqn in jaxpr.eqns:
        vals = [read(var) for var in eqn.invars]
        if eqn.primitive.name in _constant_fold_blacklist:
            new_eqns.append(eqn)
        elif all(val is not None for val in vals):
            # go ahead and eval it
            out = _eval_eqn(eqn, vals)

            # two options: either it's a jaxpr result (partial eval) or it's a value or a list of values
            if isinstance(out, Jaxpr):
                # we need to inline this
                new_eqns.extend(out.eqns)
                out = out.outvars
            elif not isinstance(out, tuple) and not isinstance(out, list):
                out = (out,)

            for var, val in zip(eqn.outvars, out):
                assert not isinstance(val, Jaxpr)
                if isinstance(val, Literal):
                    env[var] = val.val
                else:
                    env[var] = val
        else:
            new_eqns.append(eqn)

    # now that we've eval everything, inline all the constants
    out_eqns = []
    for eqn in new_eqns:
        eqn = eqn.replace(invars=tuple(read_or_self(var) for var in eqn.invars))
        out_eqns.append(eqn)

    invars_still_used = IdentitySet()
    for eqn in out_eqns:
        for var in eqn.invars:
            invars_still_used.add(var)

    invars = tuple(var for var in jaxpr.invars if var in invars_still_used)

    # sub in any constants for outvars
    outvars = tuple(read_or_self(var) for var in jaxpr.outvars)

    return jaxpr.replace(eqns=out_eqns, outvars=outvars, invars=invars, debug_info=None)


def _eval_eqn(eqn, vals) -> Union[Jaxpr, tuple, list, jax.Array]:
    if eqn.primitive.name == "closed_call":
        assert eqn.primitive.call_primitive
        assert not eqn.primitive.map_primitive

        out = _partial_eval_jaxpr(
            eqn.params['call_jaxpr'].jaxpr,
            {
                var: val
                for var, val in
                zip(eqn.params['call_jaxpr'].jaxpr.invars, vals)
            }
        )
    elif eqn.primitive.name == "scan":
        out = eqn.primitive.bind(*vals, **eqn.params)
    else:
        out = eqn.primitive.bind(*vals, **eqn.params)
    return out


def dead_code_elimination(jaxpr: Jaxpr) -> Jaxpr:
    """
    Remove equations whose outputs are not used.

    This optimization performs a backward pass to identify which variables are
    actually used, then removes equations that produce unused outputs.

    Args:
        jaxpr: The input Jaxpr to optimize

    Returns:
        A new Jaxpr with dead code removed
    """
    # Mark all variables that are used (starting from outputs)
    used_vars = IdentitySet(jaxpr.outvars)

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

    # Keep only input variables that are actually used
    new_invars = tuple(var for var in jaxpr.invars if var in used_vars)

    return jaxpr.replace(eqns=new_eqns, invars=new_invars, debug_info=None)


def common_subexpression_elimination(jaxpr: Jaxpr) -> Jaxpr:
    """
    Eliminate redundant computations by reusing results of identical operations.

    This optimization identifies equations that perform the same operation with
    the same inputs and reuses the result instead of recomputing.

    Args:
        jaxpr: The input Jaxpr to optimize

    Returns:
        A new Jaxpr with common subexpressions eliminated
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
        """Create a hashable key for an equation."""
        # Use identity of variables for comparison
        invars_ids = tuple(id(get_var(v)) for v in eqn.invars)
        # Create a hashable representation of params
        param_items = tuple(sorted(eqn.params.items()))
        return (eqn.primitive.name, invars_ids, param_items)

    new_eqns = []

    for eqn in jaxpr.eqns:
        # Update invars to use canonical variables
        canonical_invars = tuple(get_var(v) for v in eqn.invars)
        eqn = eqn.replace(invars=canonical_invars)

        # Check if we've seen this computation before
        key = make_key(eqn)

        if key in expr_cache and len(eqn.outvars) == len(expr_cache[key]):
            # Reuse previous result
            prev_outvars = expr_cache[key]
            for old_var, new_var in zip(eqn.outvars, prev_outvars):
                var_map[old_var] = new_var
        else:
            # This is a new computation, keep it
            new_eqns.append(eqn)
            expr_cache[key] = eqn.outvars

    # Update output variables
    new_outvars = tuple(get_var(v) for v in jaxpr.outvars)

    return jaxpr.replace(eqns=new_eqns, outvars=new_outvars, debug_info=None)


def copy_propagation(jaxpr: Jaxpr) -> Jaxpr:
    """
    Replace variables with their aliases to reduce unnecessary copies.

    When a variable is simply copied or renamed (identity operation), this
    optimization propagates the original variable forward, eliminating the copy.

    Args:
        jaxpr: The input Jaxpr to optimize

    Returns:
        A new Jaxpr with copies propagated
    """
    # Map from variables to their canonical representatives
    var_map = {}

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
                    # Map output to input (they're the same)
                    var_map[outvar] = invar

        if not is_identity:
            # Keep the equation with updated invars
            eqn = eqn.replace(invars=new_invars)
            new_eqns.append(eqn)

    # Update output variables
    new_outvars = tuple(get_canonical(v) for v in jaxpr.outvars)

    # Update input variables (remove unused ones)
    used_invars = IdentitySet()
    for eqn in new_eqns:
        for var in eqn.invars:
            if not isinstance(var, Literal):
                used_invars.add(var)
    for var in new_outvars:
        if not isinstance(var, Literal):
            used_invars.add(var)

    new_invars = tuple(var for var in jaxpr.invars if var in used_invars)

    return jaxpr.replace(eqns=new_eqns, invars=new_invars, outvars=new_outvars, debug_info=None)


def algebraic_simplification(jaxpr: Jaxpr) -> Jaxpr:
    """
    Apply algebraic identities to simplify operations.

    This optimization recognizes patterns like:
    - x + 0 = x
    - x * 1 = x
    - x * 0 = 0
    - x - x = 0

    Args:
        jaxpr: The input Jaxpr to optimize

    Returns:
        A new Jaxpr with algebraic simplifications applied
    """
    import numpy as np

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
        """Create a literal with the given value and abstract value."""
        return Literal(value, aval)

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
                if is_zero(lhs) or is_zero(rhs):  # 0 * x = 0 or x * 0 = 0
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
                if is_one(rhs):  # x / 1 = x
                    var_map[outvar] = lhs
                    simplified = True
                elif is_zero(lhs):  # 0 / x = 0 (assuming x != 0)
                    var_map[outvar] = make_literal(0, outvar.aval)
                    simplified = True

        if not simplified:
            # Keep the equation with updated invars
            eqn = eqn.replace(invars=canonical_invars)
            new_eqns.append(eqn)

    # Update output variables
    new_outvars = tuple(get_var(v) for v in jaxpr.outvars)

    return jaxpr.replace(eqns=new_eqns, outvars=new_outvars, debug_info=None)


def optimize_jaxpr(jaxpr: Jaxpr, max_iterations: int = 3) -> Jaxpr:
    """
    Apply multiple optimization passes to a Jaxpr.

    This function applies a sequence of optimizations in multiple iterations
    until convergence or max_iterations is reached. The optimizations are:
    1. Constant folding
    2. Algebraic simplification
    3. Copy propagation
    4. Common subexpression elimination
    5. Dead code elimination

    Args:
        jaxpr: The input Jaxpr to optimize
        max_iterations: Maximum number of optimization passes (default: 3)

    Returns:
        An optimized Jaxpr
    """
    if isinstance(jaxpr, Jaxpr):
        closed_jaxpr = None
    elif isinstance(jaxpr, ClosedJaxpr):
        closed_jaxpr = jaxpr
        jaxpr = jaxpr.jaxpr
    else:
        raise TypeError(f'Expected Jaxpr or ClosedJaxpr, got {type(jaxpr)}')

    for i in range(max_iterations):
        prev_num_eqns = len(jaxpr.eqns)

        # Apply optimizations in sequence
        jaxpr = constant_fold(jaxpr)
        jaxpr = algebraic_simplification(jaxpr)
        jaxpr = copy_propagation(jaxpr)
        jaxpr = common_subexpression_elimination(jaxpr)
        jaxpr = dead_code_elimination(jaxpr)

        # Check for convergence
        if len(jaxpr.eqns) == prev_num_eqns:
            break

    if closed_jaxpr is not None:
        jaxpr = ClosedJaxpr(jaxpr, closed_jaxpr.consts)
    return jaxpr
