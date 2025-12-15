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

"""
JAX jaxpr rewriting utilities to expand jit equations conditionally.

This module provides utilities for transforming JAX intermediate representations (jaxpr)
by selectively inlining JIT-compiled functions. This can be useful for optimization passes,
debugging, or custom transformations that need to work at the jaxpr level.
"""

from typing import Callable, Optional, Union

from brainstate._compatible_import import (
    Jaxpr, JaxprEqn, Literal, ClosedJaxpr, is_jit_primitive
)

__all__ = [
    'inline_jit',
]


def inline_jit(
    jaxpr: Union[Jaxpr, ClosedJaxpr],
    should_expand: Optional[Callable[[JaxprEqn], bool]] = None
) -> Union[Jaxpr, ClosedJaxpr]:
    """
    Rewrite a jaxpr by expanding (inlining) jit equations that satisfy the given condition.

    This function recursively traverses a jaxpr and expands (inlines) JIT-compiled function
    calls based on a user-provided predicate. Variables are carefully remapped to maintain
    correctness across scope boundaries.

    Parameters
    ----------
    jaxpr : Jaxpr or ClosedJaxpr
        The input jaxpr to rewrite. Can be either a Jaxpr or ClosedJaxpr.
    should_expand : callable, optional
        A predicate function that takes a JaxprEqn and returns True if the jit should
        be expanded. If None, all jit equations are expanded. The predicate can inspect
        equation parameters like call_jaxpr to make decisions based on the function's
        complexity, size, or content.

    Returns
    -------
    Jaxpr or ClosedJaxpr
        A new jaxpr with qualified jit equations expanded. The return type matches the
        input type (Jaxpr returns Jaxpr, ClosedJaxpr returns ClosedJaxpr).

    Examples
    --------
    .. code-block:: python

        >>> from jax import make_jaxpr
        >>> import jax.numpy as jnp
        >>> import jax
        >>>
        >>> @jax.jit
        ... def inner(x):
        ...     return x + 1
        >>>
        >>> def outer(x):
        ...     return inner(x) * 2
        >>>
        >>> jaxpr = make_jaxpr(outer)(1.0)
        >>> expanded = inline_jit(jaxpr.jaxpr)  # Expands all jits
        >>>
        >>> # Conditional expansion - only expand small functions
        >>> def expand_small(eqn):
        ...     call_jaxpr = eqn.params.get('call_jaxpr') or eqn.params.get('jaxpr')
        ...     return call_jaxpr and len(call_jaxpr.eqns) <= 5
        >>> expanded = inline_jit(jaxpr.jaxpr, expand_small)
    """
    if should_expand is None:
        should_expand = lambda eqn: True

    # Handle ClosedJaxpr by unwrapping to Jaxpr
    is_closed = isinstance(jaxpr, ClosedJaxpr)
    original_closed = jaxpr if is_closed else None
    if is_closed:
        inner_jaxpr = jaxpr.jaxpr
    else:
        inner_jaxpr = jaxpr

    new_eqns = []
    var_mapping = {v: v for v in inner_jaxpr.invars}

    for eqn in inner_jaxpr.eqns:
        # Check if this is a jit primitive that should be expanded
        if is_jit_primitive(eqn) and should_expand(eqn):
            # Get the jaxpr from the jit equation
            call_jaxpr = eqn.params.get('call_jaxpr')
            if call_jaxpr is None:
                # Fallback for different jit variants
                call_jaxpr = eqn.params.get('jaxpr')

            if call_jaxpr is not None:
                # Map input variables from outer scope to inner jaxpr
                # Extract the actual jaxpr if this is a ClosedJaxpr
                if isinstance(call_jaxpr, ClosedJaxpr):
                    actual_jaxpr = call_jaxpr.jaxpr
                else:
                    actual_jaxpr = call_jaxpr

                inner_var_mapping = {}
                for inner_var, outer_var in zip(actual_jaxpr.invars, eqn.invars):
                    mapped_var = var_mapping.get(outer_var, outer_var)
                    inner_var_mapping[inner_var] = mapped_var

                # Recursively expand the inner jaxpr
                expanded_inner = inline_jit(call_jaxpr, should_expand)

                # Unwrap if ClosedJaxpr
                if isinstance(expanded_inner, ClosedJaxpr):
                    expanded_inner = expanded_inner.jaxpr

                # Inline the equations from the inner jaxpr
                for inner_eqn in expanded_inner.eqns:
                    # Remap variables in the inner equation
                    new_invars = [
                        inner_var_mapping.get(v, v) if not isinstance(v, Literal) else v
                        for v in inner_eqn.invars
                    ]
                    new_outvars = []

                    for v in inner_eqn.outvars:
                        if v in inner_var_mapping:
                            new_outvars.append(inner_var_mapping[v])
                        else:
                            # Keep the original variable but add to mapping
                            # The variable is already unique from the inner scope
                            inner_var_mapping[v] = v
                            new_outvars.append(v)

                    # Create the remapped equation with only parameters that exist
                    # Build kwargs dynamically to handle different JAX versions
                    replace_kwargs = {
                        'primitive': inner_eqn.primitive,
                        'invars': new_invars,
                        'outvars': new_outvars,
                        'params': inner_eqn.params,
                    }

                    # Add optional fields if they exist
                    if hasattr(inner_eqn, 'effects'):
                        replace_kwargs['effects'] = inner_eqn.effects
                    if hasattr(inner_eqn, 'source_info'):
                        replace_kwargs['source_info'] = inner_eqn.source_info

                    new_eqn = inner_eqn.replace(**replace_kwargs)
                    new_eqns.append(new_eqn)

                # Map the output variables
                for inner_out, outer_out in zip(expanded_inner.outvars, eqn.outvars):
                    var_mapping[outer_out] = inner_var_mapping.get(inner_out, inner_out)

            else:
                # If we can't find the jaxpr, keep the original equation
                new_eqns.append(eqn)
                for v in eqn.outvars:
                    var_mapping[v] = v
        else:
            # Keep the equation as is, but remap variables
            new_invars = [
                var_mapping.get(v, v) if not isinstance(v, Literal) else v
                for v in eqn.invars
            ]
            new_eqns.append(eqn.replace(invars=new_invars))
            for v in eqn.outvars:
                var_mapping[v] = v

    # Remap output variables
    new_outvars = [var_mapping.get(v, v) for v in inner_jaxpr.outvars]

    # Create the new jaxpr
    new_jaxpr = inner_jaxpr.replace(eqns=new_eqns, outvars=new_outvars)

    # Rewrap in ClosedJaxpr if that's what we started with
    if is_closed:
        return original_closed.replace(jaxpr=new_jaxpr)
    else:
        return new_jaxpr
