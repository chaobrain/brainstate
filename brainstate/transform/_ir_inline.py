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
    from brainstate.transform._ir_utils import IRValidationError, make_var_factory

    if not isinstance(jaxpr, (Jaxpr, ClosedJaxpr)):
        raise IRValidationError(
            f"inline_jit expects a Jaxpr or ClosedJaxpr, got {type(jaxpr).__name__}."
        )

    if should_expand is None:
        should_expand = lambda eqn: True

    # Handle ClosedJaxpr by unwrapping to Jaxpr
    is_closed = isinstance(jaxpr, ClosedJaxpr)
    original_closed = jaxpr if is_closed else None
    inner_jaxpr = jaxpr.jaxpr if is_closed else jaxpr

    # Factory for brand-new, unique variables. Inlining a function more than
    # once must NOT reuse the callee's inner Var objects, or the resulting
    # jaxpr would bind the same variable twice (a correctness bug).
    _fresh = make_var_factory()

    new_eqns = []
    var_mapping = {v: v for v in inner_jaxpr.invars}

    # Constants lifted from inlined ClosedJaxprs are accumulated here and
    # attached to the enclosing scope so they remain bound.
    lifted_constvars = list(inner_jaxpr.constvars)
    lifted_consts = list(original_closed.consts) if is_closed else []

    def map_outer(v):
        if isinstance(v, Literal):
            return v
        return var_mapping.get(v, v)

    for eqn in inner_jaxpr.eqns:
        # Check if this is a jit primitive that should be expanded
        if is_jit_primitive(eqn) and should_expand(eqn):
            # Get the jaxpr from the jit equation (handle different jit variants)
            call_jaxpr = eqn.params.get('call_jaxpr')
            if call_jaxpr is None:
                call_jaxpr = eqn.params.get('jaxpr')

            if call_jaxpr is not None:
                # Recursively expand the inner jaxpr first; the result may be a
                # ClosedJaxpr carrying (possibly deeper-lifted) consts.
                expanded_inner = inline_jit(call_jaxpr, should_expand)
                if isinstance(expanded_inner, ClosedJaxpr):
                    inner_consts = list(expanded_inner.consts)
                    inner_actual = expanded_inner.jaxpr
                else:
                    inner_consts = []
                    inner_actual = expanded_inner

                # Map callee invars to the caller's (already-remapped) arguments.
                inner_var_mapping = {}
                for inner_var, outer_var in zip(inner_actual.invars, eqn.invars):
                    inner_var_mapping[inner_var] = map_outer(outer_var)

                # Lift the callee's constvars into the enclosing scope, each
                # under a fresh variable so they cannot collide.
                for cvar, cval in zip(inner_actual.constvars, inner_consts):
                    new_cvar = _fresh(cvar.aval)
                    inner_var_mapping[cvar] = new_cvar
                    lifted_constvars.append(new_cvar)
                    lifted_consts.append(cval)

                def remap(v):
                    # Literals pass through; every other inner variable gets a
                    # fresh binder the first time it is seen at this site.
                    if isinstance(v, Literal):
                        return v
                    if v not in inner_var_mapping:
                        inner_var_mapping[v] = _fresh(v.aval)
                    return inner_var_mapping[v]

                for inner_eqn in inner_actual.eqns:
                    new_invars = [remap(v) for v in inner_eqn.invars]
                    new_outvars = [remap(v) for v in inner_eqn.outvars]

                    replace_kwargs = {
                        'primitive': inner_eqn.primitive,
                        'invars': new_invars,
                        'outvars': new_outvars,
                        'params': inner_eqn.params,
                    }
                    if hasattr(inner_eqn, 'effects'):
                        replace_kwargs['effects'] = inner_eqn.effects
                    if hasattr(inner_eqn, 'source_info'):
                        replace_kwargs['source_info'] = inner_eqn.source_info

                    new_eqns.append(inner_eqn.replace(**replace_kwargs))

                # Map this equation's outputs to the (remapped) callee outputs.
                for inner_out, outer_out in zip(inner_actual.outvars, eqn.outvars):
                    var_mapping[outer_out] = remap(inner_out)

            else:
                # If we can't find the jaxpr, keep the original equation.
                new_eqns.append(eqn.replace(invars=[map_outer(v) for v in eqn.invars]))
                for v in eqn.outvars:
                    var_mapping[v] = v
        else:
            # Keep the equation as is, but remap its inputs.
            new_eqns.append(eqn.replace(invars=[map_outer(v) for v in eqn.invars]))
            for v in eqn.outvars:
                var_mapping[v] = v

    # Remap output variables
    new_outvars = [map_outer(v) for v in inner_jaxpr.outvars]

    # Create the new jaxpr, carrying any lifted constvars.
    new_jaxpr = inner_jaxpr.replace(
        eqns=new_eqns, outvars=new_outvars, constvars=lifted_constvars
    )

    # Return a ClosedJaxpr whenever we started closed OR nested consts were
    # lifted (the only correct way to keep those consts bound).
    if is_closed or lifted_consts:
        return ClosedJaxpr(new_jaxpr, lifted_consts)
    return new_jaxpr
