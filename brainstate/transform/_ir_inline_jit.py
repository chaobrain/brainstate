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
"""

from typing import Callable, Optional

from brainstate._compatible_import import Jaxpr, JaxprEqn, Literal, ClosedJaxpr


__all__ = [
    'inline_jit',
]


def inline_jit(
    jaxpr: Jaxpr | ClosedJaxpr,
    should_expand: Optional[Callable[[JaxprEqn], bool]] = None
) -> Jaxpr:
    """
    Rewrite a jaxpr by expanding (inlining) jit equations that satisfy the given condition.

    Args:
        jaxpr: The input jaxpr to rewrite
        should_expand: A predicate function that takes a JaxprEqn and returns True if
                      the jit should be expanded. If None, all jit equations are expanded.

    Returns:
        A new jaxpr with qualified jit equations expanded
    """
    if should_expand is None:
        should_expand = lambda eqn: True

    new_eqns = []
    var_mapping = {v: v for v in jaxpr.invars}

    for eqn in jaxpr.eqns:
        # Check if this is a jit primitive that should be expanded
        if eqn.primitive.name in ('pjit', 'jit', 'xla_call') and should_expand(eqn):
            # Get the jaxpr from the jit equation
            call_jaxpr = eqn.params.get('call_jaxpr')
            if call_jaxpr is None:
                # Fallback for different jit variants
                call_jaxpr = eqn.params.get('jaxpr')

            if call_jaxpr is not None:
                # Map input variables from outer scope to inner jaxpr
                inner_var_mapping = {}
                for inner_var, outer_var in zip(call_jaxpr.invars, eqn.invars):
                    mapped_var = var_mapping.get(outer_var, outer_var)
                    inner_var_mapping[inner_var] = mapped_var

                # Recursively expand the inner jaxpr
                expanded_inner = inline_jit(call_jaxpr, should_expand)

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
                            # Create new variable for outputs
                            new_var = v
                            inner_var_mapping[v] = new_var
                            new_outvars.append(new_var)

                    # Create the remapped equation
                    new_eqn = eqn.replace(
                        primitive=inner_eqn.primitive,
                        invars=new_invars,
                        outvars=new_outvars,
                        params=inner_eqn.params,
                        effects=inner_eqn.effects,
                        source_info=inner_eqn.source_info,
                        ctx=inner_eqn.ctx
                    )
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
    new_outvars = [var_mapping.get(v, v) for v in jaxpr.outvars]

    # Create the new jaxpr
    if isinstance(jaxpr, ClosedJaxpr):
        return jaxpr.replace(jaxpr=jaxpr.jaxpr.replace(eqns=new_eqns, outvars=new_outvars))
    elif isinstance(jaxpr, Jaxpr):
        return jaxpr.replace(eqns=new_eqns, outvars=new_outvars)
    else:
        raise ValueError(f"Unknown jaxpr type: {type(jaxpr)}")
