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

from typing import Sequence, Dict, List, Set
from collections import defaultdict
from jax.extend.core.primitives import dot_general_p, conv_general_dilated_p

from brainstate._compatible_import import is_jit_primitive, JaxprEqn, Jaxpr, ClosedJaxpr, Var, Literal
from brainstate._state import State


__all__ = [
    'eqns_to_closed_jaxpr',
    'eqns_to_jaxpr',
]


def eqns_to_jaxpr(
    eqns: Sequence[JaxprEqn],
    invars: Sequence[Var] = None,
    outvars: Sequence[Var] = None,
    constvars: Sequence[Var] = None,
) -> Jaxpr:
    """
    Convert a sequence of JaxprEqn into a Jaxpr.

    Args:
        eqns: Sequence of Jaxpr equations to convert
        invars: Input variables. If None, will be inferred from equations
        outvars: Output variables. If None, will be inferred from equations
        constvars: Constant variables. If None, will be automatically extracted from equations

    Returns:
        Jaxpr: A Jaxpr object constructed from the equations
    """
    # Collect all variables produced by equations
    produced_vars = set()
    for eqn in eqns:
        produced_vars.update(eqn.outvars)

    # Collect all variables used in equations (excluding Literals)
    used_vars_set = set()
    for eqn in eqns:
        for var in eqn.invars:
            if isinstance(var, Var):
                used_vars_set.add(var)

    # Infer invars if not provided
    if invars is None:
        # Variables that are used but not produced are potential invars or constvars
        invars = []
        for eqn in eqns:
            for var in eqn.invars:
                if isinstance(var, Var):
                    if var not in produced_vars and var not in invars:
                        invars.append(var)
    else:
        invars = list(invars)

    # Infer constvars if not provided
    # Constvars are variables used in equations but not in invars or produced_vars
    if constvars is None:
        invars_set = set(invars)
        constvars = []
        for var in used_vars_set:
            if var not in produced_vars and var not in invars_set:
                if var not in constvars:
                    constvars.append(var)
    else:
        constvars = list(constvars)

    # Infer outvars if not provided
    if outvars is None:
        # Variables that are produced but not consumed (or only consumed) are outputs
        consumed_vars = set()
        for eqn in eqns:
            for var in eqn.invars:
                if isinstance(var, Var) and var in produced_vars:
                    consumed_vars.add(var)

        outvars = list(produced_vars - consumed_vars)
    else:
        outvars = list(outvars)

    return Jaxpr(
        constvars=constvars,
        invars=invars,
        outvars=outvars,
        eqns=list(eqns),
    )


def eqns_to_closed_jaxpr(
    eqns: Sequence[JaxprEqn],
    invars: Sequence[Var] = None,
    outvars: Sequence[Var] = None,
    constvars: Sequence[Var] = None,
    consts: Sequence = None,
) -> ClosedJaxpr:
    """
    Convert a sequence of JaxprEqn into a ClosedJaxpr.

    Args:
        eqns: Sequence of Jaxpr equations to convert
        invars: Input variables. If None, will be inferred from equations
        outvars: Output variables. If None, will be inferred from equations
        constvars: Constant variables. If None, will be automatically extracted from equations
        consts: Constant values corresponding to constvars. If None, defaults to empty list

    Returns:
        ClosedJaxpr: A ClosedJaxpr object constructed from the equations

    Note:
        If constvars are automatically extracted from equations but no consts are provided,
        the resulting ClosedJaxpr will have empty consts list. This may cause runtime errors
        if the equations actually depend on these constants. In such cases, you should
        explicitly provide both constvars and consts from the original jaxpr.
    """
    # Create jaxpr (will automatically extract constvars if not provided)
    jaxpr = eqns_to_jaxpr(eqns, invars, outvars, constvars)

    # Handle consts
    if consts is None:
        # If no consts provided, create empty list
        # This is safe if there are no constvars, but may cause errors otherwise
        consts = []
    else:
        consts = list(consts)

    # Verify consts length matches constvars length
    if len(consts) != len(jaxpr.constvars):
        raise ValueError(
            f"consts length ({len(consts)}) does not match constvars length ({len(jaxpr.constvars)})"
        )

    return ClosedJaxpr(jaxpr, consts)
