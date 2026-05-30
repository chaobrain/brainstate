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
from brainstate.transform._ir_utils import IdentitySet, IRValidationError, check_all_vars


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

    Parameters
    ----------
    eqns
        Sequence of Jaxpr equations to convert
    invars
        Input variables. If None, will be inferred from equations
    outvars
        Output variables. If None, will be inferred from equations
    constvars
        Constant variables. If None, will be automatically extracted from equations

    Returns
    -------
    Jaxpr
        A Jaxpr object constructed from the equations
    """
    eqns = list(eqns)

    # Insertion-ordered set of variables produced by the equations.
    produced_vars = IdentitySet()
    for eqn in eqns:
        produced_vars.update(eqn.outvars)

    # Ordered list of used Vars (first-use order), de-duplicated by identity.
    # Using ordered structures (instead of plain ``set`` iteration) makes the
    # inferred invars/constvars/outvars deterministic across runs.
    used_order = []
    used_seen = IdentitySet()
    for eqn in eqns:
        for var in eqn.invars:
            if isinstance(var, Var) and var not in used_seen:
                used_seen.add(var)
                used_order.append(var)

    # Infer invars if not provided (used but not produced), else validate.
    if invars is None:
        invars = [v for v in used_order if v not in produced_vars]
    else:
        invars = list(invars)
        check_all_vars(invars, 'invars')
    invars_set = IdentitySet(invars)

    # Infer constvars if not provided (used, not produced, not an invar).
    if constvars is None:
        constvars = [v for v in used_order
                     if v not in produced_vars and v not in invars_set]
    else:
        constvars = list(constvars)
        check_all_vars(constvars, 'constvars')

    # Infer outvars if not provided: variables produced but never consumed,
    # in production order (deterministic). Otherwise validate the provided list.
    if outvars is None:
        consumed_vars = IdentitySet()
        for eqn in eqns:
            for var in eqn.invars:
                if isinstance(var, Var) and var in produced_vars:
                    consumed_vars.add(var)
        outvars = []
        for eqn in eqns:
            for v in eqn.outvars:
                if v not in consumed_vars:
                    outvars.append(v)
    else:
        outvars = list(outvars)
        check_all_vars(outvars, 'outvars')

    return Jaxpr(
        constvars=constvars,
        invars=invars,
        outvars=outvars,
        eqns=eqns,
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

    Parameters
    ----------
    eqns
        Sequence of Jaxpr equations to convert
    invars
        Input variables. If None, will be inferred from equations
    outvars
        Output variables. If None, will be inferred from equations
    constvars
        Constant variables. If None, will be automatically extracted from equations
    consts
        Constant values corresponding to constvars. If None, defaults to empty list

    Returns
    -------
    ClosedJaxpr
        A ClosedJaxpr object constructed from the equations

    Notes
    -----
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
        raise IRValidationError(
            f"consts length ({len(consts)}) does not match constvars length ({len(jaxpr.constvars)})"
        )

    return ClosedJaxpr(jaxpr, consts)
