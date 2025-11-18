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

from typing import Sequence, Dict

from jax.extend.core.primitives import dot_general_p, conv_general_dilated_p

from brainstate._compatible_import import is_jit_primitive, JaxprEqn, Jaxpr, ClosedJaxpr, Var
from brainstate._state import State

__all__ = [
    'eqns_to_jaxpr',
    'eqns_to_closed_jaxpr',
]


def _is_connection(eqn: JaxprEqn) -> bool:
    """
    Determine if a JaxprEqn represents a connection operation.

    Connection operations include:
    1. JIT-wrapped brainevent operations (e.g., brainevent.binary_fixed_num_mv_p_call)
    2. Inline brainevent primitives (e.g., binary_fixed_num_mv)
    3. Standard matrix operations (dot_general, conv_general_dilated)

    Args:
        eqn: A JaxprEqn to check

    Returns:
        True if this equation represents a connection, False otherwise
    """
    # Check if equation is a jit-wrapped brainevent operation that should be a connection
    if is_jit_primitive(eqn):
        # Check if the function name starts with 'brainevent'
        if 'name' in eqn.params:
            name = eqn.params['name']
            if isinstance(name, str) and name.startswith('brainevent'):
                return True

    # Check for brainevent primitive names (after inline_jit)
    # These are the actual brainevent connection primitives
    primitive_name = eqn.primitive.name
    if any(keyword in primitive_name for keyword in [
        'binary_fixed_num_mv',
        'event_ell_mv',
        'event_csr_matvec',
        'event_coo_matvec',
        'taichi_mv',
        'mv_prob',
        'mv_',  # General matrix-vector operations from brainevent
    ]):
        return True

    # check for standard connection primitives
    if eqn.primitive in [
        dot_general_p,
        conv_general_dilated_p,
    ]:
        return True

    return False


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
        constvars: Constant variables. If None, defaults to empty list

    Returns:
        Jaxpr: A Jaxpr object constructed from the equations
    """
    if constvars is None:
        constvars = []

    produced_vars = set()
    used_outvars = set()
    # Infer invars if not provided
    if invars is None:
        # Collect all variables that are used but not produced by equations
        for eqn in eqns:
            produced_vars.update(eqn.outvars)

        used_vars = []
        for eqn in eqns:
            for var in eqn.invars:
                if isinstance(var, Var):
                    if var not in produced_vars and var not in used_vars:
                        used_vars.append(var)
                    if var in produced_vars:
                        used_outvars.add(var)

        invars = used_vars

    # Infer outvars if not provided
    if outvars is None:
        # Use the output variables of the last equation
        if eqns:
            outvars = list(produced_vars.difference(used_outvars))
        else:
            outvars = []

    return Jaxpr(
        constvars=list(constvars),
        invars=list(invars),
        outvars=list(outvars),
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
        constvars: Constant variables. If None, defaults to empty list
        consts: Constant values corresponding to constvars. If None, defaults to empty list

    Returns:
        ClosedJaxpr: A ClosedJaxpr object constructed from the equations
    """
    if consts is None:
        consts = []

    jaxpr = eqns_to_jaxpr(eqns, invars, outvars, constvars)
    return ClosedJaxpr(jaxpr, list(consts))


def find_in_states(
    in_var_to_state: Dict[Var, State],
    in_vars: Sequence[Var]
):
    in_states = []
    in_state_ids = set()
    for var in in_vars:
        if var in in_var_to_state:
            st = in_var_to_state[var]
            if id(st) not in in_state_ids:
                in_state_ids.add(id(st))
                in_states.append(st)
    return in_states


def find_out_states(
    out_var_to_state: Dict[Var, State],
    out_vars: Sequence[Var]
):
    out_states = []
    out_state_ids = set()
    for var in out_vars:
        if var in out_var_to_state:
            st = out_var_to_state[var]
            if id(st) not in out_state_ids:
                out_state_ids.add(id(st))
                out_states.append(st)
    return out_states
