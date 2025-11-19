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
This module implements a compiler that transforms a ClosedJaxpr representation
of a spiking neural network (SNN) single-step update into structured computation
graph components (Groups, Projections, Inputs, Outputs).
"""

from collections import defaultdict
from functools import partial
from typing import Set
from typing import Tuple, Dict, Callable, List

import jax
from jax._src.core import ClosedJaxpr

from brainstate._compatible_import import Jaxpr, Var
from brainstate._compatible_import import JaxprEqn
from brainstate._state import State
from brainstate.transform._ir_inline import inline_jit
from brainstate.transform._ir_processing import eqns_to_closed_jaxpr
from brainstate.transform._make_jaxpr import StatefulFunction, get_arg_cache_key
from ._data import Graph, GraphElem, Group, Connection, Projection, Input, Output, CompiledGraph
from ._utils import _is_connection, UnionFind

__all__ = [
    'compile',
    'parse',
    'CompilationError',
]


class CompilationError(Exception):
    """Raised when the graph IR compiler cannot reconstruct a valid program."""
    pass


def _extract_consts_for_vars(
    constvars: List[Var],
    original_jaxpr: Jaxpr,
    original_consts: List,
) -> List:
    """Return the literal values that correspond to ``constvars``.

    Parameters
    ----------
    constvars : list[Var]
        Const variables that should be materialized in the derived ClosedJaxpr.
    original_jaxpr : Jaxpr
        Reference jaxpr that stores the canonical const ordering.
    original_consts : list
        Constants associated with ``original_jaxpr.constvars``.

    Returns
    -------
    list
        Constants aligned with ``constvars``.

    Raises
    ------
    CompilationError
        If a requested const variable cannot be located in ``original_jaxpr``.
    """
    if not constvars:
        return []

    # Build mapping from original constvars to consts
    constvar_to_const = dict(zip(original_jaxpr.constvars, original_consts))

    # Extract consts for the requested constvars
    consts = []
    for var in constvars:
        if var in constvar_to_const:
            consts.append(constvar_to_const[var])
        else:
            # This constvar is not in the original jaxpr, which shouldn't happen
            raise CompilationError(f"Constvar {var} not found in original jaxpr")

    return consts


def _make_closed_jaxpr(
    eqns: List[JaxprEqn],
    invars: List[Var],
    outvars: List[Var],
    original_jaxpr: Jaxpr,
    original_consts: List,
) -> ClosedJaxpr:
    """Create a ClosedJaxpr for a subset of equations.

    Parameters
    ----------
    eqns : list[JaxprEqn]
        Equations that should appear in the compact program.
    invars : list[Var]
        Variables that will serve as inputs to the new program.
    outvars : list[Var]
        Variables produced by ``eqns`` that form the program outputs.
    original_jaxpr : Jaxpr
        Full jaxpr from which ``eqns`` were sliced.
    original_consts : list
        Constants belonging to ``original_jaxpr``.

    Returns
    -------
    ClosedJaxpr
        The derived program plus the subset of literal constants it needs.

    Raises
    ------
    CompilationError
        Propagated when a referenced const variable no longer exists in the
        original program.
    """
    # Create ClosedJaxpr (will auto-extract constvars)
    closed_jaxpr = eqns_to_closed_jaxpr(eqns=eqns, invars=invars, outvars=outvars)

    # Extract corresponding consts for the auto-extracted constvars
    if closed_jaxpr.jaxpr.constvars:
        consts = _extract_consts_for_vars(
            closed_jaxpr.jaxpr.constvars, original_jaxpr, original_consts
        )
        return ClosedJaxpr(closed_jaxpr.jaxpr, consts)
    else:
        return closed_jaxpr


# ============================================================================
# State Grouping Analysis
# ============================================================================

def _step1_analyze_state_dependencies(
    jaxpr: Jaxpr,
    hidden_states: Tuple[State, ...],
    invar_to_state: Dict[Var, State],
    outvar_to_state: Dict[Var, State],
) -> List[Set[State]]:
    """Group hidden states that are mutually dependent via non-connection ops.

    Parameters
    ----------
    jaxpr : Jaxpr
        Program that updates every state in a single simulation step.
    hidden_states : tuple[State, ...]
        States that appear in both ``in_states`` and ``out_states``.
    invar_to_state : dict[Var, State]
        Mapping from input variables to their owning states.
    outvar_to_state : dict[Var, State]
        Mapping from output variables to their owning states.

    Returns
    -------
    list[set[State]]
        Sets of states that must be compiled into the same :class:`Group`.
    """
    # Create a mapping from state to its ID for efficient comparison
    state_to_id = {id(state): state for state in hidden_states}
    hidden_state_ids = set(state_to_id.keys())

    # Union-Find structure to track state grouping
    uf = UnionFind()
    for state in hidden_states:
        uf.make_set(id(state))

    # Track which output states depend on which input states (element-wise)
    # We'll analyze the computation graph to find element-wise dependencies

    # Build a mapping: output_var -> set of input_vars it depends on
    var_dependencies = _build_var_dependencies(jaxpr)

    # For each hidden state's output var, check which hidden state input vars it depends on
    for out_var in jaxpr.outvars:
        if out_var not in outvar_to_state:
            continue

        out_state = outvar_to_state[out_var]
        out_state_id = id(out_state)

        if out_state_id not in hidden_state_ids:
            continue

        # Get all input vars this output depends on
        dependent_vars = var_dependencies.get(out_var, set())

        for in_var in dependent_vars:
            if in_var not in invar_to_state:
                continue

            in_state = invar_to_state[in_var]
            in_state_id = id(in_state)

            if in_state_id not in hidden_state_ids:
                continue

            # If output state depends on input state, they should be in the same group
            # (assuming element-wise operations, which we verify by checking non-connection ops)
            if not _has_connection_between(jaxpr, in_var, out_var):
                uf.union(out_state_id, in_state_id)

    # Get the grouped states
    state_id_groups = uf.get_groups()
    state_groups = []
    for id_group in state_id_groups:
        state_group = {state_to_id[sid] for sid in id_group}
        state_groups.append(state_group)

    return state_groups


def _build_var_dependencies(jaxpr: Jaxpr) -> Dict[Var, Set[Var]]:
    """Compute the transitive input dependencies for every variable.

    Parameters
    ----------
    jaxpr : Jaxpr
        Program whose dependency graph should be analyzed.

    Returns
    -------
    dict[Var, set[Var]]
        Mapping from each variable in ``jaxpr`` to the set of input variables
        that influence it.
    """
    dependencies = {}

    # Input vars depend only on themselves
    for var in jaxpr.invars:
        dependencies[var] = {var}

    # Process equations in order
    for eqn in jaxpr.eqns:
        # Each output var depends on the union of dependencies of its input vars
        input_deps = set()
        for in_var in eqn.invars:
            if isinstance(in_var, Var):
                input_deps.update(dependencies.get(in_var, {in_var}))

        for out_var in eqn.outvars:
            dependencies[out_var] = input_deps.copy()

    return dependencies


def _has_connection_between(jaxpr: Jaxpr, in_var: Var, out_var: Var) -> bool:
    """Return True if a connection primitive lies between two variables.

    Parameters
    ----------
    jaxpr : Jaxpr
        Program containing the ``in_var`` → ``out_var`` path.
    in_var : Var
        Variable that serves as the path source.
    out_var : Var
        Variable that serves as the path sink.

    Returns
    -------
    bool
        ``True`` when the traversal encounters a connection equation, ``False``
        otherwise.
    """
    # Build forward dependency graph
    var_to_eqns = defaultdict(list)
    for eqn in jaxpr.eqns:
        for in_v in eqn.invars:
            if isinstance(in_v, Var):
                var_to_eqns[in_v].append(eqn)

    # BFS from in_var to out_var, looking for connection operations
    visited = set()
    queue = [in_var]

    while queue:
        current_var = queue.pop(0)
        if current_var in visited:
            continue
        visited.add(current_var)

        # If we reached the out_var, check if we went through a connection
        if current_var == out_var:
            # We need to check if there was a connection in the path
            # This requires a different approach - let's check backwards
            pass

        for eqn in var_to_eqns[current_var]:
            # Check if this equation is a connection
            if _is_connection(eqn):
                # If this connection produces a var that leads to out_var, return True
                for ov in eqn.outvars:
                    if ov == out_var:
                        return True
                    # Check if ov can reach out_var
                    if _can_reach(jaxpr, ov, out_var, var_to_eqns):
                        return True
            else:
                # Not a connection, continue searching
                for ov in eqn.outvars:
                    if ov not in visited:
                        queue.append(ov)

    return False


def _can_reach(jaxpr: Jaxpr, from_var: Var, to_var: Var, var_to_eqns: dict) -> bool:
    """Check whether ``from_var`` can reach ``to_var`` in the dataflow graph.

    Parameters
    ----------
    jaxpr : Jaxpr
        Program that specifies the equations.
    from_var : Var
        Starting variable for the reachability query.
    to_var : Var
        Destination variable for the query.
    var_to_eqns : dict
        Pre-built adjacency list that maps a variable to equations that consume
        it.

    Returns
    -------
    bool
        ``True`` if ``to_var`` is reachable, ``False`` otherwise.
    """
    if from_var == to_var:
        return True

    visited = set()
    queue = [from_var]

    while queue:
        current_var = queue.pop(0)
        if current_var in visited:
            continue
        visited.add(current_var)

        if current_var == to_var:
            return True

        for eqn in var_to_eqns.get(current_var, []):
            for ov in eqn.outvars:
                if ov not in visited:
                    queue.append(ov)

    return False


# ============================================================================
# Group Construction
# ============================================================================

def _step2_build_groups(
    jaxpr: Jaxpr,
    state_groups: List[Set[State]],
    in_states: Tuple[State, ...],
    out_states: Tuple[State, ...],
    invar_to_state: Dict[Var, State],
    outvar_to_state: Dict[Var, State],
    state_to_invars: Dict[State, Tuple[Var, ...]],
    state_to_outvars: Dict[State, Tuple[Var, ...]],
    original_consts: List,
) -> List[Group]:
    """Materialize :class:`Group` objects for each mutually dependent state set.

    Parameters
    ----------
    jaxpr : Jaxpr
        Source program to slice.
    state_groups : list[set[State]]
        Output of :func:`_step1_analyze_state_dependencies`.
    in_states, out_states : tuple[State, ...]
        Ordered input and output states for the full step.
    invar_to_state, outvar_to_state : dict[Var, State]
        Lookup tables between variables and their owning states.
    state_to_invars, state_to_outvars : dict[State, tuple[Var, ...]]
        Reverse mappings from states to the JAXPR variables that represent them.
    original_consts : list
        Constant literals to reuse when creating group jaxprs.

    Returns
    -------
    list[Group]
        One :class:`Group` per state cluster containing a ClosedJaxpr slice and
        metadata about its dependencies.
    """
    groups = []

    for state_group in state_groups:
        # Determine hidden_states, in_states, out_states for this group
        group_hidden_states = list(state_group)

        # Collect all input vars for these hidden states
        group_hidden_in_vars = []
        for state in group_hidden_states:
            group_hidden_in_vars.extend(state_to_invars.get(state))

        # Collect all output vars for these hidden states
        group_hidden_out_vars = []
        for state in group_hidden_states:
            group_hidden_out_vars.extend(state_to_outvars.get(state))

        # Find equations that produce these output vars
        relevant_eqns = []
        produced_vars = set(group_hidden_out_vars)
        queue = list(group_hidden_out_vars)
        # Track variables that come from connections (should be input_vars)
        connection_output_vars = set()

        # Backward traversal to find all equations needed to compute group outputs
        while queue:
            var = queue.pop(0)
            for eqn in jaxpr.eqns:
                if var in eqn.outvars and eqn not in relevant_eqns:
                    # Don't include connection equations in group
                    if _is_connection(eqn):
                        # Mark its outputs as connection outputs (input_vars for this group)
                        for out_var in eqn.outvars:
                            connection_output_vars.add(out_var)
                        # Don't traverse further through connections
                        continue

                    relevant_eqns.append(eqn)
                    # Add input vars to queue if not already processed
                    for in_var in eqn.invars:
                        if isinstance(in_var, Var) and in_var not in produced_vars:
                            produced_vars.add(in_var)
                            # Don't traverse beyond input state vars
                            if in_var not in group_hidden_in_vars:
                                queue.append(in_var)

        # Sort equations by their original order in jaxpr
        eqn_order = {id(eqn): i for i, eqn in enumerate(jaxpr.eqns)}
        relevant_eqns.sort(key=lambda e: eqn_order[id(e)])

        # Determine invars for the group jaxpr
        # Invars include: hidden state input vars + other input states + input currents
        group_invars = []
        group_in_states = []
        group_input_vars = []

        # First add hidden state input vars
        group_invars.extend(group_hidden_in_vars)

        # Find other required input vars
        required_vars = set()
        for eqn in relevant_eqns:
            for in_var in eqn.invars:
                if isinstance(in_var, Var):
                    required_vars.add(in_var)

        # Find vars produced by relevant equations (these are intermediate, not inputs)
        vars_produced_by_group = set()
        for eqn in relevant_eqns:
            for out_var in eqn.outvars:
                vars_produced_by_group.add(out_var)

        # Classify required vars
        for var in required_vars:
            if var in group_hidden_in_vars:
                continue  # Already added
            elif var in vars_produced_by_group:
                # This variable is produced by the group itself, not an input
                continue
            elif var in connection_output_vars:
                # This is a connection output (input_var for this group)
                if var not in group_input_vars:
                    group_input_vars.append(var)
                    group_invars.append(var)
            elif var in invar_to_state:
                # This is an input state (read-only)
                state = invar_to_state[var]
                if state not in group_hidden_states and state not in group_in_states:
                    group_in_states.append(state)
                    group_invars.append(var)
            else:
                # This is an external input variable (not a state, not a connection)
                if var not in group_input_vars:
                    group_input_vars.append(var)
                    group_invars.append(var)

        # Create the group ClosedJaxpr
        group_jaxpr = _make_closed_jaxpr(
            eqns=relevant_eqns,
            invars=group_invars,
            outvars=group_hidden_out_vars,
            original_jaxpr=jaxpr,
            original_consts=original_consts,
        )

        # Determine out_states (states produced but not consumed)
        group_out_states = []
        group_hidden_state_ids = {id(s) for s in group_hidden_states}
        for state in out_states:
            if id(state) not in group_hidden_state_ids:
                # Check if this group produces this state
                state_out_vars = state_to_outvars.get(state)
                if any(v in group_hidden_out_vars for v in state_out_vars):
                    group_out_states.append(state)
        del group_hidden_state_ids

        # Generate a name for this group based on its hidden states
        group_name = f"Group_{len(groups)}"

        group = Group(
            jaxpr=group_jaxpr,
            hidden_states=group_hidden_states,
            in_states=group_in_states,
            out_states=group_out_states,
            input_vars=group_input_vars,
            name=group_name,
        )
        groups.append(group)

    return groups


# ============================================================================
# Connection and Projection Analysis
# ============================================================================

def _step3_extract_connections(
    jaxpr: Jaxpr,
    original_consts: List,
) -> List[Tuple[JaxprEqn, Connection]]:
    """Identify connection equations and wrap them as :class:`Connection` objects.

    Parameters
    ----------
    jaxpr : Jaxpr
        Program to scan for connection primitives.
    original_consts : list
        Constant literals for materializing ClosedJaxpr objects.

    Returns
    -------
    list[tuple[JaxprEqn, Connection]]
        Pairs of the original equation and a :class:`Connection` wrapper that
        holds its ClosedJaxpr slice.
    """
    connections = []
    for eqn in jaxpr.eqns:
        if _is_connection(eqn):
            # Create a ClosedJaxpr for this connection
            conn_jaxpr = _make_closed_jaxpr(
                eqns=[eqn],
                invars=list(eqn.invars),
                outvars=list(eqn.outvars),
                original_jaxpr=jaxpr,
                original_consts=original_consts,
            )
            connection = Connection(jaxpr=conn_jaxpr)
            connections.append((eqn, connection))
    return connections


def _step4_build_projections(
    jaxpr: Jaxpr,
    groups: List[Group],
    connections: List[Tuple[JaxprEqn, Connection]],
    invar_to_state: Dict[Var, State],
    state_to_invars: Dict[State, Tuple[Var, ...]],
    original_consts: List,
) -> List[Projection]:
    """Create :class:`Projection` objects that ferry spikes between groups.

    Parameters
    ----------
    jaxpr : Jaxpr | ClosedJaxpr
        Full program that contains the per-projection subgraphs.
    groups : list[Group]
        Groups created in :func:`_step2_build_groups`.
    connections : list[tuple[JaxprEqn, Connection]]
        Connection equations identified by :func:`_step3_extract_connections`.
    invar_to_state : dict[Var, State]
        Mapping from input variables to the states they represent.
    state_to_invars : dict[State, tuple[Var, ...]]
        Variables that correspond to a given state on the input boundary.
    original_consts : list
        Constant literals for building ClosedJaxpr objects.

    Returns
    -------
    list[Projection]
        Projection descriptors that own the equations/connection metadata for
        a pre→post group path.
    """
    projections = []

    # Extract actual jaxpr and consts
    if isinstance(jaxpr, ClosedJaxpr):
        actual_jaxpr = jaxpr.jaxpr
    else:
        actual_jaxpr = jaxpr

    # Build a mapping: var -> equation that produces it
    var_to_producer_eqn = {}
    for eqn in actual_jaxpr.eqns:
        for out_var in eqn.outvars:
            var_to_producer_eqn[out_var] = eqn

    # Build a mapping: group_id -> set of input_vars consumed by the group
    group_to_input_vars = {}
    for group in groups:
        group_to_input_vars[id(group)] = set(group.input_vars)

    # For each group (as post_group), trace back its input_vars to find projections
    for post_group in groups:
        if not post_group.input_vars:
            continue

        # For each input_var of this post_group, trace back to find the source
        for input_var in post_group.input_vars:
            # Skip if this input_var is not produced by any equation
            # (it might be an external input)
            if input_var not in var_to_producer_eqn:
                continue

            # Trace back from input_var to find:
            # 1. All equations involved in producing this input_var
            # 2. The source group (pre_group) whose hidden_states are used
            # 3. Any connection operations involved

            proj_info = _trace_projection_path(
                input_var=input_var,
                post_group=post_group,
                groups=groups,
                jaxpr=jaxpr,
                var_to_producer_eqn=var_to_producer_eqn,
                invar_to_state=invar_to_state,
                connections=connections,
            )

            if proj_info is None:
                # No valid projection found (might be from external input)
                continue

            pre_group, proj_eqns, proj_hidden_states, proj_in_states, proj_connections = proj_info

            # Check if we already have a projection from pre_group to post_group
            # If so, merge the equations
            existing_proj = None
            for proj in projections:
                if proj.pre_group == pre_group and proj.post_group == post_group:
                    existing_proj = proj
                    break

            if existing_proj is not None:
                # Merge equations and states into existing projection
                # We need to reconstruct the projection with merged data
                merged_eqns = list(existing_proj.jaxpr.eqns)
                merged_hidden_states = list(existing_proj.hidden_states)
                merged_in_states = list(existing_proj.in_states)
                merged_connections = list(existing_proj.connections)

                for eqn in proj_eqns:
                    if eqn not in merged_eqns:
                        merged_eqns.append(eqn)

                for state in proj_hidden_states:
                    if state not in merged_hidden_states:
                        merged_hidden_states.append(state)

                for state in proj_in_states:
                    if state not in merged_in_states:
                        merged_in_states.append(state)

                for conn in proj_connections:
                    if conn not in merged_connections:
                        merged_connections.append(conn)

                # Sort equations by original order
                eqn_order = {id(eqn): i for i, eqn in enumerate(jaxpr.eqns)}
                merged_eqns.sort(key=lambda e: eqn_order[id(e)])

                # Collect all invars needed by merged_eqns
                proj_invars_needed = set()
                proj_produced_vars = set()

                # First collect all vars produced by merged_eqns
                for eqn in merged_eqns:
                    for out_var in eqn.outvars:
                        proj_produced_vars.add(out_var)

                # Then collect all vars needed but not produced
                for eqn in merged_eqns:
                    for in_var in eqn.invars:
                        if isinstance(in_var, Var) and in_var not in proj_produced_vars:
                            proj_invars_needed.add(in_var)

                # Convert state invars to actual vars
                proj_invars = []
                # First add hidden_states
                for state in merged_hidden_states:
                    for var in state_to_invars[state]:
                        if var in proj_invars_needed:
                            proj_invars.append(var)
                            proj_invars_needed.remove(var)
                # Then add in_states
                for state in merged_in_states:
                    for var in state_to_invars[state]:
                        if var in proj_invars_needed:
                            proj_invars.append(var)
                            proj_invars_needed.remove(var)

                # Add any remaining needed vars that aren't from states
                proj_invars.extend(sorted(proj_invars_needed, key=lambda v: str(v)))

                # Build outvars from merged equations
                proj_outvars = []
                for eqn in merged_eqns:
                    for out_var in eqn.outvars:
                        if out_var in group_to_input_vars[id(post_group)]:
                            if out_var not in proj_outvars:
                                proj_outvars.append(out_var)

                # Create new ClosedJaxpr
                proj_jaxpr = _make_closed_jaxpr(
                    eqns=merged_eqns,
                    invars=proj_invars,
                    outvars=proj_outvars,
                    original_jaxpr=jaxpr,
                    original_consts=original_consts,
                )

                # Replace existing projection
                new_proj = Projection(
                    hidden_states=merged_hidden_states,
                    in_states=merged_in_states,
                    jaxpr=proj_jaxpr,
                    connections=merged_connections,
                    pre_group=pre_group,
                    post_group=post_group,
                )
                projections[projections.index(existing_proj)] = new_proj

            else:
                # Create new projection
                # Collect all invars needed by proj_eqns
                proj_invars_needed = set()
                proj_produced_vars = set()

                # First collect all vars produced by proj_eqns
                for eqn in proj_eqns:
                    for out_var in eqn.outvars:
                        proj_produced_vars.add(out_var)

                # Then collect all vars needed but not produced
                for eqn in proj_eqns:
                    for in_var in eqn.invars:
                        if isinstance(in_var, Var) and in_var not in proj_produced_vars:
                            proj_invars_needed.add(in_var)

                # Convert state invars to actual vars
                proj_invars = []
                # First add hidden_states
                for state in proj_hidden_states:
                    for var in state_to_invars[state]:
                        if var in proj_invars_needed:
                            proj_invars.append(var)
                            proj_invars_needed.remove(var)
                # Then add in_states
                for state in proj_in_states:
                    for var in state_to_invars[state]:
                        if var in proj_invars_needed:
                            proj_invars.append(var)
                            proj_invars_needed.remove(var)

                # Add any remaining needed vars that aren't from states
                proj_invars.extend(sorted(proj_invars_needed, key=lambda v: str(v)))

                proj_outvars = []
                for eqn in proj_eqns:
                    for out_var in eqn.outvars:
                        if out_var in group_to_input_vars[id(post_group)]:
                            if out_var not in proj_outvars:
                                proj_outvars.append(out_var)

                proj_jaxpr = _make_closed_jaxpr(
                    eqns=proj_eqns,
                    invars=proj_invars,
                    outvars=proj_outvars,
                    original_jaxpr=jaxpr,
                    original_consts=original_consts,
                )

                projection = Projection(
                    hidden_states=proj_hidden_states,
                    in_states=proj_in_states,
                    jaxpr=proj_jaxpr,
                    connections=proj_connections,
                    pre_group=pre_group,
                    post_group=post_group,
                )
                projections.append(projection)

    return projections


def _trace_projection_path(
    input_var: Var,
    post_group: Group,
    groups: List[Group],
    jaxpr: Jaxpr | ClosedJaxpr,
    var_to_producer_eqn: Dict[Var, JaxprEqn],
    invar_to_state: Dict[Var, State],
    connections: List[Tuple[JaxprEqn, Connection]],
) -> Tuple[Group, List[JaxprEqn], List[State], List[State], List[Connection]] | None:
    """Trace the computation that produces ``input_var`` for a group.

    Parameters
    ----------
    input_var : Var
        Variable that appears in ``post_group.input_vars``.
    post_group : Group
        Group that consumes ``input_var``.
    groups : list[Group]
        All constructed groups, used to locate the source group.
    jaxpr : Jaxpr | ClosedJaxpr
        Original program that defines the dataflow.
    var_to_producer_eqn : dict[Var, JaxprEqn]
        Lookup table mapping a variable to the equation that produces it.
    invar_to_state : dict[Var, State]
        Mapping from state input variables to their owning states.
    connections : list[tuple[JaxprEqn, Connection]]
        Known connection equations along with their wrapped metadata.

    Returns
    -------
    tuple[Group, list[JaxprEqn], list[State], list[State], list[Connection]] | None
        The source group plus the traced equations, hidden states, input states,
        and explicit :class:`Connection` objects. ``None`` is returned when the
        path cannot be traced unambiguously.
    """
    if isinstance(jaxpr, ClosedJaxpr):
        jaxpr = jaxpr.jaxpr

    # Build connection equation set for quick lookup
    conn_eqns = {id(eqn): conn for eqn, conn in connections}

    # Trace back from input_var to collect all equations
    visited_vars = set()
    visited_eqns = set()
    proj_eqns = []
    proj_hidden_states = []
    proj_in_states = []
    proj_connections = []
    queue = [input_var]

    # Track which group is the source
    source_groups = []

    while queue:
        var = queue.pop(0)
        if var in visited_vars:
            continue
        visited_vars.add(var)

        # Check if this var is a jaxpr invar (boundary of tracing)
        if var in jaxpr.invars:
            # Check if it's a state var
            if var in invar_to_state:
                state = invar_to_state[var]
                # Find which group has this state
                source_found = False
                for group in groups:
                    if group.has_hidden_state(state):
                        if group not in source_groups:
                            source_groups.append(group)
                        if state not in proj_hidden_states:
                            proj_hidden_states.append(state)
                        source_found = True
                        break

                # If not in any group's hidden_states, it's an in_state for the projection
                if not source_found and state not in proj_in_states:
                    proj_in_states.append(state)
            continue

        # Find the equation that produces this var
        if var not in var_to_producer_eqn:
            continue

        eqn = var_to_producer_eqn[var]
        eqn_id = id(eqn)

        if eqn_id in visited_eqns:
            continue
        visited_eqns.add(eqn_id)

        # Add this equation to projection
        proj_eqns.append(eqn)

        # Check if this is a connection equation
        if eqn_id in conn_eqns:
            proj_connections.append(conn_eqns[eqn_id])

        # Add input vars of this equation to queue
        for in_var in eqn.invars:
            if isinstance(in_var, Var) and in_var not in visited_vars:
                queue.append(in_var)

    # Validate: should have exactly one source group
    if len(source_groups) != 1:
        return None

    pre_group = source_groups[0]

    # Don't create projection if pre_group == post_group (internal connection)
    if pre_group == post_group:
        return None

    # Sort equations by original order
    eqn_order = {id(eqn): i for i, eqn in enumerate(jaxpr.eqns)}
    proj_eqns.sort(key=lambda e: eqn_order[id(e)])

    return pre_group, proj_eqns, proj_hidden_states, proj_in_states, proj_connections


# ============================================================================
# Input and Output Analysis
# ============================================================================

def _trace_input_forward(
    input_var: Var,
    jaxpr: Jaxpr,
    groups: List[Group],
    var_to_consumer_eqns: Dict[Var, List[JaxprEqn]],
    group_input_vars_sets: Dict[int, Set[Var]],
) -> Tuple[Var, Group, List[JaxprEqn], List[Var]] | None:
    """Forward-trace ``input_var`` until its values flow into a group boundary.

    Parameters
    ----------
    input_var : Var
        Variable from ``jaxpr.invars`` that is not tied to a state.
    jaxpr : Jaxpr
        Program used for the reachability search.
    groups : list[Group]
        Existing groups to test for boundary conditions.
    var_to_consumer_eqns : dict[Var, list[JaxprEqn]]
        Mapping from a variable to the equations that consume it.
    group_input_vars_sets : dict[int, set[Var]]
        Cached sets of input variables for each group (keyed by ``id(group)``).

    Returns
    -------
    tuple[Var, Group, list[JaxprEqn], list[Var]] | None
        Metadata describing how the input flows into exactly one group or
        ``None`` when no group consumes this variable.
    """
    visited_vars = set()
    visited_eqns = set()
    equations_in_path = []

    # Current frontier of variables being traced
    current_frontier = {input_var}

    while current_frontier:
        # Check if all vars in current frontier belong to a single group's input_vars
        target_group = None
        for group in groups:
            group_input_set = group_input_vars_sets[id(group)]
            if all(var in group_input_set for var in current_frontier):
                # All frontier vars are input_vars of this group (stopping condition)
                target_group = group
                break

        if target_group is not None:
            # Stopping condition met: all outvars are invars of this group
            return input_var, target_group, equations_in_path, list(current_frontier)

        # Expand frontier
        next_frontier = set()

        for var in current_frontier:
            if var in visited_vars:
                continue
            visited_vars.add(var)

            # Get equations that consume this var
            consumer_eqns = var_to_consumer_eqns.get(var, [])

            for eqn in consumer_eqns:
                eqn_id = id(eqn)
                if eqn_id in visited_eqns:
                    continue
                visited_eqns.add(eqn_id)

                # Add this equation to the path
                equations_in_path.append(eqn)

                # Add output vars to next frontier
                for out_var in eqn.outvars:
                    next_frontier.add(out_var)

        current_frontier = next_frontier

    # No group found (input doesn't flow into any group)
    return None


def _step5_build_inputs(
    jaxpr: Jaxpr,
    groups: List[Group],
    invar_to_state: Dict[Var, State],
    original_consts: List,
) -> List[Input]:
    """Create :class:`Input` descriptors for external variables.

    Parameters
    ----------
    jaxpr : Jaxpr
        Program whose external inputs are being tracked.
    groups : list[Group]
        Group descriptors receiving the inputs.
    invar_to_state : dict[Var, State]
        Mapping used to filter out state variables (which should not become
        :class:`Input` objects).
    original_consts : list
        Constants required for building ClosedJaxpr objects.

    Returns
    -------
    list[Input]
        Input descriptors grouped by their destination group.
    """

    # Determine which vars are input_variables (not state vars)
    input_vars = []
    for var in jaxpr.invars:
        if var not in invar_to_state:
            input_vars.append(var)

    if not input_vars:
        return []

    # Build a mapping: var -> equations that consume it
    var_to_consumer_eqns = defaultdict(list)
    for eqn in jaxpr.eqns:
        for in_var in eqn.invars:
            if isinstance(in_var, Var):
                var_to_consumer_eqns[in_var].append(eqn)

    # Build a mapping: group -> set of input_vars (for quick lookup)
    group_input_vars_sets = {}
    for group in groups:
        group_input_vars_sets[id(group)] = set(group.input_vars)

    # For each input_var, trace forward to find which group(s) it flows into
    input_traces = []  # List of (input_var, target_group, equations, outvars)

    for input_var in input_vars:
        # Forward trace from this input_var
        trace_result = _trace_input_forward(
            input_var=input_var,
            jaxpr=jaxpr,
            groups=groups,
            var_to_consumer_eqns=var_to_consumer_eqns,
            group_input_vars_sets=group_input_vars_sets,
        )

        if trace_result is not None:
            input_traces.append(trace_result)

    # Group traces by target group (use group id as key)
    group_id_to_traces = defaultdict(list)
    id_to_group = {}

    for input_var, target_group, equations, outvars in input_traces:
        group_id = id(target_group)
        id_to_group[group_id] = target_group
        group_id_to_traces[group_id].append((input_var, equations, outvars))

    # Create Input objects for each group
    # Note: A group can have multiple Inputs (one for each input_var or set of input_vars)
    inputs = []

    for group_id, traces in group_id_to_traces.items():
        group = id_to_group[group_id]

        # Collect all input vars, equations, and output vars for this group
        all_input_vars = []
        all_equations = []
        all_output_vars = []

        for input_var, equations, outvars in traces:
            if input_var not in all_input_vars:
                all_input_vars.append(input_var)
            for eqn in equations:
                if eqn not in all_equations:
                    all_equations.append(eqn)
            for var in outvars:
                if var not in all_output_vars:
                    all_output_vars.append(var)

        # Sort equations by their original order in jaxpr
        eqn_order = {id(eqn): i for i, eqn in enumerate(jaxpr.eqns)}
        all_equations.sort(key=lambda e: eqn_order[id(e)])

        # Create the input ClosedJaxpr
        input_jaxpr = _make_closed_jaxpr(
            eqns=all_equations,
            invars=all_input_vars,
            outvars=all_output_vars,
            original_jaxpr=jaxpr,
            original_consts=original_consts,
        )

        input_obj = Input(
            jaxpr=input_jaxpr,
            group=group,
        )
        inputs.append(input_obj)

    return inputs


def _step6_build_outputs(
    jaxpr: Jaxpr,
    groups: List[Group],
    outvar_to_state: Dict[Var, State],
    state_to_outvars: Dict[State, Tuple[Var, ...]],
    invar_to_state: Dict[Var, State],
    state_to_invars: Dict[State, Tuple[Var, ...]],
    original_consts: List,
) -> List[Output]:
    """Describe how model outputs are assembled from group state variables.

    Parameters
    ----------
    jaxpr : Jaxpr
        Program whose ``outvars`` correspond to either user outputs or updated
        states.
    groups : list[Group]
        Groups that may contribute to outputs.
    outvar_to_state : dict[Var, State]
        Mapping from output variables to their state.
    state_to_outvars : dict[State, tuple[Var, ...]]
        Reverse mapping from states to the variables that encode them.
    invar_to_state : dict[Var, State]
        Lookup for state input variables.
    state_to_invars : dict[State, tuple[Var, ...]]
        Reverse mapping from states to their input variables.
    original_consts : list
        Constants required for building ClosedJaxpr objects.

    Returns
    -------
    list[Output]
        Output descriptors paired with the responsible group.

    Raises
    ------
    CompilationError
        If an output depends on unsupported intermediates or multiple groups.
    """

    # Identify state outvars (variables that correspond to state outputs)
    state_outvars_set = set([v for outvars in state_to_outvars.values() for v in outvars])

    # Identify state invars (variables that correspond to state inputs)
    state_invars_set = set(invar_to_state.keys())

    # Get output_vars (jaxpr outvars that are not state outvars)
    output_vars = [v for v in jaxpr.outvars if v not in state_outvars_set]

    if not output_vars:
        return []

    # Build a mapping: var -> equation that produces it
    var_to_producer_eqn = {}
    for eqn in jaxpr.eqns:
        for out_var in eqn.outvars:
            var_to_producer_eqn[out_var] = eqn

    # Group output vars by which group they depend on (use group id as key)
    group_id_output_mapping = defaultdict(list)
    id_to_group = {}

    # For each output_var, backward trace to find dependencies
    for out_var in output_vars:
        # Backward trace from out_var
        dependent_state_outvars = []
        dependent_state_invars = []
        equations_needed = []

        # Use worklist algorithm for backward tracing
        visited_vars = set()
        worklist = [out_var]

        while worklist:
            var = worklist.pop()

            if var in visited_vars:
                continue
            visited_vars.add(var)

            # Stopping condition 1: this is a state outvar
            if var in state_outvars_set:
                if var not in dependent_state_outvars:
                    dependent_state_outvars.append(var)
                continue

            # Stopping condition 2: this is a state invar
            if var in state_invars_set:
                if var not in dependent_state_invars:
                    dependent_state_invars.append(var)
                continue

            # If this var is not produced by any equation, it must be a jaxpr invar
            if var not in var_to_producer_eqn:
                if var in jaxpr.invars:
                    # This is an external input (not a state)
                    raise CompilationError(
                        f"Output variable {out_var} depends on external input {var}, "
                        "which is not a state variable. Outputs must only depend on state variables."
                    )
                else:
                    raise CompilationError(
                        f"Output variable {out_var} depends on unknown variable {var}"
                    )

            # Get the equation that produces this var and add it
            eqn = var_to_producer_eqn[var]
            if eqn not in equations_needed:
                equations_needed.append(eqn)

            # Add all input vars of this equation to the worklist for further tracing
            for in_var in eqn.invars:
                if isinstance(in_var, Var) and in_var not in visited_vars:
                    worklist.append(in_var)

        # Verify that all inputs to equations_needed are either:
        # 1. state_outvars
        # 2. state_invars
        # 3. produced by one of the equations_needed (intermediate vars within this output computation)

        # Collect all vars produced by equations_needed
        produced_vars = set()
        for eqn in equations_needed:
            for out_v in eqn.outvars:
                produced_vars.add(out_v)

        # Check all inputs to ensure no invalid dependencies
        for eqn in equations_needed:
            for in_var in eqn.invars:
                if isinstance(in_var, Var):
                    # Check if it's a state outvar or state invar (valid boundary)
                    if in_var in state_outvars_set or in_var in state_invars_set:
                        continue
                    # Check if it's produced by one of our equations (valid intermediate)
                    if in_var in produced_vars:
                        continue
                    # If we get here, it's an invalid dependency on external intermediate variable
                    raise CompilationError(
                        f"Output variable {out_var} depends on intermediate variable {in_var} "
                        "that is not a state variable and not produced by output equations. "
                        "This indicates an invalid output computation path."
                    )

        # Find corresponding hidden_states and in_states from dependent state vars
        dependent_hidden_states = []
        dependent_in_states = []
        dependent_groups = []

        # Process state outvars to find hidden states
        for var in dependent_state_outvars:
            if var in outvar_to_state:
                state = outvar_to_state[var]
                # Check if this is a hidden state (in some group)
                for group in groups:
                    if group.has_hidden_state(state):
                        if group not in dependent_groups:
                            dependent_groups.append(group)
                        if state not in dependent_hidden_states:
                            dependent_hidden_states.append(state)
                        break

        # Process state invars to find hidden states or in states
        for var in dependent_state_invars:
            if var in invar_to_state:
                state = invar_to_state[var]
                # Check if this is a hidden state
                found = False
                for group in groups:
                    if group.has_hidden_state(state):
                        if group not in dependent_groups:
                            dependent_groups.append(group)
                        if state not in dependent_hidden_states:
                            dependent_hidden_states.append(state)
                        found = True
                        break

                # If not a hidden state, it's an in_state
                if not found:
                    if state not in dependent_in_states:
                        dependent_in_states.append(state)

        # Validate: each output should depend on exactly one group
        if len(dependent_groups) == 0:
            raise CompilationError(
                f"Output variable {out_var} does not depend on any group"
            )
        elif len(dependent_groups) > 1:
            raise CompilationError(
                f"Output variable {out_var} depends on multiple groups: {dependent_groups}. "
                "Each output should depend on only one group."
            )

        group = dependent_groups[0]
        group_id = id(group)
        id_to_group[group_id] = group
        group_id_output_mapping[group_id].append((
            out_var,
            dependent_hidden_states,
            dependent_in_states,
            equations_needed
        ))

    # Create Output objects for each group
    outputs = []
    for group_id, output_info in group_id_output_mapping.items():
        group = id_to_group[group_id]
        output_vars_for_group = [ov for ov, _, _, _ in output_info]
        all_dependent_hidden_states = []
        all_dependent_in_states = []
        all_equations = []

        # Collect all states and equations
        for _, hidden_states, in_states, equations in output_info:
            for state in hidden_states:
                if state not in all_dependent_hidden_states:
                    all_dependent_hidden_states.append(state)
            for state in in_states:
                if state not in all_dependent_in_states:
                    all_dependent_in_states.append(state)
            for eqn in equations:
                if eqn not in all_equations:
                    all_equations.append(eqn)

        # Sort equations by their original order in jaxpr
        eqn_order = {id(eqn): i for i, eqn in enumerate(jaxpr.eqns)}
        all_equations.sort(key=lambda e: eqn_order[id(e)])

        # Determine invars for the output jaxpr
        # We need to use state outvars for hidden states and state invars for in states
        output_invars = []

        # Add hidden state outvars
        for state in all_dependent_hidden_states:
            output_invars.extend(state_to_outvars[state])

        # Add in state invars
        for state in all_dependent_in_states:
            output_invars.extend(state_to_invars[state])

        # Create the output ClosedJaxpr
        output_jaxpr = _make_closed_jaxpr(
            eqns=all_equations,
            invars=output_invars,
            outvars=output_vars_for_group,
            original_jaxpr=jaxpr,
            original_consts=original_consts,
        )

        output_obj = Output(
            jaxpr=output_jaxpr,
            hidden_states=all_dependent_hidden_states,
            in_states=all_dependent_in_states,
            group=group,
        )
        outputs.append(output_obj)

    return outputs


# ============================================================================
# Call Order Analysis
# ============================================================================

def _step7_build_graph(
    jaxpr: Jaxpr,
    groups: List[Group],
    projections: List[Projection],
    inputs: List[Input],
    outputs: List[Output],
) -> Graph:
    """Derive an execution graph that preserves the original equation order.

    Parameters
    ----------
    jaxpr : Jaxpr
        Program that defines the canonical ordering.
    groups : list[Group]
        Computation blocks that produce state updates.
    projections : list[Projection]
        Connection pipelines between groups.
    inputs : list[Input]
        External inputs to the network.
    outputs : list[Output]
        Objects describing how observable values are extracted.

    Returns
    -------
    Graph
        Directed acyclic graph with nodes ordered for execution/visualization.
    """
    call_graph = Graph()

    # Ensure inputs come first
    for inp in inputs:
        call_graph.add_node(inp)

    # Create a mapping from equations/vars to components
    eqn_to_component: Dict[int, GraphElem] = {}
    var_to_component: Dict[Var, GraphElem] = {}

    for group in groups:
        for eqn in group.jaxpr.eqns:
            eqn_to_component[id(eqn)] = group
    for proj in projections:
        for eqn in proj.jaxpr.eqns:
            eqn_to_component[id(eqn)] = proj

    # Process equations in order and add components + edges
    seen_components = set()
    for eqn in jaxpr.eqns:
        component = eqn_to_component.get(id(eqn))
        if component is None:
            continue

        component_id = id(component)
        if component_id not in seen_components:
            seen_components.add(component_id)
            call_graph.add_node(component)

        # Link dependencies based on variable producers
        for in_var in eqn.invars:
            if isinstance(in_var, Var):
                producer = var_to_component.get(in_var)
                if producer is not None and producer is not component:
                    call_graph.add_edge(producer, component)

        for out_var in eqn.outvars:
            if isinstance(out_var, Var):
                var_to_component[out_var] = component

    # Outputs are appended to maintain display parity with previous behavior
    for out in outputs:
        call_graph.add_node(out)

    # Structural dependencies derived from graph metadata
    for inp in inputs:
        call_graph.add_edge(inp, inp.group)
    for proj in projections:
        call_graph.add_edge(proj.pre_group, proj)
        call_graph.add_edge(proj, proj.post_group)
    for out in outputs:
        call_graph.add_edge(out.group, out)

    return call_graph


def _step8_validate_compilation(
    closed_jaxpr: ClosedJaxpr,
    groups: List[Group],
    projections: List[Projection],
    inputs: List[Input],
    outputs: List[Output],
    in_states: Tuple[State, ...],
    out_states: Tuple[State, ...],
    invar_to_state: Dict[Var, State],
):
    """Run structural checks on the assembled :class:`CompiledGraph`.

    Parameters
    ----------
    closed_jaxpr : ClosedJaxpr
        Original program that was compiled.
    groups, projections, inputs, outputs : list
        Components produced by previous compilation phases.
    in_states, out_states : tuple[State, ...]
        Lists of states supplied by the caller.
    invar_to_state : dict[Var, State]
        Mapping from input variables to states, used for validation.

    Raises
    ------
    CompilationError
        If invariants such as “each hidden state belongs to a group” are
        violated.
    """
    jaxpr = closed_jaxpr.jaxpr

    # Check 1: All hidden states should be in some group
    all_group_hidden_states = set()
    for group in groups:
        for state in group.hidden_states:
            all_group_hidden_states.add(id(state))

    hidden_states = set(s for s in out_states if s in in_states)
    for state in hidden_states:
        if id(state) not in all_group_hidden_states:
            raise CompilationError(
                f"Hidden state {state} is not assigned to any group"
            )

    # Check 2: All input_variables should be used (relaxed check)
    # Note: Some input variables (like scalar time) may be used in intermediate
    # computations but not directly flow into Groups. This is acceptable.
    input_vars = [v for v in jaxpr.invars if v not in invar_to_state]
    used_input_vars = set()
    for inp in inputs:
        for var in inp.jaxpr.invars:
            used_input_vars.add(var)

    # Also collect input vars used by groups
    for group in groups:
        for var in group.input_vars:
            used_input_vars.add(var)

    # Check if any input var is used in the jaxpr at all
    # (This is a softer check - we just verify the var appears somewhere)
    for var in input_vars:
        if var not in used_input_vars:
            # Check if this var is used in any equation
            found = False
            for eqn in jaxpr.eqns:
                if var in eqn.invars:
                    found = True
                    break
            # Only warn if the variable is truly unused
            if not found:
                # Don't raise error, just skip - could be a constant or unused arg
                pass

    # Check 3: Projections should have non-empty connections
    for proj in projections:
        if not proj.connections:
            raise CompilationError(
                f"Projection from {proj.pre_group} to {proj.post_group} has no connections"
            )

    # Check 4: Projection hidden_states should belong to exactly one group
    for proj in projections:
        for state in proj.hidden_states:
            count = sum(1 for g in groups if g.has_hidden_state(state))
            if count == 0:
                raise CompilationError(
                    f"Projection depends on state {state} which is not in any group"
                )
            elif count > 1:
                raise CompilationError(
                    f"Projection depends on state {state} which belongs to multiple groups"
                )


def _build_state_mapping(
    closed_jaxpr: ClosedJaxpr,
    in_states: List[State],
    out_states: List[State],
) -> Dict:
    """Map JAXPR variables to their corresponding ``State`` instances.

    Parameters
    ----------
    closed_jaxpr : ClosedJaxpr
        Program emitted by ``StatefulFunction``.
    in_states, out_states : list[State]
        Ordered state lists returned by the stateful function.

    Returns
    -------
    dict
        Dictionary containing ``invar_to_state``, ``outvar_to_state``,
        ``state_to_invars``, ``state_to_outvars``, and the original state lists.

    Raises
    ------
    TypeError
        If ``closed_jaxpr`` is not a :class:`ClosedJaxpr` or the states are not
        :class:`State` instances.
    ValueError
        If ``out_states`` is not a subset of ``in_states``.
    """
    # --- validations ---
    if not isinstance(closed_jaxpr, ClosedJaxpr):
        raise TypeError(f"closed_jaxpr must be a ClosedJaxpr, got {type(closed_jaxpr)}")

    if not all(isinstance(s, State) for s in in_states):
        bad = [type(s) for s in in_states if not isinstance(s, State)]
        raise TypeError(f"in_states must contain only State instances, got {bad}")

    if not all(isinstance(s, State) for s in out_states):
        bad = [type(s) for s in out_states if not isinstance(s, State)]
        raise TypeError(f"out_states must contain only State instances, got {bad}")

    missing_out = [s for s in out_states if s not in in_states]
    if missing_out:
        raise ValueError(
            f"All out_states must be present in in_states. Missing: {[repr(s) for s in missing_out]}"
        )

    # empty initialization
    invar_to_state = dict()
    state_to_invars = dict()
    outvar_to_state = dict()
    state_to_outvars = dict()

    # Extract the actual jaxpr from ClosedJaxpr
    jaxpr = closed_jaxpr.jaxpr

    # input states <---> input variables #
    # ---------------------------------- #

    # Get state structure information
    in_state_vals = [state.value for state in in_states]
    in_state_avals, in_state_tree = jax.tree.flatten(in_state_vals)
    n_inp_before_states = len(jaxpr.invars) - len(in_state_avals)

    # Map state tree to invars and outvars
    # Input variables: the last len(state_avals) invars correspond to states
    state_tree_invars = jax.tree.unflatten(in_state_tree, jaxpr.invars[n_inp_before_states:])

    # Build mappings using the tree structure
    # This ensures proper correspondence between states and their JAXpr variables
    assert len(in_states) == len(state_tree_invars), "Mismatch between number of input states and state tree invars"
    for state, invar in zip(in_states, state_tree_invars):
        # Always flatten the tree structure to get individual variables
        invar_leaves = jax.tree.leaves(invar)

        # Store the relationships
        for var in invar_leaves:
            invar_to_state[var] = state

        # Store the reverse mappings
        state_to_invars[state] = invar_leaves

    # output states <---> output variables #
    # ------------------------------------ #

    # Get state structure information
    out_state_vals = [state.value for state in out_states]
    out_state_avals, out_state_tree = jax.tree.flatten(out_state_vals)
    n_out_before_states = len(jaxpr.outvars) - len(out_state_avals)

    # Output variables: after the main outputs, the rest correspond to state updates
    state_tree_outvars = jax.tree.unflatten(out_state_tree, jaxpr.outvars[n_out_before_states:])
    assert len(out_states) == len(state_tree_outvars), \
        'Mismatch between number of output states and state tree outvars'

    # Build mappings using the tree structure
    # This ensures proper correspondence between states and their JAXpr variables
    for state, outvar in zip(out_states, state_tree_outvars):
        # Always flatten the tree structure to get individual variables
        outvar_leaves = jax.tree.leaves(outvar)

        # Store the relationships
        for var in outvar_leaves:
            outvar_to_state[var] = state
        state_to_outvars[state] = outvar_leaves

    return {
        'invar_to_state': invar_to_state,
        'state_to_invars': state_to_invars,
        'outvar_to_state': outvar_to_state,
        'state_to_outvars': state_to_outvars,
        'in_states': in_states,
        'out_states': out_states,
        'hidden_states': [s for s in out_states],
    }


# ============================================================================
# Main Compilation Function
# ============================================================================

def compile(
    closed_jaxpr: ClosedJaxpr,
    in_states: Tuple[State, ...],
    out_states: Tuple[State, ...],
    invar_to_state: Dict[Var, State],
    outvar_to_state: Dict[Var, State],
    state_to_invars: Dict[State, Tuple[Var, ...]],
    state_to_outvars: Dict[State, Tuple[Var, ...]],
):
    """Compile a ClosedJaxpr single-step update into Graph IR containers.

    Parameters
    ----------
    closed_jaxpr : ClosedJaxpr
        Program produced by ``jax.make_jaxpr`` for a single simulation step.
    in_states, out_states : tuple[State, ...]
        Ordered state objects provided by the caller.
    invar_to_state, outvar_to_state : dict[Var, State]
        Helper mappings between program variables and states.
    state_to_invars, state_to_outvars : dict[State, tuple[Var, ...]]
        Reverse mappings needed to reconstruct per-state programs.


    Raises
    ------
    CompilationError
        If the ClosedJaxpr violates the IR assumptions (e.g. outputs depend on
        multiple groups).
    """
    jaxpr = closed_jaxpr.jaxpr
    consts = closed_jaxpr.consts

    # Determine hidden_states (states that are both input and output)
    hidden_states = tuple(s for s in out_states if s in in_states)

    # Step 1: Analyze state dependencies and group states
    state_groups = _step1_analyze_state_dependencies(
        jaxpr, hidden_states, invar_to_state, outvar_to_state
    )

    # Step 2: Build Group objects
    groups = _step2_build_groups(
        jaxpr, state_groups, in_states, out_states,
        invar_to_state, outvar_to_state, state_to_invars, state_to_outvars,
        consts
    )

    # Step 3: Extract connections
    connections = _step3_extract_connections(jaxpr, consts)

    # Step 4: Build Projection objects
    projections = _step4_build_projections(
        jaxpr, groups, connections, invar_to_state, state_to_invars, consts
    )

    # Step 5: Build Input objects
    inputs = _step5_build_inputs(jaxpr, groups, invar_to_state, consts)

    # Step 6: Build Output objects
    outputs = _step6_build_outputs(
        jaxpr, groups, outvar_to_state, state_to_outvars, invar_to_state, state_to_invars, consts
    )

    # Step 7: Determine call order
    call_graph = _step7_build_graph(jaxpr, groups, projections, inputs, outputs)

    # Step 8: Validate the compilation
    _step8_validate_compilation(
        closed_jaxpr, groups, projections, inputs, outputs,
        in_states, out_states, invar_to_state
    )

    return groups, projections, inputs, outputs, call_graph


def parse(
    stateful_fn: StatefulFunction,
    jit_inline: bool = True,
) -> Callable[..., CompiledGraph]:
    """Create a parser that compiles ``stateful_fn`` into graph IR.

    Parameters
    ----------
    stateful_fn : StatefulFunction
        Stateful function wrapper to parse.
    jit_inline : bool, optional
        When ``True`` the parser inlines JIT-wrapped connection primitives
        before compilation.

    Returns
    -------
    Callable[..., ParsedResults]
        Function that, when invoked with runtime arguments, returns
        :class:`ParsedResults`.
    """
    assert isinstance(stateful_fn, StatefulFunction), "stateful_fn must be an instance of StatefulFunction"
    assert stateful_fn.return_only_write, (
        "Parser currently only supports stateful functions that return only write states. "
    )

    def call(*args, **kwargs):
        """Run the parser for the provided arguments."""
        # jaxpr
        jaxpr = stateful_fn.get_jaxpr(*args, **kwargs)
        if jit_inline:
            jaxpr = inline_jit(jaxpr, _is_connection)

        # Build state mappings
        in_states = stateful_fn.get_states(*args, **kwargs)
        out_states = stateful_fn.get_write_states(*args, **kwargs)
        state_mapping = _build_state_mapping(jaxpr, in_states, out_states)

        # Compile the SNN
        groups, projections, inputs, outputs, graph = compile(
            closed_jaxpr=jaxpr,
            in_states=in_states,
            out_states=out_states,
            invar_to_state=state_mapping['invar_to_state'],
            outvar_to_state=state_mapping['outvar_to_state'],
            state_to_invars=state_mapping['state_to_invars'],
            state_to_outvars=state_mapping['state_to_outvars'],
        )
        cache_fn = partial(get_arg_cache_key, stateful_fn.static_argnums, stateful_fn.static_argnames)
        cache_key = stateful_fn.get_arg_cache_key(*args, **kwargs)

        return CompiledGraph(
            static_argnums=stateful_fn.static_argnums,
            static_argnames=stateful_fn.static_argnames,
            out_treedef=stateful_fn.get_out_treedef_by_cache(cache_key),
            cache_fn=cache_fn,
            cache_key=cache_key,
            jaxpr=jaxpr,
            in_states=in_states,
            out_states=out_states,
            invar_to_state=state_mapping['invar_to_state'],
            outvar_to_state=state_mapping['outvar_to_state'],
            state_to_invars=state_mapping['state_to_invars'],
            state_to_outvars=state_mapping['state_to_outvars'],
            groups=groups,
            projections=projections,
            inputs=inputs,
            outputs=outputs,
            graph=graph,
        )

    return call
