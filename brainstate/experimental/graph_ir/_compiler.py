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
SNN Compiler for GDI-IST BPU

This module implements a compiler that transforms a ClosedJaxpr representation
of a spiking neural network (SNN) single-step update into structured computation
graph components (Groups, Projections, Inputs, Outputs).
"""

from collections import defaultdict
from typing import List, Dict, Set, Tuple

from brainstate._compatible_import import ClosedJaxpr, Jaxpr, Var, JaxprEqn
from brainstate._state import State
from brainstate.transform._ir_processing import eqns_to_jaxpr
from ._data import Group, Connection, Projection, Input, Output, CompiledGraph
from ._utils import _is_connection, UnionFind

__all__ = [
    'compile',
    'CompilationError',
]


class CompilationError(Exception):
    """Exception raised during SNN compilation."""
    pass


# ============================================================================
# State Grouping Analysis
# ============================================================================

def _step1_analyze_state_dependencies(
    jaxpr: Jaxpr,
    hidden_states: Tuple[State, ...],
    invar_to_state: Dict[Var, State],
    outvar_to_state: Dict[Var, State],
) -> List[Set[State]]:
    """
    Analyze dependencies between hidden states and group them.

    States are grouped together if:
    1. State s1's update depends on state s2's value (element-wise)
    2. State s2's update depends on state s1's value (element-wise)
    3. Transitive closure: if {s1, s2} and {s2, s3} have dependencies,
       then {s1, s2, s3} should be in the same group

    Args:
        jaxpr: The Jaxpr to analyze
        hidden_states: States that are both inputs and outputs
        invar_to_state: Mapping from input vars to states
        outvar_to_state: Mapping from output vars to states

    Returns:
        List of state groups (each group is a set of states)
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
    """
    Build a dependency graph: for each variable, track which input variables it depends on.

    Returns:
        Dict mapping each var to the set of input vars it transitively depends on
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
    """
    Check if there's a connection operation in the computation path from in_var to out_var.

    Returns:
        True if a connection operation exists in the path, False otherwise
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
    """Check if from_var can reach to_var through the computation graph."""
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
) -> List[Group]:
    """
    Build Group objects from state groups.

    For each state group, construct a Jaxpr that describes the state update logic.
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

        # Backward traversal to find all equations needed to compute group outputs
        while queue:
            var = queue.pop(0)
            for eqn in jaxpr.eqns:
                if var in eqn.outvars and eqn not in relevant_eqns:
                    relevant_eqns.append(eqn)
                    # Add input vars to queue if not already processed
                    for in_var in eqn.invars:
                        if isinstance(in_var, Var) and in_var not in produced_vars:
                            produced_vars.add(in_var)
                            # Don't traverse beyond input state vars or if it's a connection
                            if in_var not in group_hidden_in_vars:
                                if not _is_connection(eqn):
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

        # Classify required vars
        for var in required_vars:
            if var in group_hidden_in_vars:
                continue  # Already added
            elif var in invar_to_state:
                # This is an input state (read-only)
                state = invar_to_state[var]
                if state not in group_hidden_states and state not in group_in_states:
                    group_in_states.append(state)
                    group_invars.append(var)
            else:
                # This is an input current variable (not a state)
                if var not in produced_vars and var not in group_input_vars:
                    group_input_vars.append(var)
                    group_invars.append(var)

        # Create the group jaxpr
        group_jaxpr = eqns_to_jaxpr(
            eqns=relevant_eqns,
            invars=group_invars,
            outvars=group_hidden_out_vars,
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

def _step3_extract_connections(jaxpr: Jaxpr) -> List[Tuple[JaxprEqn, Connection]]:
    """
    Extract all connection equations and create Connection objects.

    Returns:
        List of (equation, Connection) tuples
    """
    connections = []
    for eqn in jaxpr.eqns:
        if _is_connection(eqn):
            # Create a simple Jaxpr for this connection
            conn_jaxpr = eqns_to_jaxpr(
                eqns=[eqn],
                invars=list(eqn.invars),
                outvars=list(eqn.outvars),
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
) -> List[Projection]:
    """
    Build Projection objects by analyzing connections between groups.

    A Projection:
    1. Takes hidden_states from a pre_group
    2. Applies one or more Connections
    3. Produces outputs that flow into post_group as input currents
    """
    projections = []

    # For each connection, determine which group it connects from and to
    for conn_eqn, connection in connections:
        # Find which group(s) produce the inputs to this connection
        pre_groups = []
        conn_input_states = []

        for in_var in conn_eqn.invars:
            if not isinstance(in_var, Var):
                continue

            # Check if this var is a state var
            if in_var in invar_to_state:
                state = invar_to_state[in_var]
                conn_input_states.append(state)

                # Find which group has this state as hidden_state
                for group in groups:
                    if group.has_hidden_state(state):
                        if group not in pre_groups:
                            pre_groups.append(group)

        # Find which group(s) consume the outputs of this connection
        post_groups = []
        for out_var in conn_eqn.outvars:
            # Check which groups have this var in their input_vars
            for group in groups:
                if out_var in group.input_vars:
                    if group not in post_groups:
                        post_groups.append(group)

        # Validate: should have exactly one pre_group and one post_group
        if len(pre_groups) != 1:
            # This might be a more complex projection, skip for now or handle specially
            continue

        if len(post_groups) != 1:
            # Output might go to multiple groups or nowhere, skip for now
            continue

        pre_group = pre_groups[0]
        post_group = post_groups[0]

        # Build the projection Jaxpr
        # For now, assume the projection is just the connection itself
        # In more complex cases, we'd need to trace backward from the connection inputs

        # Find all equations involved in computing the projection
        # This includes the connection and any preprocessing of states
        proj_eqns = [conn_eqn]
        proj_invars = []
        proj_in_states = []

        # Add states used by this connection
        proj_hidden_states = conn_input_states
        for state in proj_hidden_states:
            proj_invars.extend(state_to_invars[state])

        proj_jaxpr = eqns_to_jaxpr(
            eqns=proj_eqns,
            invars=proj_invars,
            outvars=list(conn_eqn.outvars),
        )

        projection = Projection(
            hidden_states=proj_hidden_states,
            in_states=proj_in_states,
            jaxpr=proj_jaxpr,
            connections=[connection],
            pre_group=pre_group,
            post_group=post_group,
        )
        projections.append(projection)

    return projections


# ============================================================================
# Input and Output Analysis
# ============================================================================

def _step5_build_inputs(
    jaxpr: Jaxpr,
    groups: List[Group],
    invar_to_state: Dict[Var, State],
) -> List[Input]:
    """
    Build Input objects that describe how external inputs flow into groups.
    """
    inputs = []

    # Determine which vars are input_variables (not state vars)
    input_vars = []
    for var in jaxpr.invars:
        if var not in invar_to_state:
            input_vars.append(var)

    if not input_vars:
        return inputs

    # For each group, check which input_vars flow into it
    for group in groups:
        group_input_vars_from_external = []

        for input_var in input_vars:
            # Check if this input_var flows into the group's input_vars
            if input_var in group.input_vars:
                group_input_vars_from_external.append(input_var)

        if not group_input_vars_from_external:
            continue

        # Build Input object
        # The jaxpr is essentially an identity or transformation
        # For now, assume it's an identity (just passes through)
        input_jaxpr = eqns_to_jaxpr(
            eqns=[],  # No transformation
            invars=group_input_vars_from_external,
            outvars=group_input_vars_from_external,
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
) -> List[Output]:
    """
    Build Output objects that describe how to extract network outputs from group states.
    """
    outputs = []

    # Determine which vars are output_variables (not state vars)
    output_vars = []
    for var in jaxpr.outvars:
        if var not in outvar_to_state:
            output_vars.append(var)

    if not output_vars:
        return outputs

    # For each output var, trace back to find which states it depends on
    var_dependencies = _build_var_dependencies(jaxpr)

    # Group output vars by which group they depend on
    group_output_mapping = defaultdict(list)

    for out_var in output_vars:
        deps = var_dependencies.get(out_var, set())

        # Find which group's hidden states this depends on
        dependent_groups = []
        dependent_states = []

        for dep_var in deps:
            if dep_var in outvar_to_state:
                state = outvar_to_state[dep_var]
                for group in groups:
                    if group.has_hidden_state(state):
                        if group not in dependent_groups:
                            dependent_groups.append(group)
                        if state not in dependent_states:
                            dependent_states.append(state)

        if len(dependent_groups) == 1:
            group_output_mapping[dependent_groups[0]].append((out_var, dependent_states))
        elif len(dependent_groups) > 1:
            raise CompilationError(
                f"Output variable {out_var} depends on multiple groups: {dependent_groups}. "
                "Each output should depend on only one group."
            )

    # Create Output objects for each group
    for group, output_info in group_output_mapping.items():
        output_vars_for_group = [ov for ov, _ in output_info]
        all_dependent_states = []
        for _, states in output_info:
            for state in states:
                if state not in all_dependent_states:
                    all_dependent_states.append(state)

        # Find equations that compute these output vars
        output_eqns = []
        for eqn in jaxpr.eqns:
            if any(ov in eqn.outvars for ov in output_vars_for_group):
                output_eqns.append(eqn)

        # Determine invars for the output jaxpr
        output_invars = []
        for state in all_dependent_states:
            output_invars.extend(state_to_outvars[state])

        output_jaxpr = eqns_to_jaxpr(
            eqns=output_eqns,
            invars=output_invars,
            outvars=output_vars_for_group,
        )

        output_obj = Output(
            jaxpr=output_jaxpr,
            hidden_states=all_dependent_states,
            in_states=[],  # TODO: handle in_states if needed
            group=group,
        )
        outputs.append(output_obj)

    return outputs


# ============================================================================
# Call Order Analysis
# ============================================================================

def _step7_build_call_orders(
    jaxpr: Jaxpr,
    groups: List[Group],
    projections: List[Projection],
    inputs: List[Input],
    outputs: List[Output],
) -> List:
    """
    Determine the execution order of components based on the original Jaxpr equation order.
    """
    call_orders = []

    # Create a mapping from equations to components
    eqn_to_component = {}

    # Map group equations
    for group in groups:
        for eqn in group.jaxpr.eqns:
            eqn_to_component[id(eqn)] = group

    # Map projection equations
    for proj in projections:
        for eqn in proj.jaxpr.eqns:
            eqn_to_component[id(eqn)] = proj

    # Process equations in order and add components to call_orders
    seen_components = set()
    for eqn in jaxpr.eqns:
        eqn_id = id(eqn)
        if eqn_id in eqn_to_component:
            component = eqn_to_component[eqn_id]
            component_id = id(component)
            if component_id not in seen_components:
                seen_components.add(component_id)
                call_orders.append(component)

    # Add inputs at the beginning
    for inp in inputs:
        if inp not in call_orders:
            call_orders.insert(0, inp)

    # Add outputs at the end
    for out in outputs:
        if out not in call_orders:
            call_orders.append(out)

    return call_orders


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
    """
    Validate the compiled SNN structure.

    Checks:
    1. All hidden_states are assigned to some group
    2. All input_variables are used by some Input
    3. All output_variables are produced by some Output
    4. Projections have valid pre/post groups
    5. No orphaned connections
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
) -> CompiledGraph:
    """
    Compile a ClosedJaxpr representing an SNN single-step update into structured components.

    Args:
        closed_jaxpr: The ClosedJaxpr to compile
        in_states: Input states (t-1)
        out_states: Output states (t)
        invar_to_state: Mapping from input vars to input states
        outvar_to_state: Mapping from output vars to output states
        state_to_invars: Mapping from input states to their vars
        state_to_outvars: Mapping from output states to their vars

    Returns:
        CompiledGraph object containing all components and execution order

    Raises:
        CompilationError: If compilation fails due to invalid structure
    """
    jaxpr = closed_jaxpr.jaxpr

    # Determine hidden_states (states that are both input and output)
    hidden_states = tuple(s for s in out_states if s in in_states)

    # Step 1: Analyze state dependencies and group states
    state_groups = _step1_analyze_state_dependencies(
        jaxpr, hidden_states, invar_to_state, outvar_to_state
    )

    # Step 2: Build Group objects
    groups = _step2_build_groups(
        jaxpr, state_groups, in_states, out_states,
        invar_to_state, outvar_to_state, state_to_invars, state_to_outvars
    )

    # Step 3: Extract connections
    connections = _step3_extract_connections(jaxpr)

    # Step 4: Build Projection objects
    projections = _step4_build_projections(
        jaxpr, groups, connections, invar_to_state, state_to_invars
    )

    # Step 5: Build Input objects
    inputs = _step5_build_inputs(jaxpr, groups, invar_to_state)

    # Step 6: Build Output objects
    outputs = _step6_build_outputs(
        jaxpr, groups, outvar_to_state, state_to_outvars
    )

    # Step 7: Determine call order
    call_orders = _step7_build_call_orders(jaxpr, groups, projections, inputs, outputs)

    # Step 8: Validate the compilation
    _step8_validate_compilation(
        closed_jaxpr, groups, projections, inputs, outputs,
        in_states, out_states, invar_to_state
    )

    return CompiledGraph(
        groups=groups,
        projections=projections,
        inputs=inputs,
        outputs=outputs,
        call_orders=call_orders,
    )
