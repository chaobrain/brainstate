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
from typing import Set, Tuple, Dict, Callable, List

import jax

from brainstate._compatible_import import Jaxpr, Var, JaxprEqn, ClosedJaxpr
from brainstate._state import State
from brainstate.transform._ir_inline import inline_jit
from brainstate.transform._ir_processing import eqns_to_closed_jaxpr
from brainstate.transform._make_jaxpr import StatefulFunction, get_arg_cache_key
from ._data import Graph, GraphIRElem, GroupIR, ConnectionIR, ProjectionIR, InputIR, OutputIR, CompiledGraphIR
from ._utils import _is_connection, UnionFind

__all__ = [
    'compile_jaxpr',
    'compile_fn',
    'CompilationError',
    'Compiler',
]


class CompilationError(Exception):
    """Raised when the graph IR compiler cannot reconstruct a valid program."""
    pass


# ============================================================================
# Helper Functions
# ============================================================================

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

    # InputIR vars depend only on themselves
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
        Pre-built adjacency list that maps a variable to equations that consume it.

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
    # InputIR variables: the last len(state_avals) invars correspond to states
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
# Compiler Class
# ============================================================================

class Compiler:
    """Compiler for transforming ClosedJaxpr into structured Graph IR.

    This class encapsulates the entire compilation process, tracking equation
    usage and managing the transformation from a flat Jaxpr representation to
    a structured graph of Groups, Projections, Inputs, and Outputs.

    Parameters
    ----------
    closed_jaxpr : ClosedJaxpr
        The JAX program to compile.
    in_states : tuple[State, ...]
        InputIR states for the program.
    out_states : tuple[State, ...]
        Output states produced by the program.
    invar_to_state : dict[Var, State]
        Mapping from input variables to their states.
    outvar_to_state : dict[Var, State]
        Mapping from output variables to their states.
    state_to_invars : dict[State, tuple[Var, ...]]
        Mapping from states to their input variables.
    state_to_outvars : dict[State, tuple[Var, ...]]
        Mapping from states to their output variables.

    Attributes
    ----------
    eqn_to_id : dict[int, int]
        Maps equation object id to its sequential index.
    used_eqn_ids : set[int]
        Tracks which equations have been assigned to components.
    """

    def __init__(
        self,
        closed_jaxpr: ClosedJaxpr,
        in_states: Tuple[State, ...],
        out_states: Tuple[State, ...],
        invar_to_state: Dict[Var, State],
        outvar_to_state: Dict[Var, State],
        state_to_invars: Dict[State, Tuple[Var, ...]],
        state_to_outvars: Dict[State, Tuple[Var, ...]],
    ):
        # Store the original program
        self.closed_jaxpr = closed_jaxpr
        self.jaxpr = closed_jaxpr.jaxpr
        self.consts = closed_jaxpr.consts

        # Store state information
        self.in_states = in_states
        self.out_states = out_states
        self.invar_to_state = invar_to_state
        self.outvar_to_state = outvar_to_state
        self.state_to_invars = state_to_invars
        self.state_to_outvars = state_to_outvars

        # Compute hidden states
        self.hidden_states = tuple(s for s in out_states if s in in_states)

        # Track equation usage: map equation object id to its index
        self.eqn_to_id = {id(eqn): idx for idx, eqn in enumerate(self.jaxpr.eqns)}
        self.used_eqn_ids = set()  # Set of equation object ids that have been used

    def _mark_eqns_as_used(self, eqns: List[JaxprEqn]) -> None:
        """Mark a list of equations as used.

        Parameters
        ----------
        eqns : list[JaxprEqn]
            Equations to mark as used.
        """
        for eqn in eqns:
            eqn_id = id(eqn)
            self.used_eqn_ids.add(eqn_id)

    def _make_closed_jaxpr(
        self,
        eqns: List[JaxprEqn],
        invars: List[Var],
        outvars: List[Var],
    ) -> ClosedJaxpr:
        """Create a ClosedJaxpr and mark equations as used.

        Parameters
        ----------
        eqns : list[JaxprEqn]
            Equations for the sub-program.
        invars : list[Var]
            Input variables.
        outvars : list[Var]
            Output variables.

        Returns
        -------
        ClosedJaxpr
            The constructed sub-program with appropriate constants.
        """
        # Mark these equations as used
        self._mark_eqns_as_used(eqns)

        # Create the closed jaxpr
        closed_jaxpr = eqns_to_closed_jaxpr(eqns=eqns, invars=invars, outvars=outvars)

        # Extract corresponding consts from the original jaxpr
        if closed_jaxpr.jaxpr.constvars:
            consts = _extract_consts_for_vars(
                closed_jaxpr.jaxpr.constvars,
                self.jaxpr,
                self.consts
            )
            return ClosedJaxpr(closed_jaxpr.jaxpr, consts)
        else:
            return closed_jaxpr

    def step1_analyze_state_dependencies(self) -> List[Set[State]]:
        """Group hidden states that are mutually dependent via non-connection ops.

        Returns
        -------
        list[set[State]]
            Sets of states that must be compiled into the same :class:`GroupIR`.
        """
        # Create a mapping from state to its ID for efficient comparison
        state_to_id = {id(state): state for state in self.hidden_states}
        hidden_state_ids = set(state_to_id.keys())

        # Union-Find structure to track state grouping
        uf = UnionFind()
        for state in self.hidden_states:
            uf.make_set(id(state))

        # Build a mapping: output_var -> set of input_vars it depends on
        var_dependencies = _build_var_dependencies(self.jaxpr)

        # For each hidden state's output var, check which hidden state input vars it depends on
        for out_var in self.jaxpr.outvars:
            if out_var not in self.outvar_to_state:
                continue

            out_state = self.outvar_to_state[out_var]
            out_state_id = id(out_state)

            if out_state_id not in hidden_state_ids:
                continue

            # Get all input vars this output depends on
            dependent_vars = var_dependencies.get(out_var, set())

            for in_var in dependent_vars:
                if in_var not in self.invar_to_state:
                    continue

                in_state = self.invar_to_state[in_var]
                in_state_id = id(in_state)

                if in_state_id not in hidden_state_ids:
                    continue

                # If output state depends on input state, they should be in the same group
                # (assuming element-wise operations, which we verify by checking non-connection ops)
                if not _has_connection_between(self.jaxpr, in_var, out_var):
                    uf.union(out_state_id, in_state_id)

        # Get the grouped states
        state_id_groups = uf.get_groups()
        state_groups = []
        for id_group in state_id_groups:
            state_group = {state_to_id[sid] for sid in id_group}
            state_groups.append(state_group)

        return state_groups

    def step2_build_groups(self, state_groups: List[Set[State]]) -> List[GroupIR]:
        """Materialize :class:`Group` objects for each mutually dependent state set.

        Parameters
        ----------
        state_groups : list[set[State]]
            Output of :meth:`step1_analyze_state_dependencies`.

        Returns
        -------
        list[GroupIR]
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
                group_hidden_in_vars.extend(self.state_to_invars.get(state))

            # Collect all output vars for these hidden states
            group_hidden_out_vars = []
            for state in group_hidden_states:
                group_hidden_out_vars.extend(self.state_to_outvars.get(state))

            # Find equations that produce these output vars
            relevant_eqns = []
            produced_vars = set(group_hidden_out_vars)
            queue = list(group_hidden_out_vars)
            # Track variables that come from connections (should be input_vars)
            connection_output_vars = set()

            # Backward traversal to find all equations needed to compute group outputs
            while queue:
                var = queue.pop(0)
                for eqn in self.jaxpr.eqns:
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
            eqn_order = {id(eqn): i for i, eqn in enumerate(self.jaxpr.eqns)}
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
                elif var in self.invar_to_state:
                    # This is an input state (read-only)
                    state = self.invar_to_state[var]
                    if state not in group_hidden_states and state not in group_in_states:
                        group_in_states.append(state)
                        group_invars.append(var)
                else:
                    # This is an external input variable (not a state, not a connection)
                    if var not in group_input_vars:
                        group_input_vars.append(var)
                        group_invars.append(var)

            # Create the group ClosedJaxpr
            group_jaxpr = self._make_closed_jaxpr(
                eqns=relevant_eqns,
                invars=group_invars,
                outvars=group_hidden_out_vars,
            )

            # Determine out_states (states produced but not consumed)
            group_out_states = []
            group_hidden_state_ids = {id(s) for s in group_hidden_states}
            for state in self.out_states:
                if id(state) not in group_hidden_state_ids:
                    # Check if this group produces this state
                    state_out_vars = self.state_to_outvars.get(state)
                    if any(v in group_hidden_out_vars for v in state_out_vars):
                        group_out_states.append(state)
            del group_hidden_state_ids

            # Generate a name for this group based on its hidden states
            group_name = f"Group_{len(groups)}"

            group = GroupIR(
                jaxpr=group_jaxpr,
                hidden_states=group_hidden_states,
                in_states=group_in_states,
                out_states=group_out_states,
                input_vars=group_input_vars,
                name=group_name,
            )
            groups.append(group)

        return groups

    def step3_extract_connections(self) -> List[Tuple[JaxprEqn, ConnectionIR]]:
        """Identify connection equations and wrap them as :class:`Connection` objects.

        Returns
        -------
        list[tuple[JaxprEqn, ConnectionIR]]
            Pairs of the original equation and a :class:`Connection` wrapper that
            holds its ClosedJaxpr slice.
        """
        connections = []
        for eqn in self.jaxpr.eqns:
            if _is_connection(eqn):
                # Create a ClosedJaxpr for this connection
                conn_jaxpr = self._make_closed_jaxpr(
                    eqns=[eqn],
                    invars=list(eqn.invars),
                    outvars=list(eqn.outvars),
                )
                connection = ConnectionIR(jaxpr=conn_jaxpr)
                connections.append((eqn, connection))
        return connections

    def step4_build_projections(
        self,
        groups: List[GroupIR],
        connections: List[Tuple[JaxprEqn, ConnectionIR]],
    ) -> List[ProjectionIR]:
        """Create :class:`Projection` objects that ferry spikes between groups.

        Parameters
        ----------
        groups : list[GroupIR]
            Groups created in :meth:`step2_build_groups`.
        connections : list[tuple[JaxprEqn, ConnectionIR]]
            Connection equations identified by :meth:`step3_extract_connections`.

        Returns
        -------
        list[ProjectionIR]
            Projection descriptors that own the equations/connection metadata for
            a pre→post group path.
        """
        projections = []

        # Build a mapping: var -> equation that produces it
        var_to_producer_eqn = {}
        for eqn in self.jaxpr.eqns:
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

                proj_info = self._trace_projection_path(
                    input_var=input_var,
                    post_group=post_group,
                    groups=groups,
                    var_to_producer_eqn=var_to_producer_eqn,
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
                    merged_eqns, merged_hidden_states, merged_in_states, merged_connections = \
                        self._merge_projection_data(
                            existing_proj, proj_eqns, proj_hidden_states, proj_in_states, proj_connections
                        )

                    # Build new projection jaxpr
                    proj_jaxpr, proj_outvars = self._build_projection_jaxpr(
                        merged_eqns, merged_hidden_states, merged_in_states, post_group, group_to_input_vars
                    )

                    # Replace existing projection
                    new_proj = ProjectionIR(
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
                    proj_jaxpr, proj_outvars = self._build_projection_jaxpr(
                        proj_eqns, proj_hidden_states, proj_in_states, post_group, group_to_input_vars
                    )

                    projection = ProjectionIR(
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
        self,
        input_var: Var,
        post_group: GroupIR,
        groups: List[GroupIR],
        var_to_producer_eqn: Dict[Var, JaxprEqn],
        connections: List[Tuple[JaxprEqn, ConnectionIR]],
    ):
        """Trace the computation that produces ``input_var`` for a group."""
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
            if var in self.jaxpr.invars:
                # Check if it's a state var
                if var in self.invar_to_state:
                    state = self.invar_to_state[var]
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
        eqn_order = {id(eqn): i for i, eqn in enumerate(self.jaxpr.eqns)}
        proj_eqns.sort(key=lambda e: eqn_order[id(e)])

        return pre_group, proj_eqns, proj_hidden_states, proj_in_states, proj_connections

    def _merge_projection_data(
        self,
        existing_proj: ProjectionIR,
        proj_eqns: List[JaxprEqn],
        proj_hidden_states: List[State],
        proj_in_states: List[State],
        proj_connections: List[ConnectionIR],
    ):
        """Merge new projection data with existing projection."""
        merged_eqns = list(existing_proj.jaxpr.jaxpr.eqns)
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
        eqn_order = {id(eqn): i for i, eqn in enumerate(self.jaxpr.eqns)}
        merged_eqns.sort(key=lambda e: eqn_order[id(e)])

        return merged_eqns, merged_hidden_states, merged_in_states, merged_connections

    def _build_projection_jaxpr(
        self,
        proj_eqns: List[JaxprEqn],
        proj_hidden_states: List[State],
        proj_in_states: List[State],
        post_group: GroupIR,
        group_to_input_vars: Dict[int, Set[Var]],
    ):
        """Build a ClosedJaxpr for a projection."""
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
            for var in self.state_to_invars[state]:
                if var in proj_invars_needed:
                    proj_invars.append(var)
                    proj_invars_needed.remove(var)
        # Then add in_states
        for state in proj_in_states:
            for var in self.state_to_invars[state]:
                if var in proj_invars_needed:
                    proj_invars.append(var)
                    proj_invars_needed.remove(var)

        # Add any remaining needed vars that aren't from states
        proj_invars.extend(sorted(proj_invars_needed, key=lambda v: str(v)))

        # Build outvars from equations
        proj_outvars = []
        for eqn in proj_eqns:
            for out_var in eqn.outvars:
                if out_var in group_to_input_vars[id(post_group)]:
                    if out_var not in proj_outvars:
                        proj_outvars.append(out_var)

        # Create ClosedJaxpr
        proj_jaxpr = self._make_closed_jaxpr(
            eqns=proj_eqns,
            invars=proj_invars,
            outvars=proj_outvars,
        )

        return proj_jaxpr, proj_outvars

    def step5_build_inputs(self, groups: List[GroupIR]) -> List[InputIR]:
        """Create :class:`Input` descriptors for external variables.

        Parameters
        ----------
        groups : list[GroupIR]
            Group descriptors receiving the inputs.

        Returns
        -------
        list[InputIR]
            Input descriptors grouped by their destination group.
        """
        # Determine which vars are input_variables (not state vars)
        input_vars = []
        for var in self.jaxpr.invars:
            if var not in self.invar_to_state:
                input_vars.append(var)

        if not input_vars:
            return []

        # Build a mapping: var -> equations that consume it
        var_to_consumer_eqns = defaultdict(list)
        for eqn in self.jaxpr.eqns:
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
            trace_result = self._trace_input_forward(
                input_var=input_var,
                groups=groups,
                var_to_consumer_eqns=var_to_consumer_eqns,
                group_input_vars_sets=group_input_vars_sets,
            )

            if trace_result is not None:
                input_traces.append(trace_result)

        # GroupIR traces by target group (use group id as key)
        group_id_to_traces = defaultdict(list)
        id_to_group = {}

        for input_var, target_group, equations, outvars in input_traces:
            group_id = id(target_group)
            id_to_group[group_id] = target_group
            group_id_to_traces[group_id].append((input_var, equations, outvars))

        # Create InputIR objects for each group
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
            eqn_order = {id(eqn): i for i, eqn in enumerate(self.jaxpr.eqns)}
            all_equations.sort(key=lambda e: eqn_order[id(e)])

            # Create the input ClosedJaxpr
            input_jaxpr = self._make_closed_jaxpr(
                eqns=all_equations,
                invars=all_input_vars,
                outvars=all_output_vars,
            )

            input_obj = InputIR(
                jaxpr=input_jaxpr,
                group=group,
            )
            inputs.append(input_obj)

        return inputs

    def _trace_input_forward(
        self,
        input_var: Var,
        groups: List[GroupIR],
        var_to_consumer_eqns: Dict[Var, List[JaxprEqn]],
        group_input_vars_sets: Dict[int, Set[Var]],
    ):
        """Forward-trace ``input_var`` until its values flow into a group boundary."""
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

    def step6_build_outputs(self, groups: List[GroupIR]) -> List[OutputIR]:
        """Describe how model outputs are assembled from group state variables.

        Parameters
        ----------
        groups : list[GroupIR]
            Groups that may contribute to outputs.

        Returns
        -------
        list[OutputIR]
            Output descriptors paired with the responsible group.

        Raises
        ------
        CompilationError
            If an output depends on unsupported intermediates or multiple groups.
        """
        # Identify state outvars (variables that correspond to state outputs)
        state_outvars_set = set([v for outvars in self.state_to_outvars.values() for v in outvars])

        # Identify state invars (variables that correspond to state inputs)
        state_invars_set = set(self.invar_to_state.keys())

        # Get output_vars (jaxpr outvars that are not state outvars)
        output_vars = [v for v in self.jaxpr.outvars if v not in state_outvars_set]

        if not output_vars:
            return []

        # Build a mapping: var -> equation that produces it
        var_to_producer_eqn = {}
        for eqn in self.jaxpr.eqns:
            for out_var in eqn.outvars:
                var_to_producer_eqn[out_var] = eqn

        # GroupIR output vars by which group they depend on (use group id as key)
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
                    if var in self.jaxpr.invars:
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

            # Verify that all inputs to equations_needed are valid
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
                if var in self.outvar_to_state:
                    state = self.outvar_to_state[var]
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
                if var in self.invar_to_state:
                    state = self.invar_to_state[var]
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
            eqn_order = {id(eqn): i for i, eqn in enumerate(self.jaxpr.eqns)}
            all_equations.sort(key=lambda e: eqn_order[id(e)])

            # Determine invars for the output jaxpr
            # We need to use state outvars for hidden states and state invars for in states
            output_invars = []

            # Add hidden state outvars
            for state in all_dependent_hidden_states:
                output_invars.extend(self.state_to_outvars[state])

            # Add in state invars
            for state in all_dependent_in_states:
                output_invars.extend(self.state_to_invars[state])

            # Create the output ClosedJaxpr
            output_jaxpr = self._make_closed_jaxpr(
                eqns=all_equations,
                invars=output_invars,
                outvars=output_vars_for_group,
            )

            output_obj = OutputIR(
                jaxpr=output_jaxpr,
                hidden_states=all_dependent_hidden_states,
                in_states=all_dependent_in_states,
                group=group,
            )
            outputs.append(output_obj)

        return outputs

    def step7_build_graph(
        self,
        groups: List[GroupIR],
        projections: List[ProjectionIR],
        inputs: List[InputIR],
        outputs: List[OutputIR],
    ) -> Graph:
        """Derive an execution graph that preserves the original equation order.

        Parameters
        ----------
        groups : list[GroupIR]
            Computation blocks that produce state updates.
        projections : list[ProjectionIR]
            ConnectionIR pipelines between groups.
        inputs : list[InputIR]
            External inputs to the network.
        outputs : list[OutputIR]
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
        eqn_to_component: Dict[int, GraphIRElem] = {}
        var_to_component: Dict[Var, GraphIRElem] = {}

        for group in groups:
            for eqn in group.jaxpr.jaxpr.eqns:
                eqn_to_component[id(eqn)] = group
        for proj in projections:
            for eqn in proj.jaxpr.jaxpr.eqns:
                eqn_to_component[id(eqn)] = proj

        # Process equations in order and add components + edges
        seen_components = set()
        for eqn in self.jaxpr.eqns:
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

    def step8_validate_compilation(
        self,
        groups: List[GroupIR],
        projections: List[ProjectionIR],
        inputs: List[InputIR],
        outputs: List[OutputIR],
    ) -> None:
        """Run structural checks on the assembled compilation result.

        Parameters
        ----------
        groups, projections, inputs, outputs : list
            Components produced by previous compilation phases.

        Raises
        ------
        CompilationError
            If invariants such as "each hidden state belongs to a group" or
            "all equations are used" are violated.
        """
        # Check 1: All hidden states should be in some group
        all_group_hidden_states = set()
        for group in groups:
            for state in group.hidden_states:
                all_group_hidden_states.add(id(state))

        hidden_states = set(s for s in self.out_states if s in self.in_states)
        for state in hidden_states:
            if id(state) not in all_group_hidden_states:
                raise CompilationError(
                    f"Hidden state {state} is not assigned to any group"
                )

        # Check 2: Projections should have non-empty connections
        for proj in projections:
            if not proj.connections:
                raise CompilationError(
                    f"ProjectionIR from {proj.pre_group} to {proj.post_group} has no connections"
                )

        # Check 3: ProjectionIR hidden_states should belong to exactly one group
        for proj in projections:
            for state in proj.hidden_states:
                count = sum(1 for g in groups if g.has_hidden_state(state))
                if count == 0:
                    raise CompilationError(
                        f"ProjectionIR depends on state {state} which is not in any group"
                    )
                elif count > 1:
                    raise CompilationError(
                        f"ProjectionIR depends on state {state} which belongs to multiple groups"
                    )

        # Check 4: All equations should be used
        unused_eqn_ids = set(self.eqn_to_id.keys()) - self.used_eqn_ids
        if unused_eqn_ids:
            unused_indices = sorted([self.eqn_to_id[eqn_id] for eqn_id in unused_eqn_ids])
            raise CompilationError(
                f"Not all equations were used in compilation. "
                f"Unused equation indices: {unused_indices}. "
                f"This may indicate that some computations are not part of any GroupIR, "
                f"ProjectionIR, InputIR, or Output."
            )

    def compile(self) -> Tuple[List[GroupIR], List[ProjectionIR], List[InputIR], List[OutputIR], Graph]:
        """Execute the complete compilation pipeline.

        Returns
        -------
        tuple
            A 5-tuple containing:
            - groups : list[GroupIR]
            - projections : list[ProjectionIR]
            - inputs : list[InputIR]
            - outputs : list[Output]
            - graph : Graph

        Raises
        ------
        CompilationError
            If any validation step fails.
        """
        # Step 1: Analyze state dependencies and group states
        state_groups = self.step1_analyze_state_dependencies()

        # Step 2: Build GroupIR objects
        groups = self.step2_build_groups(state_groups)

        # Step 3: Extract connections
        connections = self.step3_extract_connections()

        # Step 4: Build ProjectionIR objects
        projections = self.step4_build_projections(groups, connections)

        # Step 5: Build InputIR objects
        inputs = self.step5_build_inputs(groups)

        # Step 6: Build Output objects
        outputs = self.step6_build_outputs(groups)

        # Step 7: Determine call order
        graph = self.step7_build_graph(groups, projections, inputs, outputs)

        # Step 8: Validate the compilation
        self.step8_validate_compilation(groups, projections, inputs, outputs)

        return groups, projections, inputs, outputs, graph


# ============================================================================
# Main Compilation Functions
# ============================================================================

def compile_jaxpr(
    closed_jaxpr: ClosedJaxpr,
    in_states: Tuple[State, ...],
    out_states: Tuple[State, ...],
    invar_to_state: Dict[Var, State],
    outvar_to_state: Dict[Var, State],
    state_to_invars: Dict[State, Tuple[Var, ...]],
    state_to_outvars: Dict[State, Tuple[Var, ...]],
) -> Tuple:
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

    Returns
    -------
    tuple
        A 5-tuple containing (groups, projections, inputs, outputs, graph).

    Raises
    ------
    CompilationError
        If the ClosedJaxpr violates the IR assumptions (e.g. outputs depend on
        multiple groups, or not all equations are used).
    """
    compiler = Compiler(
        closed_jaxpr=closed_jaxpr,
        in_states=in_states,
        out_states=out_states,
        invar_to_state=invar_to_state,
        outvar_to_state=outvar_to_state,
        state_to_invars=state_to_invars,
        state_to_outvars=state_to_outvars,
    )
    return compiler.compile()


def compile_fn(
    target: StatefulFunction | Callable,
    jit_inline: bool = True,
) -> Callable[..., CompiledGraphIR]:
    """Create a compiler that compiles ``stateful_fn`` into graph IR.

    Parameters
    ----------
    target : StatefulFunction, Callable
        Stateful function or callable target to compile.
    jit_inline : bool, optional
        When ``True`` the compiler inlines JIT-wrapped connection primitives
        before compilation.

    Returns
    -------
    Callable[..., CompiledGraphIR]
        Function that, when invoked with runtime arguments, returns
        :class:`CompiledGraphIR`.
    """
    if isinstance(target, StatefulFunction):
        stateful_fn = target
    elif callable(target):
        stateful_fn = StatefulFunction(target, return_only_write=True, ir_optimizations='dce')
    else:
        raise TypeError(
            "Target must be either a StatefulFunction or a callable object."
        )
    assert stateful_fn.return_only_write, (
        "Compiler currently only supports stateful functions that return only write states. "
    )

    def call(*args, **kwargs):
        """Run the compiler for the provided arguments."""
        # Get jaxpr
        jaxpr = stateful_fn.get_jaxpr(*args, **kwargs)
        if jit_inline:
            jaxpr = inline_jit(jaxpr, _is_connection)

        # Build state mappings
        in_states = stateful_fn.get_states(*args, **kwargs)
        out_states = stateful_fn.get_write_states(*args, **kwargs)
        state_mapping = _build_state_mapping(jaxpr, in_states, out_states)

        # Compile the SNN
        groups, projections, inputs, outputs, graph = compile_jaxpr(
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

        return CompiledGraphIR(
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
