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

import warnings
from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from typing import Set, Tuple, Dict, Callable, List, Optional, Union, Sequence

import jax

from brainstate._compatible_import import Jaxpr, Var, JaxprEqn, ClosedJaxpr
from brainstate._state import State
from brainstate.transform._ir_inline import inline_jit
from brainstate.transform._ir_processing import eqns_to_jaxpr
from brainstate.transform._make_jaxpr import StatefulFunction
from ._data import (
    NeuroGraph,
    GraphElem,
    ConnectionPrim,
    GroupPrim, ProjectionPrim, InputPrim, OutputPrim, UnknownPrim,
    CompiledGraphIR,
)
from ._utils import _is_connection, UnionFind, get_hidden_name

__all__ = [
    'compile_jaxpr',
    'compile_fn',
    'CompilationError',
    'NeuronIRCompiler',
    'CompilationContext',
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
    for idx, var in enumerate(constvars):
        if var in constvar_to_const:
            consts.append(constvar_to_const[var])
        else:
            # This constvar is not in the original jaxpr, which shouldn't happen
            available_vars = [str(v) for v in original_jaxpr.constvars[:5]]
            if len(original_jaxpr.constvars) > 5:
                available_vars.append('...')
            raise CompilationError(
                f"Constvar {var} (at index {idx}) not found in original jaxpr.\n"
                f"  Requested: {var}\n"
                f"  Available constvars ({len(original_jaxpr.constvars)}): {', '.join(available_vars)}\n"
                f"  Expected {len(constvars)} constvars, but {len(original_consts)} consts provided.\n"
                f"  This may indicate a mismatch between equation extraction and const extraction."
            )

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

    # InputPrim vars depend only on themselves
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
    in_states: Sequence[State],
    out_states: Sequence[State],
) -> Dict:
    """Map JAXPR variables to their corresponding ``State`` instances.

    Parameters
    ----------
    closed_jaxpr : ClosedJaxpr
        Program emitted by ``StatefulFunction``.
    in_states, out_states : sequence of State
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
    # InputPrim variables: the last len(state_avals) invars correspond to states
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

    # OutputPrim variables: after the main outputs, the rest correspond to state updates
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


@dataclass
class CompilationContext:
    """Container for compilation state and metadata.

    This dataclass encapsulates all the state mappings and metadata needed
    for compilation, simplifying the NeuronIRCompiler API.

    Parameters
    ----------
    closed_jaxpr : ClosedJaxpr
        The JAX program to compile.
    in_states : tuple[State, ...]
        InputPrim states for the program.
    out_states : tuple[State, ...]
        OutputPrim states produced by the program.
    invar_to_state : dict[Var, State]
        Mapping from input variables to their states.
    outvar_to_state : dict[Var, State]
        Mapping from output variables to their states.
    state_to_invars : dict[State, tuple[Var, ...]]
        Mapping from states to their input variables.
    state_to_outvars : dict[State, tuple[Var, ...]]
        Mapping from states to their output variables.
    """
    closed_jaxpr: ClosedJaxpr
    in_states: Tuple[State, ...]
    out_states: Tuple[State, ...]
    invar_to_state: Dict[Var, State]
    outvar_to_state: Dict[Var, State]
    state_to_invars: Dict[State, Tuple[Var, ...]]
    state_to_outvars: Dict[State, Tuple[Var, ...]]


def _group_consecutive_indices(indices: List[int]) -> List[List[int]]:
    """GroupPrim a list of indices into consecutive groups.

    Parameters
    ----------
    indices : list[int]
        List of indices to group (will be sorted internally).

    Returns
    -------
    list[list[int]]
        List of groups, where each group contains consecutive indices.

    Examples
    --------
    >>> _group_consecutive_indices([1, 2, 3, 7, 8, 9])
    [[1, 2, 3], [7, 8, 9]]
    >>> _group_consecutive_indices([5])
    [[5]]
    >>> _group_consecutive_indices([1, 3, 5])
    [[1], [3], [5]]
    """
    if not indices:
        return []

    sorted_indices = sorted(indices)
    groups = []
    current_group = [sorted_indices[0]]

    for i in range(1, len(sorted_indices)):
        if sorted_indices[i] == sorted_indices[i - 1] + 1:
            # Consecutive, add to current group
            current_group.append(sorted_indices[i])
        else:
            # Not consecutive, start a new group
            groups.append(current_group)
            current_group = [sorted_indices[i]]

    # Don't forget the last group
    groups.append(current_group)
    return groups


# ============================================================================
# Compiler Class
# ============================================================================

class NeuronIRCompiler:
    """Compiler for transforming ClosedJaxpr into structured Graph IR.

    This class encapsulates the entire compilation process, tracking equation
    usage and managing the transformation from a flat Jaxpr representation to
    a structured graph of Groups, Projections, Inputs, and Outputs.

    Parameters
    ----------
    closed_jaxpr : ClosedJaxpr
        The JAX program to compile.
    in_states : tuple[State, ...]
        InputPrim states for the program.
    out_states : tuple[State, ...]
        OutputPrim states produced by the program.
    invar_to_state : dict[Var, State]
        Mapping from input variables to their states.
    outvar_to_state : dict[Var, State]
        Mapping from output variables to their states.
    state_to_invars : dict[State, tuple[Var, ...]]
        Mapping from states to their input variables.
    state_to_outvars : dict[State, tuple[Var, ...]]
        Mapping from states to their output variables.
    validation: Union[str, Sequence[str]] = None
        Validation methods to apply.

        - `hidden_state_belong_to_group`: Check that each hidden state is assigned to exactly one group.
        - `projection_connections`: Check that projections connect valid groups.
        - `hidden_state_belong_to_one_group`: Check that hidden states are not shared across groups.
        - `all_equations_used`: Check that all equations are assigned to groups.


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
        validation: Union[str, Sequence[str]] = 'all',
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

        # validation
        if isinstance(validation, str):
            validation = (validation,)
        elif isinstance(validation, (list, tuple)):
            validation = tuple(validation)
        elif validation is None:
            validation = tuple()
        else:
            raise TypeError(
                f"'validation' must be None, 'all', or a sequence of strings, "
                f"but got {validation}."
            )
        self.validation = validation

    @classmethod
    def from_context(cls, context: CompilationContext) -> 'NeuronIRCompiler':
        """Create a compiler from a CompilationContext.

        This is an alternative constructor that accepts a CompilationContext
        dataclass instead of individual parameters, providing a cleaner API.

        Parameters
        ----------
        context : CompilationContext
            Container with all compilation state and metadata.

        Returns
        -------
        NeuronIRCompiler
            Compiler instance initialized with the context data.

        Examples
        --------
        >>> context = CompilationContext(
        ...     closed_jaxpr=jaxpr,
        ...     in_states=in_states,
        ...     out_states=out_states,
        ...     invar_to_state=mappings['invar_to_state'],
        ...     outvar_to_state=mappings['outvar_to_state'],
        ...     state_to_invars=mappings['state_to_invars'],
        ...     state_to_outvars=mappings['state_to_outvars'],
        ... )
        >>> compiler = NeuronIRCompiler.from_context(context)
        """
        return cls(
            closed_jaxpr=context.closed_jaxpr,
            in_states=context.in_states,
            out_states=context.out_states,
            invar_to_state=context.invar_to_state,
            outvar_to_state=context.outvar_to_state,
            state_to_invars=context.state_to_invars,
            state_to_outvars=context.state_to_outvars,
        )

    @cached_property
    def _var_to_producer_eqn(self) -> Dict[Var, JaxprEqn]:
        """Build a mapping from variable to the equation that produces it.

        This cached property avoids O(n) linear searches during compilation.

        Returns
        -------
        dict[Var, JaxprEqn]
            Mapping from each output variable to the equation that produces it.
        """
        mapping = {}
        for eqn in self.jaxpr.eqns:
            for out_var in eqn.outvars:
                mapping[out_var] = eqn
        return mapping

    @cached_property
    def _var_to_consumer_eqns(self) -> Dict[Var, List[JaxprEqn]]:
        """Build a mapping from variable to equations that consume it.

        This cached property avoids rebuilding the mapping for every input trace.

        Returns
        -------
        dict[Var, list[JaxprEqn]]
            Mapping from each variable to the list of equations that use it as input.
        """
        mapping = defaultdict(list)
        for eqn in self.jaxpr.eqns:
            for in_var in eqn.invars:
                if isinstance(in_var, Var):
                    mapping[in_var].append(eqn)
        return dict(mapping)

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

    def _sort_equations_by_order(self, eqns: List[JaxprEqn]) -> List[JaxprEqn]:
        """Sort equations by their original order in the jaxpr.

        This helper method extracts the common pattern of sorting equations
        that appears throughout the compilation process.

        Parameters
        ----------
        eqns : list[JaxprEqn]
            Equations to sort.

        Returns
        -------
        list[JaxprEqn]
            Equations sorted by their position in the original jaxpr.

        Notes
        -----
        Time complexity: O(n log n) where n is len(eqns).
        """
        eqn_order = {id(eqn): i for i, eqn in enumerate(self.jaxpr.eqns)}
        return sorted(eqns, key=lambda e: eqn_order[id(e)])

    def _make_closed_jaxpr(
        self,
        eqns: List[JaxprEqn],
        invars: List[Var],
        outvars: List[Var],
        mark_as_used: bool = True,
    ) -> ClosedJaxpr:
        """Create a ClosedJaxpr and optionally mark equations as used.

        Parameters
        ----------
        eqns : list[JaxprEqn]
            Equations for the sub-program.
        invars : list[Var]
            InputPrim variables.
        outvars : list[Var]
            OutputPrim variables.
        mark_as_used : bool, optional
            Whether to mark these equations as used. Default is True.
            Set to False when creating intermediate representations that
            will be marked as used later (e.g., Connections in step3).

        Returns
        -------
        ClosedJaxpr
            The constructed sub-program with appropriate constants.
        """
        # Mark these equations as used if requested
        if mark_as_used:
            self._mark_eqns_as_used(eqns)

        # Create the closed jaxpr
        jaxpr = eqns_to_jaxpr(eqns=eqns, invars=invars, outvars=outvars)

        # Extract corresponding consts from the original jaxpr
        if jaxpr.constvars:
            consts = _extract_consts_for_vars(
                jaxpr.constvars,
                self.jaxpr,
                self.consts
            )
        else:
            consts = []
        return ClosedJaxpr(jaxpr, consts)

    def step1_analyze_state_dependencies(self) -> List[Set[State]]:
        """GroupPrim hidden states that are mutually dependent via non-connection ops.

        Returns
        -------
        list[set[State]]
            Sets of states that must be compiled into the same :class:`GroupPrim`.
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

    def step2_build_groups(self, state_groups: List[Set[State]]) -> List[GroupPrim]:
        """Materialize :class:`Group` objects for each mutually dependent state set.

        Parameters
        ----------
        state_groups : list[set[State]]
            OutputPrim of :meth:`step1_analyze_state_dependencies`.

        Returns
        -------
        list[GroupPrim]
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
            relevant_eqns = self._sort_equations_by_order(relevant_eqns)

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

            group = GroupPrim(
                jaxpr=group_jaxpr,
                hidden_states=group_hidden_states,
                in_states=group_in_states,
                out_states=group_out_states,
                input_vars=group_input_vars,
                name=group_name,
            )
            groups.append(group)

        return groups

    def step3_extract_connections(self) -> List[Tuple[JaxprEqn, ConnectionPrim]]:
        """Identify connection equations and wrap them as :class:`Connection` objects.

        Note: Connection equations are NOT marked as used here. They will be
        marked as used when they are composed into ProjectionPrim instances in step4.

        Returns
        -------
        list[tuple[JaxprEqn, ConnectionPrim]]
            Pairs of the original equation and a :class:`Connection` wrapper that
            holds its ClosedJaxpr slice.
        """
        connections = []
        conn_idx = 0
        for eqn in self.jaxpr.eqns:
            if _is_connection(eqn):
                # Create a ClosedJaxpr for this connection WITHOUT marking as used
                conn_jaxpr = self._make_closed_jaxpr(
                    eqns=[eqn],
                    invars=list(eqn.invars),
                    outvars=list(eqn.outvars),
                    mark_as_used=False,  # Don't mark yet - will be marked in step4
                )
                connection = ConnectionPrim(jaxpr=conn_jaxpr, name=f"Connection_{conn_idx}")
                connections.append((eqn, connection))
                conn_idx += 1
        return connections

    def step4_build_projections(
        self,
        groups: List[GroupPrim],
        connections: List[Tuple[JaxprEqn, ConnectionPrim]],
    ) -> List[ProjectionPrim]:
        """Create :class:`ProjectionPrim` objects that ferry spikes between groups.

        Note: ConnectionPrim equations from step3 are marked as used HERE when they
        are composed into ProjectionPrim instances via _build_projection_jaxpr().

        Parameters
        ----------
        groups : list[GroupPrim]
            Groups created in :meth:`step2_build_groups`.
        connections : list[tuple[JaxprEqn, ConnectionIR]]
            ConnectionPrim equations identified by :meth:`step3_extract_connections`.

        Returns
        -------
        list[ProjectionIR]
            ProjectionPrim descriptors that own the equations/connection metadata for
            a pre→post group path.
        """
        projections = []
        proj_idx = 0

        # Use cached mapping: var -> equation that produces it
        var_to_producer_eqn = self._var_to_producer_eqn

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
                    out = self._merge_projection_data(
                        existing_proj, proj_eqns, proj_hidden_states, proj_in_states, proj_connections
                    )
                    merged_eqns, merged_hidden_states, merged_in_states, merged_connections = out

                    # Build new projection jaxpr
                    proj_jaxpr, proj_outvars = self._build_projection_jaxpr(
                        merged_eqns, merged_hidden_states, merged_in_states, post_group, group_to_input_vars
                    )

                    # Replace existing projection
                    new_proj = ProjectionPrim(
                        hidden_states=merged_hidden_states,
                        in_states=merged_in_states,
                        jaxpr=proj_jaxpr,
                        connections=merged_connections,
                        pre_group=pre_group,
                        post_group=post_group,
                        name=existing_proj.name,  # Keep the same name when merging
                    )
                    projections[projections.index(existing_proj)] = new_proj

                else:
                    # Create new projection
                    proj_jaxpr, proj_outvars = self._build_projection_jaxpr(
                        proj_eqns, proj_hidden_states, proj_in_states, post_group, group_to_input_vars
                    )

                    projection = ProjectionPrim(
                        hidden_states=proj_hidden_states,
                        in_states=proj_in_states,
                        jaxpr=proj_jaxpr,
                        connections=proj_connections,
                        pre_group=pre_group,
                        post_group=post_group,
                        name=f"Projection_{proj_idx}",
                    )
                    projections.append(projection)
                    proj_idx += 1

        return projections

    def _trace_projection_path(
        self,
        input_var: Var,
        groups: List[GroupPrim],
        var_to_producer_eqn: Dict[Var, JaxprEqn],
        connections: List[Tuple[JaxprEqn, ConnectionPrim]],
    ) -> Optional[Tuple[GroupPrim, List[JaxprEqn], List[State], List[State], List[ConnectionPrim]]]:
        """Trace the computation that produces ``input_var`` for a group.

        This method performs a backward traversal from an input variable to find
        the source group, equations, states, and connections involved in producing
        that variable's value.

        Parameters
        ----------
        input_var : Var
            Variable consumed by post_group to trace back.
        groups : list[GroupPrim]
            All groups in the compilation.
        var_to_producer_eqn : dict[Var, JaxprEqn]
            Cached mapping from variables to producing equations.
        connections : list[tuple[JaxprEqn, ConnectionPrim]]
            Connection equations and their wrappers.

        Returns
        -------
        tuple or None
            If a valid projection is found, returns a 5-tuple:
            (pre_group, proj_eqns, proj_hidden_states, proj_in_states, proj_connections).
            Returns None if no valid projection exists (e.g., no source group found,
            or multiple source groups detected). Self-connections (pre_group == post_group)
            are fully supported.

        Notes
        -----
        Time complexity: O(E + V) where E is equations, V is variables.
        Uses breadth-first search to traverse the computation graph backward.
        """
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

        # Sort equations by original order
        proj_eqns = self._sort_equations_by_order(proj_eqns)

        return pre_group, proj_eqns, proj_hidden_states, proj_in_states, proj_connections

    def _merge_projection_data(
        self,
        existing_proj: ProjectionPrim,
        proj_eqns: List[JaxprEqn],
        proj_hidden_states: List[State],
        proj_in_states: List[State],
        proj_connections: List[ConnectionPrim],
    ) -> Tuple[List[JaxprEqn], List[State], List[State], List[ConnectionPrim]]:
        """Merge new projection data with existing projection.

        When multiple input_vars from the same pre→post group pair are found,
        this method combines their equations, states, and connections into a
        single unified projection.

        Parameters
        ----------
        existing_proj : ProjectionPrim
            Projection that already exists for this pre→post pair.
        proj_eqns : list[JaxprEqn]
            New equations to merge.
        proj_hidden_states : list[State]
            New hidden states to merge.
        proj_in_states : list[State]
            New input states to merge.
        proj_connections : list[ConnectionPrim]
            New connections to merge.

        Returns
        -------
        tuple
            4-tuple of (merged_eqns, merged_hidden_states, merged_in_states,
            merged_connections) with duplicates removed and equations sorted.

        Notes
        -----
        Preserves original equation order via sorting.
        """
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
        merged_eqns = self._sort_equations_by_order(merged_eqns)

        return merged_eqns, merged_hidden_states, merged_in_states, merged_connections

    def _build_projection_jaxpr(
        self,
        proj_eqns: List[JaxprEqn],
        proj_hidden_states: List[State],
        proj_in_states: List[State],
        post_group: GroupPrim,
        group_to_input_vars: Dict[int, Set[Var]],
    ) -> Tuple[ClosedJaxpr, List[Var]]:
        """Build a ClosedJaxpr for a projection.

        Constructs the jaxpr representing the projection computation, determining
        the correct input and output variables based on states and group dependencies.

        Parameters
        ----------
        proj_eqns : list[JaxprEqn]
            Equations that comprise the projection.
        proj_hidden_states : list[State]
            Hidden states from the source group.
        proj_in_states : list[State]
            Additional input states needed.
        post_group : GroupPrim
            Destination group consuming the projection outputs.
        group_to_input_vars : dict[int, set[Var]]
            Mapping from group IDs to their input variables.

        Returns
        -------
        tuple
            2-tuple of (proj_jaxpr, proj_outvars) where proj_jaxpr is the compiled
            ClosedJaxpr and proj_outvars are the output variables feeding into post_group.

        Notes
        -----
        InputPrim variable ordering: hidden_states vars, in_states vars, other vars.
        """
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

    def step5_build_inputs(self, groups: List[GroupPrim]) -> List[InputPrim]:
        """Create :class:`Input` descriptors for external variables.

        Parameters
        ----------
        groups : list[GroupPrim]
            Group descriptors receiving the inputs.

        Returns
        -------
        list[InputPrim]
            Input descriptors grouped by their destination group.
        """
        # Determine which vars are input_variables (not state vars)
        input_vars = []
        for var in self.jaxpr.invars:
            if var not in self.invar_to_state:
                input_vars.append(var)

        if not input_vars:
            return []

        # Use cached mapping: var -> equations that consume it
        var_to_consumer_eqns = self._var_to_consumer_eqns

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

        # GroupPrim traces by target group (use group id as key)
        group_id_to_traces = defaultdict(list)
        id_to_group = {}

        for input_var, target_group, equations, outvars in input_traces:
            group_id = id(target_group)
            id_to_group[group_id] = target_group
            group_id_to_traces[group_id].append((input_var, equations, outvars))

        # Create InputPrim objects for each group
        inputs = []
        input_idx = 0

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
            all_equations = self._sort_equations_by_order(all_equations)

            # Create the input ClosedJaxpr
            input_jaxpr = self._make_closed_jaxpr(
                eqns=all_equations,
                invars=all_input_vars,
                outvars=all_output_vars,
            )

            input_obj = InputPrim(
                jaxpr=input_jaxpr,
                group=group,
                name=f"Input_{input_idx}",
            )
            inputs.append(input_obj)
            input_idx += 1

        return inputs

    def _trace_input_forward(
        self,
        input_var: Var,
        groups: List[GroupPrim],
        var_to_consumer_eqns: Dict[Var, List[JaxprEqn]],
        group_input_vars_sets: Dict[int, Set[Var]],
    ) -> Optional[Tuple[Var, GroupPrim, List[JaxprEqn], List[Var]]]:
        """Forward-trace ``input_var`` until its values flow into a group boundary.

        Performs a forward traversal from an external input variable through
        intermediate computations until reaching a group's input boundary.

        Parameters
        ----------
        input_var : Var
            External input variable to trace forward.
        groups : list[GroupPrim]
            All groups to check as potential targets.
        var_to_consumer_eqns : dict[Var, list[JaxprEqn]]
            Cached mapping from variables to consuming equations.
        group_input_vars_sets : dict[int, set[Var]]
            Mapping from group IDs to their input variable sets.

        Returns
        -------
        tuple or None
            If the input flows into a group, returns a 4-tuple:
            (input_var, target_group, equations_in_path, output_vars).
            Returns None if the input doesn't flow into any group.

        Notes
        -----
        Time complexity: O(E + V) where E is equations, V is variables.
        Uses breadth-first search to traverse the computation graph forward.
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

    def step6_build_outputs(self, groups: List[GroupPrim]) -> List[OutputPrim]:
        """Describe how model outputs are assembled from group state variables.

        Parameters
        ----------
        groups : list[GroupPrim]
            Groups that may contribute to outputs.

        Returns
        -------
        list[OutputPrim]
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

        # Use cached mapping: var -> equation that produces it
        var_to_producer_eqn = self._var_to_producer_eqn

        # GroupPrim output vars by which group they depend on (use group id as key)
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
                        invar_idx = self.jaxpr.invars.index(var)
                        raise CompilationError(
                            f"OutputPrim variable {out_var} depends on external input {var} (jaxpr.invars[{invar_idx}]), "
                            f"which is not a state variable.\n"
                            f"  Suggestion: Outputs must only depend on state variables.\n"
                            f"  - Add an intermediate state to store {var}\n"
                            f"  - Or ensure the computation producing {out_var} only uses state variables\n"
                            f"  - Check that the function signature matches expected state inputs/outputs"
                        )
                    else:
                        raise CompilationError(
                            f"OutputPrim variable {out_var} depends on unknown variable {var}.\n"
                            f"  This variable is neither:\n"
                            f"    - A jaxpr input (len={len(self.jaxpr.invars)})\n"
                            f"    - Produced by any equation (total={len(self.jaxpr.eqns)})\n"
                            f"  This indicates a potential bug in the compilation process."
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
                        eqn_idx = self.eqn_to_id.get(id(eqn), -1)
                        raise CompilationError(
                            f"OutputPrim variable {out_var} depends on intermediate variable {in_var}\n"
                            f"  in equation {eqn_idx}: {eqn.primitive.name if hasattr(eqn.primitive, 'name') else str(eqn.primitive)}\n"
                            f"  Problem: {in_var} is not a state variable and not produced by output equations.\n"
                            f"  Suggestion:\n"
                            f"    - Ensure output computation only uses state variables\n"
                            f"    - Or create a state to store the intermediate result\n"
                            f"  Context: {len(equations_needed)} equations needed, {len(produced_vars)} vars produced"
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
                raise CompilationError(f"OutputPrim variable {out_var} does not depend on any group")
            elif len(dependent_groups) > 1:
                raise CompilationError(
                    f"OutputPrim variable {out_var} depends on multiple groups: {dependent_groups}. "
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

        # Create OutputPrim objects for each group
        outputs = []
        output_idx = 0
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
            all_equations = self._sort_equations_by_order(all_equations)

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

            output_obj = OutputPrim(
                jaxpr=output_jaxpr,
                hidden_states=all_dependent_hidden_states,
                in_states=all_dependent_in_states,
                group=group,
                name=f"Output_{output_idx}",
            )
            outputs.append(output_obj)
            output_idx += 1

        return outputs

    def step7_handle_remaining_equations(self) -> List[UnknownPrim]:
        """Handle equations that were not assigned to any known component.

        This method processes equations that remain unused after steps 1-6.
        It groups consecutive unused equations into Unknown objects and raises
        an error if unused equations have non-consecutive indices (indicating
        a potential compilation logic issue).

        Returns
        -------
        list[UnknownPrim]
            List of Unknown objects containing unassigned equations.
            Empty if all equations were assigned.

        Raises
        ------
        CompilationError
            If unused equations have non-consecutive indices, indicating
            that the compilation logic may have issues.
        """

        # Find unused equation indices
        unused_eqn_ids = set(self.eqn_to_id.keys()) - self.used_eqn_ids
        if not unused_eqn_ids:
            return []

        # Convert to sorted indices
        unused_indices = sorted([self.eqn_to_id[eqn_id] for eqn_id in unused_eqn_ids])

        # GroupPrim consecutive indices
        index_groups = _group_consecutive_indices(unused_indices)

        all_unknown_objs = []
        unknown_idx = 0
        for index_group in index_groups:

            # Check for non-consecutive indices (more than one group)
            if len(index_group) < 2:
                # Format the groups for the error message
                groups_str = ", ".join([f"[{g[0]}..{g[-1]}]" if len(g) > 1 else f"[{g[0]}]" for g in index_groups])

                # Show all equation primitives from each group
                sample_details = []
                for group_indices in index_groups:
                    # Collect all primitive names in this group
                    primitives = []
                    for idx in group_indices:
                        eqn = self.jaxpr.eqns[idx]
                        primitives.append(f"[{idx}] {eqn.primitive.name}")

                    # Format group header
                    group_range = f"[{group_indices[0]}..{group_indices[-1]}]" if len(
                        group_indices) > 1 else f"[{group_indices[0]}]"

                    # Format primitives list (one per line if more than 3, comma-separated otherwise)
                    if len(primitives) <= 3:
                        primitives_str = ", ".join(primitives)
                        sample_details.append(f"  GroupPrim {group_range}: {primitives_str}")
                    else:
                        primitives_lines = "\n    ".join(primitives)
                        sample_details.append(f"  GroupPrim {group_range}:\n    {primitives_lines}")

                raise CompilationError(
                    f"Unused equations have non-consecutive indices, indicating a potential compilation issue.\n"
                    f"  Total unused equations: {len(unused_indices)}\n"
                    f"  Index groups: {groups_str}\n"
                    f"  Sample equations from each group:\n" +
                    "\n".join(sample_details) + "\n"
                                                f"  Suggestion:\n"
                                                f"    - Non-consecutive unused equations suggest that some computations were\n"
                                                f"      partially traced while others were missed.\n"
                                                f"    - Review the compilation steps to ensure all related computations\n"
                                                f"      are being captured together."
                )

            # Single consecutive group - create UnknownPrim object
            consecutive_eqns = [self.jaxpr.eqns[idx] for idx in index_group]

            # Collect all input and output variables for the unknown block
            all_invars = set()
            all_outvars = []
            for eqn in consecutive_eqns:
                for in_var in eqn.invars:
                    if isinstance(in_var, Var):
                        all_invars.add(in_var)
                all_outvars.extend(eqn.outvars)

            # Remove variables that are produced within the block from invars
            produced_vars = set(all_outvars)
            external_invars = [v for v in all_invars if v not in produced_vars]

            # Sort external invars by their first occurrence in the equations
            invar_order = {}
            for eqn in consecutive_eqns:
                for in_var in eqn.invars:
                    if isinstance(in_var, Var) and in_var in external_invars and in_var not in invar_order:
                        invar_order[in_var] = len(invar_order)
            external_invars = sorted(external_invars, key=lambda v: invar_order.get(v, float('inf')))

            # Create ClosedJaxpr for the unknown block
            unknown_jaxpr = self._make_closed_jaxpr(
                eqns=consecutive_eqns,
                invars=list(external_invars),
                outvars=all_outvars,
            )

            all_unknown_objs.append(
                UnknownPrim(
                    jaxpr=unknown_jaxpr,
                    eqn_indices=tuple(index_group),
                    name=f"Unknown_{unknown_idx}",
                )
            )
            unknown_idx += 1

            # Issue a warning about unassigned equations
            warnings.warn(
                f"Found {len(index_group)} unassigned equations "
                f"(indices {index_group[0]}..{index_group[-1]}) "
                f"that were packaged into an UnknownPrim block. "
                f"\n\n"
                f"{unknown_jaxpr}"
                f"\n\n"
                f"This may indicate computations that couldn't be "
                f"classified as GroupPrim, ProjectionPrim, InputPrim, or OutputPrim.",
                UserWarning,
                stacklevel=2,
            )

        return all_unknown_objs

    def step8_build_graph(
        self,
        groups: List[GroupPrim],
        projections: List[ProjectionPrim],
        inputs: List[InputPrim],
        outputs: List[OutputPrim],
        unknowns: List[UnknownPrim],
    ) -> NeuroGraph:
        """Derive an execution graph that preserves the original equation order.

        This method builds a directed acyclic graph (DAG) where nodes represent
        computational components and edges represent data dependencies. The graph
        construction process:

        1. Maps all equations to their containing components (Groups, Projections, Unknowns)
        2. Tracks which component produces each variable
        3. Creates dependency edges when a component consumes variables produced by another
        4. Adds structural edges for InputPrim->Group, Group->ProjectionPrim->Group, and Group->Output

        Unknown nodes are integrated into the dependency graph automatically through
        the variable tracking mechanism. If an Unknown node's equations consume variables
        produced by a Group or ProjectionPrim, a dependency edge is created. Similarly, if
        a Group or ProjectionPrim consumes variables produced by an Unknown node, the reverse
        dependency is established.

        Parameters
        ----------
        groups : list[GroupPrim]
            Computation blocks that produce state updates.
        projections : list[ProjectionIR]
            ConnectionIR pipelines between groups.
        inputs : list[InputIR]
            External inputs to the network.
        outputs : list[OutputPrim]
            Objects describing how observable values are extracted.
        unknowns : list[UnknownPrim]
            Unknown computation blocks that couldn't be classified.

        Returns
        -------
        NeuroGraph
            Directed acyclic graph with nodes ordered for execution/visualization.
            Nodes appear in the order they were first encountered during equation
            processing, which preserves the original jaxpr execution order.

        Notes
        -----
        The dependency tracking mechanism works as follows:

        - For each equation in the original jaxpr (processed in order):
          - Identify which component owns this equation
          - For each input variable to the equation:
            - Check if another component produced this variable
            - If yes, add edge: producer -> current component
          - Record that this component produces the equation's output variables

        This approach ensures that:
        1. Unknown nodes are properly positioned in the execution order
        2. Dependencies between Unknowns and other components are explicit
        3. The graph respects the original jaxpr's data flow
        """
        graph = NeuroGraph()

        # Ensure inputs come first
        for inp in inputs:
            graph.add_node(inp)

        # Create a mapping from equations to components
        # This includes all component types: Groups, Projections, and Unknowns
        eqn_to_component: Dict[int, GraphElem] = {}
        var_to_component: Dict[Var, GraphElem] = {}

        # Map all equations to their containing components
        for group in groups:
            for eqn in group.jaxpr.jaxpr.eqns:
                eqn_to_component[id(eqn)] = group
        for proj in projections:
            for eqn in proj.jaxpr.jaxpr.eqns:
                eqn_to_component[id(eqn)] = proj
        for unknown in unknowns:
            for eqn in unknown.jaxpr.jaxpr.eqns:
                eqn_to_component[id(eqn)] = unknown

        # Process equations in original jaxpr order to build dependency graph
        # This automatically handles UnknownPrim dependencies through variable tracking
        seen_components = set()
        for eqn in self.jaxpr.eqns:
            component = eqn_to_component.get(id(eqn))
            if component is None:
                # Skip equations not assigned to any component (shouldn't happen after step7)
                continue

            component_id = id(component)
            if component_id not in seen_components:
                seen_components.add(component_id)
                graph.add_node(component)

            # Build dependency edges based on variable producers
            # This handles dependencies for ALL component types including UnknownPrim nodes
            for in_var in eqn.invars:
                if isinstance(in_var, Var):
                    producer = var_to_component.get(in_var)
                    if producer is not None and producer is not component:
                        # Add edge from producer to consumer
                        # This creates dependencies for UnknownPrim nodes with Groups/Projections
                        # and vice versa based on actual data flow
                        graph.add_edge(producer, component)

            # Record which component produces each output variable
            # This allows subsequent components (including Unknowns) to track their dependencies
            for out_var in eqn.outvars:
                if isinstance(out_var, Var):
                    var_to_component[out_var] = component

        # Outputs are appended to maintain display parity with previous behavior
        for out in outputs:
            graph.add_node(out)

        # Add structural dependencies based on component metadata
        # These edges represent logical relationships beyond data flow
        for inp in inputs:
            graph.add_edge(inp, inp.group)
        for proj in projections:
            graph.add_edge(proj.pre_group, proj)
            graph.add_edge(proj, proj.post_group)
        for out in outputs:
            graph.add_edge(out.group, out)

        return graph

    def step9_validate_compilation(
        self,
        groups: List[GroupPrim],
        projections: List[ProjectionPrim],
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

        def _validate_hidden_state_belong_to_group():
            hidden_states = set(s for s in self.out_states if s in self.in_states)
            for state in hidden_states:
                if id(state) not in all_group_hidden_states:
                    state_name = get_hidden_name(state)
                    raise CompilationError(
                        f"Hidden state '{state_name}' (id={id(state)}) is not assigned to any group.\n"
                        f"  Total groups: {len(groups)}\n"
                        f"  Total hidden states: {len(hidden_states)}\n"
                        f"  States assigned to groups: {len(all_group_hidden_states)}\n"
                        f"  Suggestion: Verify that state dependency analysis (step1) correctly identified all states."
                    )

        # Check 2: Projections should have non-empty connections
        def _validate_projection_connections():
            for idx, proj in enumerate(projections):
                if not proj.connections:
                    pre_name = proj.pre_group.name
                    post_name = proj.post_group.name
                    raise CompilationError(
                        f"ProjectionPrim {idx} from '{pre_name}' to '{post_name}' has no connections.\n"
                        f"  This may indicate:\n"
                        f"    - No connection primitives found between these groups\n"
                        f"    - ConnectionPrim extraction (step3) failed to identify connections\n"
                        f"  Suggestion: Check that connection primitives are properly marked with _is_connection()"
                    )

        # Check 3: ProjectionPrim hidden_states should belong to exactly one group
        def _validate_hidden_state_belong_to_one_group():
            for proj_idx, proj in enumerate(projections):
                for state in proj.hidden_states:
                    count = sum(1 for g in groups if g.has_hidden_state(state))
                    state_name = get_hidden_name(state)
                    if count == 0:
                        raise CompilationError(
                            f"ProjectionPrim {proj_idx} depends on state '{state_name}' which is not in any group.\n"
                            f"  Total groups: {len(groups)}\n"
                            f"  Suggestion: This state should be added to a group's hidden_states "
                            f"during group building (step2)."
                        )
                    elif count > 1:
                        group_names = [g.name for g in groups if g.has_hidden_state(state)]
                        raise CompilationError(
                            f"ProjectionPrim {proj_idx} depends on state '{state_name}' which belongs to {count} groups.\n"
                            f"  Groups: {', '.join(group_names)}\n"
                            f"  Suggestion: Each state should belong to exactly one group.\n"
                            f"  This may indicate an error in state dependency analysis (step1)."
                        )

        # Check 4: All equations should be used
        def _validate_all_equations_used():
            """Validate that all equations have been used.

            Note: This should not trigger in normal operation since step6b handles
            remaining equations. If this does trigger, it indicates a bug in step6b.
            """
            unused_eqn_ids = set(self.eqn_to_id.keys()) - self.used_eqn_ids
            if unused_eqn_ids:
                unused_indices = sorted([self.eqn_to_id[eqn_id] for eqn_id in unused_eqn_ids])
                # Show first few unused equations with details
                sample_indices = unused_indices
                sample_details = []
                for idx in sample_indices:
                    eqn = self.jaxpr.eqns[idx]
                    sample_details.append(f"    [{idx}] {eqn}")
                details = "\n".join(sample_details)

                raise CompilationError(
                    f"Internal error: Unused equations detected after step6b.\n"
                    f"  This indicates a bug in step6b_handle_remaining_equations().\n"
                    f"  Total equations: {len(self.jaxpr.eqns)}\n"
                    f"  Used equations: {len(self.used_eqn_ids)}\n"
                    f"  Unused equations: {len(unused_eqn_ids)}\n"
                    f"  Unused equation indices: {unused_indices}\n"
                    f"  Unused equations:\n{details}\n"
                    f"  Suggestion:\n"
                    f"    - This should not happen. Please report this as a bug.\n"
                    f"    - Step6b should have either packaged these equations into UnknownPrim objects\n"
                    f"      or raised an error about non-consecutive indices."
                )

        # Validation dispatch - run requested validation checks
        validation_functions = {
            'hidden_state_belong_to_group': _validate_hidden_state_belong_to_group,
            'projection_connections': _validate_projection_connections,
            'hidden_state_belong_to_one_group': _validate_hidden_state_belong_to_one_group,
            'all_equations_used': _validate_all_equations_used,
        }

        # If 'all' is specified, run all validations
        if 'all' in self.validation:
            for validate_fn in validation_functions.values():
                validate_fn()
        else:
            # Run only specified validations
            for val_name in self.validation:
                if val_name in validation_functions:
                    validation_functions[val_name]()
                else:
                    available = ', '.join(sorted(validation_functions.keys()))
                    raise ValueError(
                        f"UnknownPrim validation '{val_name}'. "
                        f"Available validations: {available}, 'all'"
                    )

    def compile(self) -> NeuroGraph:
        """Execute the complete compilation pipeline.

        Returns
        -------
        NeuroGraph
            A directed graph containing all compiled elements.
            Groups, projections, inputs, outputs, and unknowns can be
            accessed through the graph's properties:

            - graph.groups
            - graph.projections
            - graph.inputs
            - graph.outputs
            - graph.unknowns

        Raises
        ------
        CompilationError
            If any validation step fails.
        """
        # Step 1: Analyze state dependencies and group states
        state_groups = self.step1_analyze_state_dependencies()

        # Step 2: Build GroupPrim objects
        groups = self.step2_build_groups(state_groups)

        # Step 3: Extract connections
        connections = self.step3_extract_connections()

        # Step 4: Build ProjectionPrim objects
        projections = self.step4_build_projections(groups, connections)

        # Step 5: Build InputPrim objects
        inputs = self.step5_build_inputs(groups)

        # Step 6: Build OutputPrim objects
        outputs = self.step6_build_outputs(groups)

        # Step 7: Handle remaining equations (new step)
        unknowns = self.step7_handle_remaining_equations()

        # Step 8: Determine call order
        graph = self.step8_build_graph(groups, projections, inputs, outputs, unknowns)

        # Step 9: Validate the compilation
        self.step9_validate_compilation(groups, projections)

        return graph


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
) -> NeuroGraph:
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
    NeuroGraph
        A directed graph containing all compiled elements.
        Access components via: graph.groups, graph.projections, graph.inputs,
        graph.outputs, graph.unknowns.

    Raises
    ------
    CompilationError
        If the ClosedJaxpr violates the IR assumptions (e.g. outputs depend on
        multiple groups, or unused equations have non-consecutive indices).
    """
    compiler = NeuronIRCompiler(
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
    validation: Optional[Union[str, Sequence[str]]] = 'all',
) -> Callable[..., CompiledGraphIR]:
    """Create a compiler that compiles ``stateful_fn`` into graph IR.

    Parameters
    ----------
    target : StatefulFunction, Callable
        Stateful function or callable target to compile.
    jit_inline : bool, optional
        When ``True`` the compiler inlines JIT-wrapped connection primitives
        before compilation.
    validation : str, Sequence[str], optional
        Validation steps to perform after compilation. If ``None``, no validation
        is performed. If ``'all'``, all validation steps are performed. Otherwise,
        a sequence of validation step names can be provided.

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
        graph = NeuronIRCompiler(
            closed_jaxpr=jaxpr,
            in_states=in_states,
            out_states=out_states,
            invar_to_state=state_mapping['invar_to_state'],
            outvar_to_state=state_mapping['outvar_to_state'],
            state_to_invars=state_mapping['state_to_invars'],
            state_to_outvars=state_mapping['state_to_outvars'],
            validation=validation,
        ).compile()

        # cache key
        cache_key = stateful_fn.get_arg_cache_key(*args, **kwargs)

        return CompiledGraphIR(
            static_argnums=stateful_fn.static_argnums,
            static_argnames=stateful_fn.static_argnames,
            out_treedef=stateful_fn.get_out_treedef_by_cache(cache_key),
            cache_key=cache_key,
            jaxpr=jaxpr,
            in_states=in_states,
            out_states=out_states,
            write_states=stateful_fn.get_state_trace_by_cache(cache_key).get_write_states(True),
            invar_to_state=state_mapping['invar_to_state'],
            outvar_to_state=state_mapping['outvar_to_state'],
            state_to_invars=state_mapping['state_to_invars'],
            state_to_outvars=state_mapping['state_to_outvars'],
            graph=graph,
        )

    return call
