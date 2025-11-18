# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
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

from typing import Tuple, Dict, NamedTuple, Sequence

import jax

from brainstate._compatible_import import Jaxpr, Var
from brainstate._state import State
from brainstate.transform._ir_inline_jit import inline_jit
from brainstate.transform._make_jaxpr import StatefulFunction
from ._compiler import compile
from ._data import CompiledGraph
from ._utils import _is_connection


class ParseOutput(NamedTuple):
    jaxpr: Jaxpr
    in_states: Sequence[State]
    out_states: Sequence[State]
    invar_to_state: Dict[Var, State]
    outvar_to_state: Dict[Var, State]
    state_to_invars: Dict[State, Sequence[Var]]
    state_to_outvars: Dict[State, Sequence[Var]]
    compiled: CompiledGraph


def parse(
    stateful_fn: StatefulFunction,
    inputs: Tuple,
    jit_inline: bool = True,
):
    assert isinstance(stateful_fn, StatefulFunction), "stateful_fn must be an instance of StatefulFunction"
    assert stateful_fn.return_only_write, (
        "Parser currently only supports stateful functions that return only write states. "
    )

    # jaxpr
    jaxpr = stateful_fn.get_jaxpr(*inputs[0], **inputs[1])
    if jit_inline:
        jaxpr = inline_jit(jaxpr, _is_connection)

    # Build state mappings
    in_states = stateful_fn.get_states(*inputs[0], **inputs[1])
    out_states = stateful_fn.get_write_states(*inputs[0], **inputs[1])
    state_mapping = _build_state_mapping(jaxpr, in_states, out_states)

    # Compile the SNN
    compiled_snn = compile(
        closed_jaxpr=jaxpr,
        in_states=in_states,
        out_states=out_states,
        invar_to_state=state_mapping['invar_to_state'],
        outvar_to_state=state_mapping['outvar_to_state'],
        state_to_invars=state_mapping['state_to_invars'],
        state_to_outvars=state_mapping['state_to_outvars'],
    )
    return ParseOutput(
        jaxpr=jaxpr,
        in_states=in_states,
        out_states=out_states,
        invar_to_state=state_mapping['invar_to_state'],
        outvar_to_state=state_mapping['outvar_to_state'],
        state_to_invars=state_mapping['state_to_invars'],
        state_to_outvars=state_mapping['state_to_outvars'],
        compiled=compiled_snn,
    )


def _build_state_mapping(closed_jaxpr, in_states, out_states) -> Dict:
    # Clear previous mappings
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
        if len(invar_leaves) == 1:
            state_to_invars[state] = invar_leaves[0]
        else:
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
        if len(outvar_leaves) == 1:
            state_to_outvars[state] = outvar_leaves[0]
        else:
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
