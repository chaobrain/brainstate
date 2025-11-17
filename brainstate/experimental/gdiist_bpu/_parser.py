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

from typing import Tuple, Dict

import jax

from brainstate.transform._ir_inline_jit import inline_jit
from brainstate.transform._make_jaxpr import StatefulFunction
from ._data import Dynamics, Connection
from ._utils import _is_connection, eqns_to_jaxpr, find_in_states, find_out_states


class ParserV2:
    """
    Parser for BPU (second generation) operations and connections.
    """

    def __init__(
        self,
        stateful_fn: StatefulFunction,
        inputs: Tuple,
        jit_inline: bool = True,
    ):
        assert isinstance(stateful_fn, StatefulFunction), "stateful_fn must be an instance of StatefulFunction"
        self.stateful_fn = stateful_fn
        assert stateful_fn.return_only_write, (
            "Parser currently only supports stateful functions that return only write states. "
        )
        self.inputs = inputs

        self.nodes = []
        self.connections = []

        # Track current operation state
        self.current_eqns = []
        self.current_operation_name = None
        self.node_counter = 0

        # jaxpr
        jaxpr = self.stateful_fn.get_jaxpr(*self.inputs[0], **self.inputs[1])
        if jit_inline:
            jaxpr = inline_jit(jaxpr, _is_connection)

        # Build state mappings
        in_states = self.stateful_fn.get_states(*self.inputs[0], **self.inputs[1])
        out_states = self.stateful_fn.get_write_states(*self.inputs[0], **self.inputs[1])
        self.state_mapping = _build_state_mapping(jaxpr, in_states, out_states)

        # Parse equations to identify groups and connections
        self._parse_equations(jaxpr)

    @property
    def invar_to_state(self):
        return self.state_mapping['invar_to_state']

    @property
    def state_to_invars(self):
        return self.state_mapping['state_to_invars']

    @property
    def outvar_to_state(self):
        return self.state_mapping['outvar_to_state']

    @property
    def state_to_outvars(self):
        return self.state_mapping['state_to_outvars']

    @property
    def in_states(self):
        return self.state_mapping['in_states']

    @property
    def out_states(self):
        return self.state_mapping['out_states']

    def _finalize_current_operation(self):
        """Finalize the current operation and add it to operations list"""
        if len(self.current_eqns) > 0:
            jaxpr = eqns_to_jaxpr(self.current_eqns)
            node = Dynamics(
                jaxpr=jaxpr,
                in_states=find_in_states(self.invar_to_state, jaxpr.invars),
                out_states=find_out_states(self.outvar_to_state, jaxpr.outvars),
            )
            self.nodes.append(node)
            self.node_counter += 1
            self.current_eqns.clear()
            return node
        return None

    def _expand_nested_jaxpr(self, eqn):
        """Expand nested JAXpr (like jit) by replacing them with their inner equations

        Exception: brainevent jit operations are kept as-is for connection detection
        """
        expanded_eqns = []

        # if is_jit_primitive(eqn):
        #     if _is_connection(eqn):
        #         expanded_eqns.append(eqn)
        #
        #     # Expand other jit operations
        #     elif 'jaxpr' in eqn.params:
        #         nested_jaxpr = eqn.params['jaxpr']
        #         # Recursively process nested equations
        #         for nested_eqn in nested_jaxpr.eqns:
        #             expanded_eqns.extend(self._expand_nested_jaxpr(nested_eqn))
        #     else:
        #         expanded_eqns.append(eqn)
        #
        # elif _is_connection(eqn):
        #     expanded_eqns.append(eqn)
        #
        # else:
        #     expanded_eqns.append(eqn)

        if _is_connection(eqn):
            expanded_eqns.append(eqn)

        else:
            expanded_eqns.append(eqn)

        return expanded_eqns

    def _find_node_has_outvar(self, var):
        """Find which operation produces or consumes a given variable"""
        for node in self.nodes:
            node: Dynamics
            if node.has_out_var(var):
                return node
        return None

    def _find_operations_using_variable(self, var):
        """Find which operations use a given variable as input"""
        using_operations = []
        for operation in self.nodes:
            for eqn in operation.jaxpr.eqns:
                # Check if this equation uses the variable as input
                if var in eqn.invars:
                    using_operations.append(operation)
                    break  # Don't add the same operation multiple times
        return using_operations

    def _create_connection(self, pre_operation, post_operation, jit_eqn):
        """Create a connection between two operations using the inner jaxpr from jit"""
        # Extract the inner jaxpr from the jit equation
        if 'jaxpr' in jit_eqn.params:
            inner_jaxpr = jit_eqn.params['jaxpr'].jaxpr
        else:
            # Fallback - use the jit equation itself
            inner_jaxpr = jit_eqn

        # Avoid duplicate connections
        for existing_conn in self.connections:
            if existing_conn.pre == pre_operation and existing_conn.post == post_operation:
                return existing_conn

        # Create new connection
        connection = Connection(pre=pre_operation, post=post_operation, jaxpr=inner_jaxpr)
        self.connections.append(connection)
        return connection

    def _parse_equations(self, closed_jaxpr):
        """Parse JAXpr equations to identify operations and connections"""

        # Extract the actual jaxpr from ClosedJaxpr
        # First expand all nested JAXpr
        jaxpr = closed_jaxpr.jaxpr
        expanded_eqns = []
        for eqn in jaxpr.eqns:
            expanded_eqns.extend(self._expand_nested_jaxpr(eqn))

        # First pass: create operations and identify connections
        for eqn in expanded_eqns:
            # Check if this equation should split the current operation
            if _should_split_operation(eqn):

                # Finalize current operation before the split
                self._finalize_current_operation()

                # If it's a brainevent connection, don't add to operations - we'll handle separately
                if _is_connection(eqn):
                    continue

            else:
                # Regular equation - add to current operation
                self.current_eqns.append(eqn)

        # Finalize the last operation
        self._finalize_current_operation()

        # Second pass: create connections based on data flow analysis
        for i, eqn in enumerate(expanded_eqns):
            if _is_connection(eqn):
                # Analyze data flow for this connection
                pre_operation = None
                post_operations = []

                # Find pre_operation: which operation produces the input variables for this connection
                for input_var in eqn.invars:
                    producer = self._find_node_has_outvar(input_var)
                    if producer is not None:
                        pre_operation = producer
                        break  # Use the first producer we find

                # Find post_operations: which operations use the output variables from this connection
                for output_var in eqn.outvars:
                    consumers = self._find_operations_using_variable(output_var)
                    post_operations.extend(consumers)

                # Remove duplicates from post_operations
                seen = set()
                unique_post_operations = []
                for op in post_operations:
                    if id(op) not in seen:
                        seen.add(id(op))
                        unique_post_operations.append(op)
                post_operations = unique_post_operations

                # Create connections from pre_operation to each post_operation
                if pre_operation is not None:
                    for post_operation in post_operations:
                        if post_operation != pre_operation:  # Avoid self-connections
                            self._create_connection(pre_operation, post_operation, eqn)


def _should_split_operation(eqn):
    """Check if equation should cause a split in current operation"""
    # slice operations should cause a split and become their own operation
    if eqn.primitive.name == 'slice':
        return True

    # brainevent jit connections should cause a split (but don't become operations themselves)
    if _is_connection(eqn):
        return True

    # All other operations can be merged into the current operation
    return False


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
    }
