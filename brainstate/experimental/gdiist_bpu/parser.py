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

from typing import Tuple

import jax

from brainstate.transform._make_jaxpr import StatefulFunction
from .component import Node, Connection
from .utils import _is_connection, eqns_to_jaxpr, find_in_states, find_out_states


class Parser:
    def __init__(
        self,
        stateful_fn: StatefulFunction,
        inputs: Tuple,
    ):
        assert isinstance(stateful_fn, StatefulFunction), "stateful_fn must be an instance of StatefulFunction"
        self.stateful_fn = stateful_fn
        self.inputs = inputs

        self.operations = []
        self.connections = []
        self.invars_to_state = dict()
        self.state_to_invars = dict()
        self.outvars_to_state = dict()
        self.state_to_outvars = dict()

        # Track current operation state
        self.current_operation_eqns = []
        self.current_operation_name = None
        self.operation_counter = 0

    def _build_state_mappings(self, closed_jaxpr, states, out_shapes):
        """
        Build mappings between state variables and JAXpr input/output variables

        This implementation is inspired by brainscale's _etrace_compiler_module_info.py
        which provides more accurate state mapping by considering the actual model structure
        """

        # Clear previous mappings
        self.invars_to_state.clear()
        self.state_to_invars.clear()
        self.outvars_to_state.clear()
        self.state_to_outvars.clear()

        # Extract the actual jaxpr from ClosedJaxpr
        jaxpr = closed_jaxpr.jaxpr

        # Get output shapes to understand the structure
        out_shapes = out_shapes[0]

        # Flatten inputs and outputs to understand structure
        # This follows the pattern from brainscale
        state_vals = [state.value for state in states]
        out_avals, _ = jax.tree.flatten(out_shapes)
        num_out = len(out_avals)

        # Get state structure information
        state_avals, state_tree = jax.tree.flatten(state_vals)
        num_inputs_before_states = len(jaxpr.invars) - len(state_avals)

        # Map state tree to invars and outvars
        # Input variables: the last len(state_avals) invars correspond to states
        state_tree_invars = jax.tree.unflatten(state_tree, jaxpr.invars[num_inputs_before_states:])

        # Output variables: after the main outputs, the rest correspond to state updates
        state_tree_outvars = jax.tree.unflatten(state_tree, jaxpr.outvars[num_out:])

        # Build mappings using the tree structure
        # This ensures proper correspondence between states and their JAXpr variables
        for state, invar, outvar in zip(states, state_tree_invars, state_tree_outvars):
            # Always flatten the tree structure to get individual variables
            invar_leaves = jax.tree.leaves(invar)
            outvar_leaves = jax.tree.leaves(outvar)

            # Store the relationships
            for var in invar_leaves:
                self.invars_to_state[var] = state
            for var in outvar_leaves:
                self.outvars_to_state[var] = state

            # Store the reverse mappings
            if len(invar_leaves) == 1:
                self.state_to_invars[state] = invar_leaves[0]
            else:
                self.state_to_invars[state] = invar_leaves

            if len(outvar_leaves) == 1:
                self.state_to_outvars[state] = outvar_leaves[0]
            else:
                self.state_to_outvars[state] = outvar_leaves

    def _should_split_operation(self, eqn):
        """Check if equation should cause a split in current operation"""
        # slice operations should cause a split and become their own operation
        if eqn.primitive.name == 'slice':
            return True

        # brainevent jit connections should cause a split (but don't become operations themselves)
        if _is_connection(eqn):
            return True

        # All other operations can be merged into the current operation
        return False

    def _finalize_current_operation(self):
        """Finalize the current operation and add it to operations list"""
        if self.current_operation_eqns:
            jaxpr = eqns_to_jaxpr(self.current_operation_eqns)
            operation_name = f"node{self.operation_counter}"

            operation = Node(
                name=operation_name,
                jaxpr=jaxpr,
                in_states=find_in_states(self.invars_to_state, jaxpr.invars),
                out_states=find_out_states(self.outvars_to_state, jaxpr.outvars),
            )
            self.operations.append(operation)
            self.operation_counter += 1
            self.current_operation_eqns.clear()
            return operation
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

    def _find_operation_by_variable(self, var, expanded_eqns):
        """Find which operation produces or consumes a given variable"""
        for operation in self.operations:
            for eqn in operation.jaxpr.eqns:
                # Check if this equation produces the variable
                if var in eqn.outvars:
                    return operation
        return None

    def _find_operations_using_variable(self, var, expanded_eqns):
        """Find which operations use a given variable as input"""
        using_operations = []
        for operation in self.operations:
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
        jaxpr = closed_jaxpr.jaxpr
        # First expand all nested JAXpr
        expanded_eqns = []
        for eqn in jaxpr.eqns:
            expanded_eqns.extend(self._expand_nested_jaxpr(eqn))

        # First pass: create operations and identify connections
        for eqn in expanded_eqns:
            # Check if this equation should split the current operation
            if self._should_split_operation(eqn):

                # Finalize current operation before the split
                self._finalize_current_operation()

                # If it's a brainevent connection, don't add to operations - we'll handle separately
                if _is_connection(eqn):
                    continue

                # If it's a slice, add it as a new operation
                elif eqn.primitive.name == 'slice':
                    self.current_operation_eqns.append(eqn)

            else:
                # Regular equation - add to current operation
                self.current_operation_eqns.append(eqn)

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
                    producer = self._find_operation_by_variable(input_var, expanded_eqns)
                    if producer:
                        pre_operation = producer
                        break  # Use the first producer we find

                # Find post_operations: which operations use the output variables from this connection
                for output_var in eqn.outvars:
                    consumers = self._find_operations_using_variable(output_var, expanded_eqns)
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

    def parse(self):
        jaxpr = self.stateful_fn.get_jaxpr(*self.inputs[0], **self.inputs[1])
        states = self.stateful_fn.get_states(*self.inputs[0], **self.inputs[1])
        out_shapes = self.stateful_fn.get_out_shapes(*self.inputs[0], **self.inputs[1])

        # Build state mappings
        self._build_state_mappings(jaxpr, states, out_shapes)
        state_mapping = {
            'invars_to_state': self.invars_to_state,
            'state_to_invars': self.state_to_invars,
            'outvars_to_state': self.outvars_to_state,
            'state_to_outvars': self.state_to_outvars,
        }

        # Parse equations to identify groups and connections
        self._parse_equations(jaxpr)

        return self.operations, self.connections, state_mapping
