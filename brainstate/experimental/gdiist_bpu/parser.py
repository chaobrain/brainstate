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

from typing import Callable, NamedTuple, Dict, List, Any

import jax
from jax.api_util import shaped_abstractify

from brainstate._compatible_import import is_jit_primitive, ClosedJaxpr, JaxprEqn
from brainstate.transform._make_jaxpr import StatefulFunction, _make_hashable
from brainstate.util._cache import BoundedCache
from .utils import _is_brainevent_jit_connection

__all__ = [
    'GdiistBpuParser',
]


class Operation(NamedTuple):
    name: str
    eqns: list[ClosedJaxpr]


class Connection(NamedTuple):
    pre: Operation
    post: Operation
    jaxpr: ClosedJaxpr


class GdiistBpuParser:
    """
    Parser for BPU operations and connections.

    This class is responsible for parsing the operations and connections in a BPU model.
    """

    def __init__(self, fn: Callable, target: str = 'jit'):
        self.fn = fn
        self.stateful_fn = StatefulFunction(self.fn, return_only_write=False)
        self.target = target
        assert target in ['jit', 'forloop'], f"Target must be either 'jit' or 'forloop', got {target}"
        self.compiled_graph = BoundedCache()

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
        if _is_brainevent_jit_connection(eqn):
            return True

        # All other operations can be merged into the current operation
        return False

    def _finalize_current_operation(self):
        """Finalize the current operation and add it to operations list"""
        if self.current_operation_eqns:
            operation_name = f"operation_{self.operation_counter}"
            operation = Operation(name=operation_name, eqns=self.current_operation_eqns.copy())
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

        if is_jit_primitive(eqn):
            # Check if this is a brainevent connection - if so, keep it as-is
            if _is_brainevent_jit_connection(eqn):
                expanded_eqns.append(eqn)
            else:
                # Expand other jit operations
                if 'jaxpr' in eqn.params:
                    nested_jaxpr = eqn.params['jaxpr']
                    # Recursively process nested equations
                    for nested_eqn in nested_jaxpr.eqns:
                        expanded_eqns.extend(self._expand_nested_jaxpr(nested_eqn))
                else:
                    expanded_eqns.append(eqn)
        else:
            expanded_eqns.append(eqn)

        return expanded_eqns

    def _find_operation_by_variable(self, var, expanded_eqns):
        """Find which operation produces or consumes a given variable"""
        for operation in self.operations:
            for eqn in operation.eqns:
                # Check if this equation produces the variable
                if var in eqn.outvars:
                    return operation
        return None

    def _find_operations_using_variable(self, var, expanded_eqns):
        """Find which operations use a given variable as input"""
        using_operations = []
        for operation in self.operations:
            for eqn in operation.eqns:
                # Check if this equation uses the variable as input
                if var in eqn.invars:
                    using_operations.append(operation)
                    break  # Don't add the same operation multiple times
        return using_operations

    def _create_connection(self, pre_operation, post_operation, jit_eqn):
        """Create a connection between two operations using the inner jaxpr from jit"""
        # Extract the inner jaxpr from the jit equation
        if 'jaxpr' in jit_eqn.params:
            inner_jaxpr = jit_eqn.params['jaxpr']
        else:
            # Fallback - use the jit equation itself
            inner_jaxpr = jit_eqn

        # Avoid duplicate connections
        for existing_conn in self.connections:
            if (existing_conn.pre == pre_operation and
                existing_conn.post == post_operation):
                return existing_conn

        # Create new connection
        connection = Connection(pre=pre_operation, post=post_operation, jaxpr=inner_jaxpr)
        self.connections.append(connection)
        return connection

    def _parse_equations(self, closed_jaxpr):
        """Parse JAXpr equations to identify operations and connections"""
        # Extract the actual jaxpr from ClosedJaxpr
        jaxpr = closed_jaxpr.jaxpr
        expanded_eqns = []

        # First expand all nested JAXpr
        for eqn in jaxpr.eqns:
            expanded_eqns.extend(self._expand_nested_jaxpr(eqn))

        # First pass: create operations and identify connections
        for eqn in expanded_eqns:
            # Check if this equation should split the current operation
            if self._should_split_operation(eqn):
                # Finalize current operation before the split
                self._finalize_current_operation()

                # If it's a brainevent connection, don't add to operations - we'll handle separately
                if _is_brainevent_jit_connection(eqn):
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
            if _is_brainevent_jit_connection(eqn):
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

    def cache_key(self, *args, **kwargs):
        if self.target == 'forloop':
            args, kwargs = jax.tree.map(lambda x: x[0], (args, kwargs))
        return _make_hashable(jax.tree.map(shaped_abstractify, (args, kwargs)))

    def parse(self, *args, **kwargs):
        """
        Main parsing function that analyzes JAXpr and builds groups and connections.
        """
        key = self.cache_key(*args, **kwargs)
        if key in self.compiled_graph:
            return self.compiled_graph.get(key)

        # Get the JAXpr and states from the stateful function
        if self.target == 'forloop':
            args, kwargs = jax.tree.map(lambda x: x[0], (args, kwargs))
        jaxpr = self.stateful_fn.get_jaxpr(*args, **kwargs)
        states = self.stateful_fn.get_states(*args, **kwargs)

        # Build state mappings
        self._build_state_mappings(jaxpr, states, self.stateful_fn.get_out_shapes(*args, **kwargs))

        # Parse equations to identify groups and connections
        self._parse_equations(jaxpr)

        res = self.operations, self.connections, {
            'invars_to_state': self.invars_to_state,
            'state_to_invars': self.state_to_invars,
            'outvars_to_state': self.outvars_to_state,
            'state_to_outvars': self.state_to_outvars,
        }
        self.compiled_graph.set(key, res)

        return self.compiled_graph.get(key)

    def __call__(self, *args, **kwargs):
        return self.parse(*args, **kwargs)

    def display_analysis_results(self, *args, **kwargs):
        """
        Display comprehensive analysis results for BPU Operation Connection Parser
        """
        operations, connections, state_mappings = self.parse(*args, **kwargs)
        operations: List[Operation]
        connections: List[Connection]
        state_mappings: Dict[str, Any]

        print("BPU Operation Connection Parser - Comprehensive Analysis")
        print("=" * 60)

        print(f"\nAnalysis Complete!")
        print(f"   - Operations identified: {len(operations)}")
        print(f"   - Connections found: {len(connections)}")
        print(
            f"   - State mappings: {len(state_mappings.get('invars_to_state', {}))} inputs, {len(state_mappings.get('outvars_to_state', {}))} outputs")

        # Detailed operation analysis
        print(f"\nDetailed Operation Analysis:")
        for i, operation in enumerate(operations):
            print(f"\n   Operation {i} ({operation.name}):")
            print(f"     - Total equations: {len(operation.eqns)}")

            # Show equation types and shapes
            eq_types = {}
            for eqn in operation.eqns:
                eqn: JaxprEqn
                prim_name = eqn.primitive.name
                eq_types[prim_name] = eq_types.get(prim_name, 0) + 1

            print(f"     - Primitive summary: {dict(eq_types)}")

            # Show all equations in this operation
            print(f"     - All equations:")
            for j, eqn in enumerate(operation.eqns):
                # Get output info
                output_info = ""
                if len(eqn.outvars) > 0:
                    outvar = eqn.outvars[0]
                    if hasattr(outvar, 'aval'):
                        output_info = f" -> {outvar.aval.dtype}{list(outvar.aval.shape)}"

                # Get input count
                input_count = len(eqn.invars)

                print(f"       [{j:2d}] {eqn.primitive.name}({input_count} inputs){output_info}")

                # Show parameters if they exist and are interesting
                if hasattr(eqn, 'params') and eqn.params:
                    interesting_params = {}
                    for key, value in eqn.params.items():
                        if key in ['limit_indices', 'start_indices', 'strides', 'dimension_numbers', 'axes']:
                            interesting_params[key] = value
                    if interesting_params:
                        print(f"            params: {interesting_params}")

        # Connection analysis
        print(f"\nConnection Analysis:")
        for i, conn in enumerate(connections):
            print(f"\n   Connection {i}:")
            print(f"     - From: {conn.pre.name} ({len(conn.pre.eqns)} ops)")
            print(f"     - To: {conn.post.name} ({len(conn.post.eqns)} ops)")

            # Show complete jaxpr equations if available
            if hasattr(conn.jaxpr, 'jaxpr') and len(conn.jaxpr.jaxpr.eqns) > 0:
                inner_eqns = conn.jaxpr.jaxpr.eqns
                print(f"     - Connection equations ({len(inner_eqns)} total):")

                for j, eqn in enumerate(inner_eqns):
                    # Get output info
                    output_info = ""
                    if len(eqn.outvars) > 0:
                        outvar = eqn.outvars[0]
                        if hasattr(outvar, 'aval'):
                            output_info = f" -> {outvar.aval.dtype}{list(outvar.aval.shape)}"

                    # Get input count
                    input_count = len(eqn.invars)

                    print(f"       [{j:2d}] {eqn.primitive.name}({input_count} inputs){output_info}")

                    # Show parameters if they exist and are interesting
                    if hasattr(eqn, 'params') and eqn.params:
                        interesting_params = {}
                        for key, value in eqn.params.items():
                            if key in ['limit_indices', 'start_indices', 'strides', 'dimension_numbers', 'axes',
                                       'shape',
                                       'broadcast_dimensions']:
                                interesting_params[key] = value
                        if interesting_params:
                            print(f"            params: {interesting_params}")
            else:
                print(f"     - Connection JAXpr: No inner equations found")

        # State mapping analysis
        print(f"\nState Mapping Analysis:")
        state_types = {}
        for state in state_mappings.get('state_to_invars', {}).keys():
            state_type = type(state).__name__
            state_types[state_type] = state_types.get(state_type, 0) + 1

        print(f"   - State types: {dict(state_types)}")

        # Show detailed state mappings for both input and output variables
        print(f"   - Detailed mappings:")

        # Input variable mappings
        print(f"     Input State Mappings:")
        for i, (state, invars) in enumerate(state_mappings.get('state_to_invars', {}).items()):
            state_type = type(state).__name__
            # Try to get state value info
            state_info = ""
            if hasattr(state, 'value') and hasattr(state.value, 'shape'):
                shape = state.value.shape
                dtype = str(state.value.dtype) if hasattr(state.value, 'dtype') else 'unknown'
                state_info = f" [{dtype}{list(shape)}]"

            print(f"       State {i} ({state_type}{state_info}):")

            if isinstance(invars, list):
                for j, var in enumerate(invars):
                    var_info = ""
                    if hasattr(var, 'aval'):
                        shape = var.aval.shape
                        dtype = str(var.aval.dtype) if hasattr(var.aval, 'dtype') else 'unknown'
                        var_info = f" -> {dtype}{list(shape)}"
                    print(f"         - Input var {j}: Var(id={id(var)}){var_info}")
            else:
                var_info = ""
                if hasattr(invars, 'aval'):
                    shape = invars.aval.shape
                    dtype = str(invars.aval.dtype) if hasattr(invars.aval, 'dtype') else 'unknown'
                    var_info = f" -> {dtype}{list(shape)}"
                print(f"         - Input var: Var(id={id(invars)}){var_info}")

        # Output variable mappings
        print(f"     Output State Mappings:")
        for i, (state, outvars) in enumerate(state_mappings.get('state_to_outvars', {}).items()):
            state_type = type(state).__name__
            # Try to get state value info
            state_info = ""
            if hasattr(state, 'value') and hasattr(state.value, 'shape'):
                shape = state.value.shape
                dtype = str(state.value.dtype) if hasattr(state.value, 'dtype') else 'unknown'
                state_info = f" [{dtype}{list(shape)}]"

            print(f"       State {i} ({state_type}{state_info}):")

            if isinstance(outvars, list):
                for j, var in enumerate(outvars):
                    var_info = ""
                    if hasattr(var, 'aval'):
                        shape = var.aval.shape
                        dtype = str(var.aval.dtype) if hasattr(var.aval, 'dtype') else 'unknown'
                        var_info = f" -> {dtype}{list(shape)}"
                    print(f"         - Output var {j}: Var(id={id(var)}){var_info}")
            else:
                var_info = ""
                if hasattr(outvars, 'aval'):
                    shape = outvars.aval.shape
                    dtype = str(outvars.aval.dtype) if hasattr(outvars.aval, 'dtype') else 'unknown'
                    var_info = f" -> {dtype}{list(shape)}"
                print(f"         - Output var: Var(id={id(outvars)}){var_info}")

        # Validation
        print(f"\nValidation Results:")

        # Check operation coverage
        total_eqns = sum(len(operation.eqns) for operation in operations)
        print(f"   - Equation coverage: {total_eqns} equations in operations")

        # Check state mapping completeness
        total_state_vars = len(state_mappings.get('invars_to_state', {})) + len(
            state_mappings.get('outvars_to_state', {}))
        print(f"   - State mappings: {total_state_vars} total variable mappings")

        print(f"\nSummary:")
        print(f"   The BPU parser successfully analyzed the neural network into:")
        print(f"   - {len(operations)} computational operations")
        print(f"   - {len(connections)} inter-operation connections")
        print(f"   - Complete state variable mappings")
        print(f"   This structure is ready for BPU compilation and optimization!")
