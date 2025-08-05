# Copyright 2024 BDP Ecosystem Limited. All Rights Reserved.
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
from brainevent import get_all_primitive_names

import brainstate.compile
from .data import Operation, Connection
import jax


class BpuOperationConnectionParser:
    """
    Parser for BPU operations and connections.
    This class is responsible for parsing the operations and connections in a BPU model.
    """

    def __init__(
        self,
        module_instance,
    ):
        self.module_instance = module_instance
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

    def _build_state_mappings(self, closed_jaxpr, states, cache_key, stateful_module):
        """Build mappings between state variables and JAXpr input/output variables
        
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
        out_shapes = stateful_module.get_out_shapes(cache_key)[0]
        
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
        state_tree_invars = jax.tree.unflatten(
            state_tree, 
            jaxpr.invars[num_inputs_before_states:]
        )
        
        # Output variables: after the main outputs, the rest correspond to state updates
        state_tree_outvars = jax.tree.unflatten(
            state_tree,
            jaxpr.outvars[num_out:]
        )
        
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

    def _is_brainevent_jit_connection(self, eqn):
        """Check if equation is a jit-wrapped brainevent operation that should be a connection"""
        if eqn.primitive.name == 'jit':
            # Check if the function name starts with 'brainevent'
            if 'name' in eqn.params:
                name = eqn.params['name']
                if isinstance(name, str) and name.startswith('brainevent'):
                    return True
        return False

    def _should_split_operation(self, eqn):
        """Check if equation should cause a split in current operation"""
        # slice operations should cause a split and become their own operation
        if eqn.primitive.name == 'slice':
            return True
        
        # brainevent jit connections should cause a split (but don't become operations themselves)
        if self._is_brainevent_jit_connection(eqn):
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
        
        if eqn.primitive.name == 'jit':
            # Check if this is a brainevent connection - if so, keep it as-is
            if self._is_brainevent_jit_connection(eqn):
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
                if self._is_brainevent_jit_connection(eqn):
                    continue
                # If it's a slice, add it as a new operation
                elif eqn.primitive.name == 'slice':
                    self.current_operation_eqns.append(eqn)
            else:
                # Regular equation - add to current operation
                self.current_operation_eqns.append(eqn)
        
        # Finalize the last operation
        self._finalize_current_operation()
        
        # Second pass: create connections between adjacent operations
        for i, eqn in enumerate(expanded_eqns):
            if self._is_brainevent_jit_connection(eqn):
                # Find the pre and post operations around this connection
                pre_operation = None
                post_operation = None
                
                # Find pre_operation: the operation containing equations before this position
                for j in range(i-1, -1, -1):
                    for operation in self.operations:
                        if expanded_eqns[j] in operation.eqns:
                            pre_operation = operation
                            break
                    if pre_operation:
                        break
                
                # Find post_operation: the operation containing equations after this position  
                for j in range(i+1, len(expanded_eqns)):
                    for operation in self.operations:
                        if expanded_eqns[j] in operation.eqns:
                            post_operation = operation
                            break
                    if post_operation:
                        break
                
                # Create connection if we have both operations
                if pre_operation is not None and post_operation is not None:
                    self._create_connection(pre_operation, post_operation, eqn)
    

    def parse(self, *args, **kwargs):
        """Main parsing function that analyzes JAXpr and builds groups and connections"""
        stateful_module = brainstate.compile.StatefulFunction(self.module_instance)
        stateful_module.make_jaxpr(*args, **kwargs)
        cache_key = stateful_module.get_arg_cache_key(*args, **kwargs)
        jaxpr = stateful_module.get_jaxpr(cache_key)
        states = stateful_module.get_states(cache_key)
        
        # Build state mappings
        self._build_state_mappings(jaxpr, states, cache_key, stateful_module)
        
        # Parse equations to identify groups and connections
        self._parse_equations(jaxpr)
        
        return self.operations, self.connections, {
            'invars_to_state': self.invars_to_state,
            'state_to_invars': self.state_to_invars,
            'outvars_to_state': self.outvars_to_state,
            'state_to_outvars': self.state_to_outvars,
        }

    def debug_raw_jaxpr(self, *args, **kwargs):
        """Debug function to print raw JAXpr for inspection"""
        stateful_module = brainstate.compile.StatefulFunction(self.module_instance)
        stateful_module.make_jaxpr(*args, **kwargs)
        cache_key = stateful_module.get_arg_cache_key(*args, **kwargs)
        jaxpr = stateful_module.get_jaxpr(cache_key)

        print("Raw JAXpr:")
        print(jaxpr)

        return jaxpr
