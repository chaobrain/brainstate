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
from .data import Group, Connection
import jax


class BpuGroupConnectionParser:
    """
    Parser for BPU group and connections.
    This class is responsible for parsing the connections between groups in a BPU model.
    """

    def __init__(
        self,
        module_instance,
    ):
        self.module_instance = module_instance
        self.groups = []
        self.connections = []
        self.invars_to_state = dict()
        self.state_to_invars = dict()
        self.outvars_to_state = dict()
        self.state_to_outvars = dict()
        
        # Track current grouping state
        self.current_group_eqns = []
        self.current_group_name = None
        self.group_counter = 0

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

    def _is_connection_primitive(self, primitive_name):
        """Check if primitive represents a connection operation that should end current group"""
        connection_primitives = {
            'dot_general',
            *get_all_primitive_names()
        }
        return primitive_name in connection_primitives

    def _can_group_together(self, eqn1, eqn2):
        """Check if two equations can be grouped together based on shape and dimension consistency"""
        # Some operations should always be grouped with adjacent operations
        always_group_ops = {
            'broadcast_in_dim', 'stop_gradient', 'convert_element_type', 
            'squeeze', 'expand_dims', 'reshape', 'slice'
        }
        
        # If either operation is in always_group_ops, they can be grouped
        if (eqn1.primitive.name in always_group_ops or 
            eqn2.primitive.name in always_group_ops):
            return True
        
        # Check shape and dimension consistency for other operations
        if hasattr(eqn1, 'outvars') and hasattr(eqn2, 'outvars'):
            if len(eqn1.outvars) > 0 and len(eqn2.outvars) > 0:
                out1 = eqn1.outvars[0]
                out2 = eqn2.outvars[0]
                
                # Check if shapes are consistent
                if hasattr(out1, 'aval') and hasattr(out2, 'aval'):
                    if out1.aval.shape == out2.aval.shape and out1.aval.ndim == out2.aval.ndim:
                        return True
        return False

    def _finalize_current_group(self):
        """Finalize the current group and add it to groups list"""
        if self.current_group_eqns:
            group_name = f"group_{self.group_counter}"
            group = Group(name=group_name, eqns=self.current_group_eqns.copy())
            self.groups.append(group)
            self.group_counter += 1
            self.current_group_eqns.clear()
            return group
        return None

    def _expand_nested_jaxpr(self, eqn):
        """Expand nested JAXpr (like jit) by replacing them with their inner equations"""
        expanded_eqns = []
        
        if eqn.primitive.name == 'jit':
            # Get nested jaxpr
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

    def _parse_equations(self, closed_jaxpr):
        """Parse JAXpr equations to identify groups and connections"""
        # Extract the actual jaxpr from ClosedJaxpr
        jaxpr = closed_jaxpr.jaxpr
        expanded_eqns = []
        
        # First expand all nested JAXpr
        for eqn in jaxpr.eqns:
            expanded_eqns.extend(self._expand_nested_jaxpr(eqn))
        
        # Process expanded equations for grouping
        i = 0
        while i < len(expanded_eqns):
            eqn = expanded_eqns[i]
            primitive_name = eqn.primitive.name
            
            # Check if this is a connection operation
            if self._is_connection_primitive(primitive_name):
                # Finalize current group as pre_group
                pre_group = self._finalize_current_group()
                
                # The connection eqn itself is not added to any group
                connection_eqn = eqn
                
                # Move to next equation
                i += 1
                
                # Start collecting equations for post_group right away
                # The post_group will be finalized when we hit the next connection
                # or when we reach the end
                
                # Create connection if we have a pre_group
                # We'll set post_group to None for now and update it later
                if pre_group is not None:
                    # We'll create a temporary connection and update post_group later
                    # For now, just continue with normal grouping
                    pass
                
                # Continue normal processing - the next non-connection equations
                # will start forming a new group
                continue
            else:
                # Non-connection operation, add to current group
                if (not self.current_group_eqns or 
                    len(self.current_group_eqns) == 0 or 
                    self._can_group_together(self.current_group_eqns[-1], eqn)):
                    self.current_group_eqns.append(eqn)
                else:
                    # Start new group
                    self._finalize_current_group()
                    self.current_group_eqns.append(eqn)
                i += 1
        
        # Finalize the last group
        self._finalize_current_group()
        
        # Now we need to create connections by scanning for connection operations
        # and linking adjacent groups
        self._create_connections_from_groups(expanded_eqns)
    
    def _create_connections_from_groups(self, expanded_eqns):
        """Create connections by analyzing connection operations between groups"""
        # Find all connection operations and their positions
        connection_positions = []
        for i, eqn in enumerate(expanded_eqns):
            if self._is_connection_primitive(eqn.primitive.name):
                connection_positions.append((i, eqn))
        
        # For each connection operation, find the pre and post groups
        for pos, conn_eqn in connection_positions:
            pre_group = None
            post_group = None
            pre_group_max_pos = -1
            post_group_min_pos = float('inf')
            
            # Find pre_group: look for the group with equations closest before this position
            for group in self.groups:
                group_max_pos = -1
                for group_eqn in group.eqns:
                    try:
                        group_eqn_pos = expanded_eqns.index(group_eqn)
                        if group_eqn_pos < pos:
                            group_max_pos = max(group_max_pos, group_eqn_pos)
                    except ValueError:
                        continue
                
                # Update pre_group if this group is closer to the connection
                if group_max_pos > pre_group_max_pos:
                    pre_group_max_pos = group_max_pos
                    pre_group = group
            
            # Find post_group: look for the group with equations closest after this position
            for group in self.groups:
                group_min_pos = float('inf')
                for group_eqn in group.eqns:
                    try:
                        group_eqn_pos = expanded_eqns.index(group_eqn)
                        if group_eqn_pos > pos:
                            group_min_pos = min(group_min_pos, group_eqn_pos)
                    except ValueError:
                        continue
                
                # Update post_group if this group is closer to the connection
                if group_min_pos < post_group_min_pos:
                    post_group_min_pos = group_min_pos
                    post_group = group
            
            # Create connection if we have both groups
            if pre_group is not None and post_group is not None:
                # Avoid duplicate connections
                existing_connection = None
                for existing_conn in self.connections:
                    if (existing_conn.pre == pre_group and 
                        existing_conn.post == post_group and
                        existing_conn.eqn.primitive.name == conn_eqn.primitive.name):
                        existing_connection = existing_conn
                        break
                
                if existing_connection is None:
                    connection = Connection(pre=pre_group, post=post_group, eqn=conn_eqn)
                    self.connections.append(connection)

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
        
        return self.groups, self.connections, {
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
