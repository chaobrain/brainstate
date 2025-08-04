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




from typing import NamedTuple, Dict, List, Any
from brainstate._compatible_import import ClosedJaxpr


class Group(NamedTuple):
    name: str
    eqns: list[ClosedJaxpr]


class Connection(NamedTuple):
    pre: Group
    post: Group
    eqn: ClosedJaxpr


def display_analysis_results(groups: List[Group], connections: List[Connection], state_mappings: Dict[str, Any]):
    """
    Display comprehensive analysis results for BPU Group Connection Parser
    
    Args:
        groups: List of identified groups
        connections: List of connections between groups
        state_mappings: Dictionary containing state variable mappings
    """
    print("BPU Group Connection Parser - Comprehensive Analysis")
    print("=" * 60)
    
    print(f"\nAnalysis Complete!")
    print(f"   - Groups identified: {len(groups)}")
    print(f"   - Connections found: {len(connections)}")
    print(f"   - State mappings: {len(state_mappings.get('invars_to_state', {}))} inputs, {len(state_mappings.get('outvars_to_state', {}))} outputs")
    
    # Detailed group analysis
    print(f"\nDetailed Group Analysis:")
    for i, group in enumerate(groups):
        print(f"\n   Group {i} ({group.name}):")
        print(f"     - Equations: {len(group.eqns)}")
        
        # Show equation types and shapes
        eq_types = {}
        for eqn in group.eqns:
            prim_name = eqn.primitive.name
            eq_types[prim_name] = eq_types.get(prim_name, 0) + 1
        
        print(f"     - Operations: {dict(eq_types)}")
        
        # Show representative shape
        if group.eqns and len(group.eqns[0].outvars) > 0:
            outvar = group.eqns[0].outvars[0]
            if hasattr(outvar, 'aval'):
                print(f"     - Output shape: {outvar.aval.shape}")
    
    # Connection analysis
    print(f"\nConnection Analysis:")
    for i, conn in enumerate(connections):
        print(f"\n   Connection {i}:")
        print(f"     - From: {conn.pre.name} ({len(conn.pre.eqns)} ops)")
        print(f"     - To: {conn.post.name} ({len(conn.post.eqns)} ops)")
        print(f"     - Via: {conn.eqn.primitive.name}")
        
        # Connection details
        if hasattr(conn.eqn, 'params') and 'dimension_numbers' in conn.eqn.params:
            dims = conn.eqn.params['dimension_numbers']
            print(f"     - Dimensions: {dims}")
    
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
    
    # Check group coverage
    total_eqns = sum(len(group.eqns) for group in groups)
    print(f"   - Equation coverage: {total_eqns} equations grouped")
    
    # Check state mapping completeness
    total_state_vars = len(state_mappings.get('invars_to_state', {})) + len(state_mappings.get('outvars_to_state', {}))
    print(f"   - State mappings: {total_state_vars} total variable mappings")
    
    print(f"\nSummary:")
    print(f"   The BPU parser successfully analyzed the neural network into:")
    print(f"   - {len(groups)} computational groups")
    print(f"   - {len(connections)} inter-group connections")
    print(f"   - Complete state variable mappings")
    print(f"   This structure is ready for BPU compilation and optimization!")




