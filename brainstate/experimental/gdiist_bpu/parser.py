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

from typing import Dict, List, Set, Tuple, Any, Optional
from collections import defaultdict
import hashlib
from dataclasses import dataclass

from jax._src.core import Jaxpr, JaxprEqn, Var, Literal
from .data import Group, Connection


@dataclass
class JaxprAnalysisResult:
    """Simplified result structure for JAXpr analysis - only groups and connections."""
    groups: List[Group]
    connections: List[Connection]


class GroupTracker:
    """Tracks equations that should be grouped together based on similarity."""
    
    def __init__(self, eqn: JaxprEqn, group_id: str):
        self.group_id = group_id
        self.primitive_name = eqn.primitive.name
        # Filter only Var objects, exclude Literal objects
        self.input_vars: Set[Var] = {v for v in eqn.invars if isinstance(v, Var)}
        self.output_vars: Set[Var] = {v for v in eqn.outvars if isinstance(v, Var)}
        self.equations: List[JaxprEqn] = [eqn]
        self.parameters = dict(eqn.params) if hasattr(eqn, 'params') else {}
    
    def can_merge(self, eqn: JaxprEqn) -> bool:
        """Check if equation can be merged into this group."""
        # Same primitive type
        if eqn.primitive.name != self.primitive_name:
            return False
        
        # Similar parameters (allowing some flexibility)
        eqn_params = dict(eqn.params) if hasattr(eqn, 'params') else {}
        if not self._params_compatible(eqn_params):
            return False
        
        return True
    
    def _params_compatible(self, other_params: Dict[str, Any]) -> bool:
        """Check if parameters are compatible for grouping."""
        # For now, require exact parameter match
        # Could be extended to allow compatible parameters
        return self.parameters == other_params
    
    def add_equation(self, eqn: JaxprEqn):
        """Add equation to this group."""
        self.equations.append(eqn)
        # Filter only Var objects when updating
        self.input_vars.update(v for v in eqn.invars if isinstance(v, Var))
        self.output_vars.update(v for v in eqn.outvars if isinstance(v, Var))
    
    def to_group(self) -> Group:
        """Convert tracker to Group object."""
        return Group(
            id=self.group_id,
            input_vars=self.input_vars.copy(),
            output_vars=self.output_vars.copy(),
            equations=self.equations.copy(),
            parameters=self.parameters.copy(),
            primitive_name=self.primitive_name
        )


class JaxprGroupConnectionParser:
    """
    Parser that extracts Groups and Connections from JAX program representation (jaxpr).
    
    Based on the JaxprEvaluation design pattern:
    - Uses visitor pattern to traverse jaxpr
    - Tracks variables and their relationships
    - Groups similar equations together
    - Identifies connections between groups
    """
    
    def __init__(self, jaxpr: Jaxpr):
        self.jaxpr = jaxpr
        
        # Tracking state
        self.group_trackers: List[GroupTracker] = []
        self.var_to_producer_group: Dict[Var, str] = {}  # Maps variable to group that produces it
        self.var_to_consumer_groups: Dict[Var, Set[str]] = defaultdict(set)  # Maps variable to groups that consume it
        
        # Results
        self.groups: List[Group] = []
        self.connections: List[Connection] = []
    
    def parse(self) -> Tuple[List[Group], List[Connection]]:
        """
        Parse jaxpr into groups and connections.
        
        Returns:
            Tuple of (groups, connections)
        """
        # Reset state
        self._reset_state()
        
        # Parse equations into groups
        self._parse_equations()
        
        # Convert trackers to groups
        self._finalize_groups()
        
        # Extract connections between groups
        self._extract_connections()
        
        return self.groups, self.connections
    
    def _reset_state(self):
        """Reset parser state for fresh parsing."""
        self.group_trackers.clear()
        self.var_to_producer_group.clear()
        self.var_to_consumer_groups.clear()
        self.groups.clear()
        self.connections.clear()
    
    def _parse_equations(self):
        """Parse all equations in jaxpr into groups."""
        for eqn in self.jaxpr.eqns:
            self._process_equation(eqn)
    
    def _process_equation(self, eqn: JaxprEqn):
        """Process a single equation, either adding to existing group or creating new one."""
        # Try to find compatible existing group
        compatible_tracker = None
        for tracker in self.group_trackers:
            if tracker.can_merge(eqn):
                compatible_tracker = tracker
                break
        
        if compatible_tracker is not None:
            # Add to existing group
            compatible_tracker.add_equation(eqn)
            group_id = compatible_tracker.group_id
        else:
            # Create new group
            group_id = self._generate_group_id(eqn)
            new_tracker = GroupTracker(eqn, group_id)
            self.group_trackers.append(new_tracker)
        
        # Track variable relationships
        self._update_variable_tracking(eqn, group_id)
    
    def _generate_group_id(self, eqn: JaxprEqn) -> str:
        """Generate unique group ID based on equation characteristics."""
        # Create a hash based on primitive name and parameters
        content = f"{eqn.primitive.name}"
        if hasattr(eqn, 'params'):
            params_str = str(sorted(eqn.params.items()))
            content += params_str
        
        hash_obj = hashlib.md5(content.encode())
        base_id = f"{eqn.primitive.name}_{hash_obj.hexdigest()[:8]}"
        
        # Ensure uniqueness
        existing_ids = {tracker.group_id for tracker in self.group_trackers}
        counter = 0
        unique_id = base_id
        while unique_id in existing_ids:
            counter += 1
            unique_id = f"{base_id}_{counter}"
        
        return unique_id
    
    def _update_variable_tracking(self, eqn: JaxprEqn, group_id: str):
        """Update variable to group mappings."""
        # Track which group produces each output variable (outvars are always Var)
        for outvar in eqn.outvars:
            if isinstance(outvar, Var):
                self.var_to_producer_group[outvar] = group_id
        
        # Track which groups consume each input variable (filter only Var objects)
        for invar in eqn.invars:
            if isinstance(invar, Var):
                self.var_to_consumer_groups[invar].add(group_id)
    
    def _finalize_groups(self):
        """Convert all trackers to final Group objects."""
        self.groups = [tracker.to_group() for tracker in self.group_trackers]
    
    def _extract_connections(self):
        """Extract connections between groups based on variable dependencies."""
        # Create mapping from group_id to Group object for easy lookup
        id_to_group = {group.id: group for group in self.groups}
        
        # Find connections by examining variable flow
        connections = []  # Use list instead of set
        connection_pairs_seen = set()  # Track unique (pre_id, post_id) pairs
        
        # Create a copy of the items to avoid "dictionary changed size during iteration" error
        var_consumer_items = list(self.var_to_consumer_groups.items())
        
        for var, consumer_group_ids in var_consumer_items:
            producer_group_id = self.var_to_producer_group.get(var)
            
            if producer_group_id is None:
                # Variable is an input to the jaxpr, not produced by any group
                continue
            
            # Create connections from producer to each consumer
            for consumer_group_id in consumer_group_ids:
                if producer_group_id != consumer_group_id:
                    # Different groups - check if we've seen this connection pair
                    pair_key = (producer_group_id, consumer_group_id)
                    if pair_key in connection_pairs_seen:
                        continue
                    connection_pairs_seen.add(pair_key)
                    
                    # Different groups - create connection
                    pre_group = id_to_group[producer_group_id]
                    post_group = id_to_group[consumer_group_id]
                    
                    # Create connection between groups
                    connection = Connection(
                        pre=pre_group,
                        post=post_group
                    )
                    connections.append(connection)
        
        self.connections = connections


def parse_jaxpr_to_groups_and_connections(jaxpr: Jaxpr) -> Tuple[List[Group], List[Connection]]:
    """
    Convenience function to parse jaxpr into groups and connections.
    
    Args:
        jaxpr: JAX program representation to parse
        
    Returns:
        Tuple of (groups, connections)
    """
    parser = JaxprGroupConnectionParser(jaxpr)
    return parser.parse()


def analyze_jaxpr_group_connection(
    fn, 
    *args, 
    context_kwargs: Optional[dict] = None,
    **make_jaxpr_kwargs
) -> JaxprAnalysisResult:
    """
    High-level function to analyze any callable by generating JAXpr and parsing it.
    
    This function encapsulates the complete workflow:
    1. Generate JAXpr using brainstate.compile.make_jaxpr
    2. Parse JAXpr into Groups and Connections
    3. Return simplified analysis results
    
    Args:
        fn: Any callable function to analyze
        *args: Arguments to pass to the function for JAXpr generation
        context_kwargs: Environment context for BrainState functions (e.g., {'t': 1*u.ms})
        **make_jaxpr_kwargs: Additional arguments for make_jaxpr
        
    Returns:
        JaxprAnalysisResult containing groups and connections with computed properties
    """
    import brainstate as bst
    
    # Create wrapper function if context is needed
    if context_kwargs:
        def wrapped_fn(*fn_args):
            with bst.environ.context(**context_kwargs):
                return fn(*fn_args)
        target_fn = wrapped_fn
    else:
        target_fn = fn
    
    # Generate JAXpr
    try:
        jaxpr, states = bst.compile.make_jaxpr(target_fn, **make_jaxpr_kwargs)(*args)
    except Exception as e:
        raise RuntimeError(f"Failed to generate JAXpr: {e}") from e
    
    # Parse Groups and Connections
    try:
        groups, connections = parse_jaxpr_to_groups_and_connections(jaxpr.jaxpr)
    except Exception as e:
        raise RuntimeError(f"Failed to parse JAXpr: {e}") from e
    
    return JaxprAnalysisResult(groups=groups, connections=connections)


