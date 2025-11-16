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

from typing import Callable, NamedTuple, Dict, List, Any, Tuple, Optional
import json
import warnings

import jax
from jax.api_util import shaped_abstractify

from brainstate._compatible_import import is_jit_primitive, ClosedJaxpr, JaxprEqn
from brainstate.transform._make_jaxpr import StatefulFunction, _make_hashable
from brainstate.util._cache import BoundedCache
from .utils import _is_brainevent_jit_connection

__all__ = [
    'GdiistBpuParser',
]


class Node(NamedTuple):
    name: str
    eqns: ClosedJaxpr



class Connection(NamedTuple):
    pre: Node
    post: Node
    jaxpr: ClosedJaxpr


class Output(NamedTuple):
    pass




class Parser:
    def __init__(self, stateful_fn: StatefulFunction, inputs: Tuple):
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
        if _is_brainevent_jit_connection(eqn):
            return True

        # All other operations can be merged into the current operation
        return False

    def _finalize_current_operation(self):
        """Finalize the current operation and add it to operations list"""
        if self.current_operation_eqns:
            operation_name = f"operation_{self.operation_counter}"
            operation = Node(name=operation_name, eqns=self.current_operation_eqns.copy())
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


class GdiistBpuParser:
    """
    Parser for BPU operations and connections.

    This class is responsible for parsing the operations and connections in a BPU model.
    It provides comprehensive analysis capabilities including:
    - Operation and connection parsing
    - Statistics and metrics computation
    - Multiple display formats (text, summary, graph)
    - Export capabilities (dict, JSON)
    - Cache management
    """

    def __init__(self, fn: Callable, target: str = 'jit', cache_size: int = 128):
        """
        Initialize the GdiistBpuParser.

        Args:
            fn: The function to parse
            target: Target mode, either 'jit' or 'forloop'
            cache_size: Maximum size of the cache for parsed results
        """
        self.fn = fn
        self.stateful_fn = StatefulFunction(self.fn, return_only_write=False, ir_optimizations='all')
        self.target = target
        if target not in ['jit', 'forloop']:
            raise ValueError(f"Target must be either 'jit' or 'forloop', got {target}")
        self.compiled_graph = BoundedCache(maxsize=cache_size)
        self._last_parse_result = None
        self._last_cache_key = None

    def cache_key(self, *args, **kwargs) -> Any:
        """
        Generate a hashable cache key from the input arguments.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            A hashable key for caching
        """
        if self.target == 'forloop':
            args, kwargs = jax.tree.map(lambda x: x[0], (args, kwargs))
        return _make_hashable(jax.tree.map(shaped_abstractify, (args, kwargs)))

    def parse(
        self,
        *args,
        display: Optional[str] = None,
        verbose: bool = False,
        **kwargs
    ) -> Tuple[List[Node], List[Connection], Dict[str, Any]]:
        """
        Main parsing function that analyzes JAXpr and builds groups and connections.

        Args:
            *args: Positional arguments for the function
            display: Display mode ('text', 'summary', 'graph', or None)
            verbose: If True, show additional parsing information
            **kwargs: Keyword arguments for the function

        Returns:
            Tuple of (nodes, connections, state_mapping)
        """
        key = self.cache_key(*args, **kwargs)
        self._last_cache_key = key

        if key in self.compiled_graph:
            if verbose:
                print(f"Cache hit for key: {key}")
            result = self.compiled_graph.get(key)
        else:
            if verbose:
                print(f"Cache miss for key: {key}, parsing...")

            # Get the JAXpr and states from the stateful function
            parse_args, parse_kwargs = args, kwargs
            if self.target == 'forloop':
                parse_args, parse_kwargs = jax.tree.map(lambda x: x[0], (args, kwargs))

            # IR parsing
            nodes, connections, state_mapping = Parser(self.stateful_fn, (parse_args, parse_kwargs)).parse()
            self.compiled_graph.set(key, (nodes, connections, state_mapping))
            result = (nodes, connections, state_mapping)

        self._last_parse_result = result

        # Handle display options
        if display is not None:
            self.display(result, mode=display)

        return result

    def display(
        self,
        result: Optional[Tuple] = None,
        mode: str = 'text'
    ) -> None:
        """
        Display the parsed results in various formats.

        Args:
            result: Parse result tuple, uses last parse result if None
            mode: Display mode ('text', 'summary', 'graph')
        """
        if result is None:
            if self._last_parse_result is None:
                raise ValueError("No parse result available. Call parse() first.")
            result = self._last_parse_result

        nodes, connections, state_mapping = result

        if mode == 'text':
            text_display(nodes, connections, state_mapping)
        else:
            raise ValueError(f"Unknown display mode: {mode}. Choose from 'text', 'summary', 'graph'")

    def get_statistics(
        self,
        result: Optional[Tuple] = None
    ) -> Dict[str, Any]:
        """
        Compute statistics about the parsed graph.

        Args:
            result: Parse result tuple, uses last parse result if None

        Returns:
            Dictionary containing various statistics
        """
        if result is None:
            if self._last_parse_result is None:
                raise ValueError("No parse result available. Call parse() first.")
            result = self._last_parse_result

        nodes, connections, state_mapping = result

        # Basic counts
        num_nodes = len(nodes)
        num_connections = len(connections)
        total_eqns = sum(len(node.eqns) for node in nodes)

        # Operation type distribution
        op_types = {}
        for node in nodes:
            for eqn in node.eqns:
                op_name = eqn.primitive.name
                op_types[op_name] = op_types.get(op_name, 0) + 1

        # Connection statistics
        connection_sources = {}
        connection_targets = {}
        for conn in connections:
            connection_sources[conn.pre.name] = connection_sources.get(conn.pre.name, 0) + 1
            connection_targets[conn.post.name] = connection_targets.get(conn.post.name, 0) + 1

        # Node complexity (number of equations per node)
        node_complexities = {node.name: len(node.eqns) for node in nodes}
        avg_complexity = total_eqns / num_nodes if num_nodes > 0 else 0

        # State mapping statistics
        num_states = len(state_mapping.get('invars_to_state', {}))

        return {
            'num_nodes': num_nodes,
            'num_connections': num_connections,
            'total_equations': total_eqns,
            'average_node_complexity': avg_complexity,
            'operation_types': op_types,
            'connection_sources': connection_sources,
            'connection_targets': connection_targets,
            'node_complexities': node_complexities,
            'num_states': num_states,
        }

    def to_dict(
        self,
        result: Optional[Tuple] = None,
        include_details: bool = True
    ) -> Dict[str, Any]:
        """
        Convert the parsed result to a dictionary format.

        Args:
            result: Parse result tuple, uses last parse result if None
            include_details: If True, include detailed equation information

        Returns:
            Dictionary representation of the parse result
        """
        if result is None:
            if self._last_parse_result is None:
                raise ValueError("No parse result available. Call parse() first.")
            result = self._last_parse_result

        nodes, connections, state_mapping = result

        # Convert nodes
        nodes_dict = []
        for node in nodes:
            node_info = {
                'name': node.name,
                'num_equations': len(node.eqns),
            }
            if include_details:
                eqns_info = []
                for eqn in node.eqns:
                    eqn_info = {
                        'primitive': eqn.primitive.name,
                        'num_inputs': len(eqn.invars),
                        'num_outputs': len(eqn.outvars),
                    }
                    # Add output shapes
                    if len(eqn.outvars) > 0:
                        output_shapes = []
                        for outvar in eqn.outvars:
                            if hasattr(outvar, 'aval'):
                                output_shapes.append({
                                    'dtype': str(outvar.aval.dtype),
                                    'shape': list(outvar.aval.shape)
                                })
                        eqn_info['outputs'] = output_shapes
                    eqns_info.append(eqn_info)
                node_info['equations'] = eqns_info
            nodes_dict.append(node_info)

        # Convert connections
        connections_dict = []
        for conn in connections:
            conn_info = {
                'source': conn.pre.name,
                'target': conn.post.name,
            }
            if include_details and hasattr(conn.jaxpr, 'jaxpr'):
                conn_info['num_inner_equations'] = len(conn.jaxpr.jaxpr.eqns)
            connections_dict.append(conn_info)

        return {
            'nodes': nodes_dict,
            'connections': connections_dict,
            'statistics': self.get_statistics(result),
        }

    def to_json(
        self,
        result: Optional[Tuple] = None,
        include_details: bool = True,
        indent: int = 2
    ) -> str:
        """
        Convert the parsed result to JSON format.

        Args:
            result: Parse result tuple, uses last parse result if None
            include_details: If True, include detailed equation information
            indent: JSON indentation level

        Returns:
            JSON string representation
        """
        data = self.to_dict(result, include_details)
        return json.dumps(data, indent=indent, default=str)

    def clear_cache(self) -> None:
        """Clear the entire cache."""
        self.compiled_graph = BoundedCache(maxsize=self.compiled_graph.maxsize)
        self._last_parse_result = None
        self._last_cache_key = None

    def cache_info(self) -> Dict[str, Any]:
        """
        Get information about the cache status.

        Returns:
            Dictionary with cache statistics
        """
        return {
            'size': len(self.compiled_graph),
            'maxsize': self.compiled_graph.maxsize,
            'last_key': self._last_cache_key,
        }

    def validate(self, result: Optional[Tuple] = None) -> List[str]:
        """
        Validate the parsed result for potential issues.

        Args:
            result: Parse result tuple, uses last parse result if None

        Returns:
            List of warning messages (empty if no issues found)
        """
        if result is None:
            if self._last_parse_result is None:
                raise ValueError("No parse result available. Call parse() first.")
            result = self._last_parse_result

        nodes, connections, state_mapping = result
        warnings_list = []

        # Check for empty nodes
        for node in nodes:
            if len(node.eqns) == 0:
                warnings_list.append(f"Node '{node.name}' has no equations")

        # Check for isolated nodes (no connections)
        connected_nodes = set()
        for conn in connections:
            connected_nodes.add(conn.pre.name)
            connected_nodes.add(conn.post.name)

        for node in nodes:
            if node.name not in connected_nodes and len(nodes) > 1:
                warnings_list.append(f"Node '{node.name}' is isolated (no connections)")

        # Check for self-loops
        for conn in connections:
            if conn.pre.name == conn.post.name:
                warnings_list.append(f"Self-loop detected in node '{conn.pre.name}'")

        # Check for duplicate connections
        conn_pairs = [(conn.pre.name, conn.post.name) for conn in connections]
        if len(conn_pairs) != len(set(conn_pairs)):
            warnings_list.append("Duplicate connections detected")

        return warnings_list

    def __call__(self, *args, **kwargs):
        """Alias for parse() method."""
        return self.parse(*args, **kwargs)

    def __repr__(self) -> str:
        """String representation of the parser."""
        return (
            f"GdiistBpuParser(target='{self.target}', "
            f"cache_size={self.compiled_graph.maxsize}, "
            f"cached_graphs={len(self.compiled_graph)})"
        )


def text_display(
    operations: List[Node],
    connections: List[Connection],
    state_mappings: Dict[str, Any]
):
    """
    Display comprehensive analysis results for BPU Node Connection Parser
    """

    print(f"\nSummary:")
    print(f"   The BPU parser successfully analyzed the neural network into:")
    print(f"   - {len(operations)} computational operations")
    print(f"   - {len(connections)} inter-operation connections")

    # Detailed operation analysis
    print(f"\nNode Analysis:")
    for i, operation in enumerate(operations):
        print(f"\n  Node {i}:")
        formater = _no_formatter(len(operation.eqns))
        for j, eqn in enumerate(operation.eqns):
            _text_one_eqn(eqn, formater.format(j))

    # Connection analysis
    print(f"\nConnection Analysis:")
    for i, conn in enumerate(connections):
        print(f"\n  Connection {i}:")
        print(f"     - From: {conn.pre.name} ({len(conn.pre.eqns)} ops)")
        print(f"     - To: {conn.post.name} ({len(conn.post.eqns)} ops)")

        # Show complete jaxpr equations if available
        if hasattr(conn.jaxpr, 'jaxpr') and len(conn.jaxpr.jaxpr.eqns) > 0:
            inner_eqns = conn.jaxpr.jaxpr.eqns
            print(f"     - Connection equations ({len(inner_eqns)} total):")
            formater = _no_formatter(len(inner_eqns))
            for j, eqn in enumerate(inner_eqns):
                _text_one_eqn(eqn, formater.format(j))

        else:
            print(f"     - Connection JAXpr: No inner equations found")


def _text_one_eqn(eqn: JaxprEqn, no):
    # Get output info
    output_info = ""
    if len(eqn.outvars) > 0:
        if len(eqn.outvars) == 1:
            output_info = f" -> {eqn.outvars[0].aval.dtype}{list(eqn.outvars[0].aval.shape)}"
        else:
            outvar_infos = []
            for outvar in eqn.outvars:
                outvar_infos.append(f"{outvar.aval.dtype}{list(outvar.aval.shape)}")
            output_info = " -> [" + ", ".join(outvar_infos) + "]"
    # Get input count
    input_count = len(eqn.invars)
    print(f"       [{no}] {eqn.primitive.name}({input_count} inputs){output_info}")

    # Show parameters if they exist and are interesting
    if eqn.params:
        interesting_params = {}
        for key, value in eqn.params.items():
            if key in [
                'limit_indices',
                'start_indices',
                'strides',
                'dimension_numbers',
                'axes',

                'limit_indices',
                'start_indices',
                'strides',
                'dimension_numbers',
                'axes',
                'shape',
                'broadcast_dimensions'
            ]:
                interesting_params[key] = value
        if interesting_params:
            print(f"           params: {interesting_params}")


def _no_formatter(num):
    if num < 10:
        formater = '{:1d}'
    elif num < 100:
        formater = '{:2d}'
    elif num < 1000:
        formater = '{:3d}'
    else:
        formater = '{:4d}'
    return formater
