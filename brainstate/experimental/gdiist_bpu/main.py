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


import json
from typing import Callable, Dict, List, Any, Tuple, Optional

import jax
from jax.api_util import shaped_abstractify

from brainstate._compatible_import import JaxprEqn
from brainstate.transform._make_jaxpr import StatefulFunction, _make_hashable
from brainstate.util._cache import BoundedCache
from .component import Node, Connection
from .parser import Parser

__all__ = [
    'GdiistBPUParser',
]


class GdiistBPUParser:
    """
    Parser for BPU (second generation) operations and connections.

    This class is responsible for parsing the operations and connections in a BPU model.
    It provides comprehensive analysis capabilities including:

    - Operation and connection parsing
    - Statistics and metrics computation
    - Multiple display formats (text, summary, graph)
    - Export capabilities (dict, JSON)
    - Cache management

    """

    def __init__(
        self,
        fn: Callable,
        target: str = 'jit',
        cache_size: int = 128,
    ):
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
            _text_display(nodes, connections, state_mapping)
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
            f"GdiistBPUParser(target='{self.target}', "
            f"cache_size={self.compiled_graph.maxsize}, "
            f"cached_graphs={len(self.compiled_graph)})"
        )


def _text_display(
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
    print('----------------------------------------')
    for i, operation in enumerate(operations):
        print(f"\nNode {i}:")
        formater = _no_formatter(len(operation.eqns))
        for j, eqn in enumerate(operation.eqns):
            _text_one_eqn(eqn, formater.format(j))

    # Connection analysis
    print(f"\nConnection Analysis:")
    print('----------------------------------------')
    for i, conn in enumerate(connections):
        print(f"\nConnection {i}:")
        print(f"     - From: {conn.pre.name} ({len(conn.pre.eqns)} ops)")
        print(f"     - To: {conn.post.name} ({len(conn.post.eqns)} ops)")

        # Show complete jaxpr equations if available
        inner_eqns = conn.jaxpr.eqns
        print(f"     - Connection equations ({len(inner_eqns)} total):")
        formater = _no_formatter(len(inner_eqns))
        for j, eqn in enumerate(inner_eqns):
            _text_one_eqn(eqn, formater.format(j))


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
    print(f"     [{no}] {eqn.primitive.name}({input_count} inputs){output_info}")

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
            print(f"         params: {interesting_params}")


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
