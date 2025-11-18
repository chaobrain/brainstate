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


from typing import Callable, Dict, List, Any, Tuple, Optional

import jax
from jax.api_util import shaped_abstractify

from brainstate._compatible_import import JaxprEqn
from brainstate.experimental.graph_ir._data import Group, Connection
from brainstate.experimental.graph_ir._parser import parse, ParsedResults
from brainstate.transform._make_jaxpr import StatefulFunction, _make_hashable
from brainstate.util._cache import BoundedCache

__all__ = [
    'GdiistBPUParser',
]


class GdiistBPUParser:
    """Parser for BPU (second generation) operations and connections.
    
    This class is responsible for parsing the operations and connections in a BPU model.
    It provides comprehensive analysis capabilities including:
    
    - Operation and connection parsing
    - Statistics and metrics computation
    - Multiple display formats (text, summary, graph)
    - Export capabilities (dict, JSON)
    - Cache management

    Parameters
    ----------

    Returns
    -------

    """

    def __init__(
        self,
        fn: Callable,
        target: str = 'jit',
        cache_size: int = 128,
    ):
        self.fn = fn
        self.stateful_fn = StatefulFunction(self.fn, ir_optimizations='dce')
        # self.stateful_fn = StatefulFunction(self.fn)
        if target not in ['jit', 'forloop']:
            raise ValueError(f"Target must be either 'jit' or 'forloop', got {target}")
        self.target = target
        self.compiled_graph = BoundedCache(maxsize=cache_size)

    def cache_key(self, *args, **kwargs) -> Any:
        """Generate a hashable cache key from the input arguments.

        Parameters
        ----------
        *args :
            Positional arguments
        **kwargs :
            Keyword arguments

        Returns
        -------
        
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
    ) -> ParsedResults:
        """Main parsing function that analyzes JAXpr and builds groups and connections.

        Parameters
        ----------
        *args :
            Positional arguments for the function
        **kwargs :
            Keyword arguments for the function
        display :
            Display mode ('text', 'summary', 'graph', or None)
        verbose :
            If True, show additional parsing information
        display: Optional[str] :
             (Default value = None)
        verbose: bool :
             (Default value = False)

        Returns
        -------
        
            Tuple of (nodes, connections, state_mapping)

        """
        key = self.cache_key(*args, **kwargs)

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
            result = parse(self.stateful_fn)(*parse_args, **parse_kwargs)
            self.compiled_graph.set(key, result)

        # # Handle display options
        # if display is not None:
        #     self.display(result, mode=display)

        return result

    def display(self, result: Tuple, mode: str = 'text') -> None:
        """Display the parsed results in various formats.

        Parameters
        ----------
        result :
            Parse result tuple, uses last parse result if None
        mode :
            Display mode ('text', 'summary', 'graph')
        result: Tuple :
            
        mode: str :
             (Default value = 'text')

        Returns
        -------

        """
        nodes, connections, state_mapping = result
        if mode == 'text':
            _text_display(nodes, connections, state_mapping)
        else:
            raise ValueError(f"Unknown display mode: {mode}. Choose from 'text', 'summary', 'graph'")

    def clear_cache(self) -> None:
        """Clear the entire cache."""
        self.compiled_graph.clear()

    def __call__(self, *args, **kwargs):
        """Alias for parse() method."""
        return self.parse(*args, **kwargs)

    def __repr__(self) -> str:
        """String representation of the parser."""
        return (
            f"{self.__class__.__name__}("
            f"target='{self.target}', "
            f"cache_size={self.compiled_graph.maxsize}, "
            f"cached_graphs={len(self.compiled_graph)}"
            f")"
        )


def _text_display(
    operations: List[Group],
    connections: List[Connection],
    state_mappings: Dict[str, Any]
):
    """Display comprehensive analysis results for BPU Node Connection Parser

    Parameters
    ----------
    operations: List[Group] :
        
    connections: List[Connection] :
        
    state_mappings: Dict[str :
        
    Any] :
        

    Returns
    -------

    """

    print(f"\nSummary:")
    print(f"   The BPU parser successfully analyzed the neural network into:")
    print(f"   - {len(operations)} computational operations")
    print(f"   - {len(connections)} inter-operation connections")

    # Detailed operation analysis
    print(f"\nNode Analysis:")
    print('----------------------------------------')
    for i, operation in enumerate(operations):
        print(f"\nNode {i}: {operation.name}")

        # Display input states
        if operation.in_states:
            print(f"     Input States ({len(operation.in_states)}):")
            for state in operation.in_states:
                state_info = f"{state.__class__.__name__}"
                if hasattr(state, 'value') and hasattr(state.value, 'shape'):
                    state_info += f" {state.value.dtype}{list(state.value.shape)}"
                print(f"       - {state_info}")
        else:
            print(f"     Input States: None")
        print()

        # Display output states
        if operation.out_states:
            print(f"     Output States ({len(operation.out_states)}):")
            for state in operation.out_states:
                state_info = f"{state.__class__.__name__}"
                if hasattr(state, 'value') and hasattr(state.value, 'shape'):
                    state_info += f" {state.value.dtype}{list(state.value.shape)}"
                print(f"       - {state_info}")
        else:
            print(f"     Output States: None")
        print()

        # Display equations
        print(f"     Equations ({len(operation.eqns)}):")
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
    """

    Parameters
    ----------
    eqn: JaxprEqn :
        
    no :
        

    Returns
    -------

    """
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
    """

    Parameters
    ----------
    num :
        

    Returns
    -------

    """
    if num < 10:
        formater = '{:1d}'
    elif num < 100:
        formater = '{:2d}'
    elif num < 1000:
        formater = '{:3d}'
    else:
        formater = '{:4d}'
    return formater
