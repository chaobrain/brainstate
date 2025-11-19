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

"""Typed containers and visualization helpers for the experimental graph IR."""

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Iterator, Set, Tuple, Dict, NamedTuple, Sequence, Any, Callable, Hashable, List

import jax

from brainstate._compatible_import import ClosedJaxpr, Var
from brainstate._state import State
from ._utils import get_hidden_name

__all__ = [
    'GraphElem',
    'NeuroGraph',
    'Group',
    'Connection',
    'Projection',
    'Output',
    'Input',
    'Spike',
    'CompiledGraphIR',
]


@dataclass
class GraphElem:
    """Base class for compiled graph IR elements."""

    jaxpr: ClosedJaxpr

    def __repr__(self) -> str:
        """Return a concise representation showing jaxpr signature."""
        n_eqns = len(self.jaxpr.jaxpr.eqns)
        n_invars = len(self.jaxpr.jaxpr.invars)
        n_outvars = len(self.jaxpr.jaxpr.outvars)
        n_constvars = len(self.jaxpr.jaxpr.constvars)

        parts = [
            f"{self.__class__.__name__}(",
            f"eqns={n_eqns}",
            f"invars={n_invars}",
            f"outvars={n_outvars}",
        ]
        if n_constvars > 0:
            parts.append(f"constvars={n_constvars}")
        parts.append(")")
        return " ".join(parts)


@dataclass
class Group(GraphElem):
    """Logical container for a compiled neuron group."""
    hidden_states: List[State]
    in_states: List[State]
    out_states: List[State]
    input_vars: List[Var]
    name: str = "Group"  # Add name field with default value

    @property
    def eqns(self):
        """list[JaxprEqn]: Equations contained in the group's ClosedJaxpr."""
        return self.jaxpr.jaxpr.eqns

    def has_outvar(self, outvar: Var) -> bool:
        """Return True if ``outvar`` is produced by this group.

        Parameters
        ----------
        outvar : Var
            Variable to test.

        Returns
        -------
        bool
            ``True`` when the variable appears in ``jaxpr.outvars``.
        """
        return outvar in self.jaxpr.jaxpr.outvars

    def has_invar(self, invar: Var) -> bool:
        """Return True if ``invar`` is consumed at the group boundary.

        Parameters
        ----------
        invar : Var
            Variable to test.

        Returns
        -------
        bool
            ``True`` when the variable appears in ``jaxpr.invars``.
        """
        return invar in self.jaxpr.jaxpr.invars

    def has_in_state(self, state: State) -> bool:
        """Return True if ``state`` is used in read-only fashion by the group.

        Parameters
        ----------
        state : State
            State instance to search for.

        Returns
        -------
        bool
            ``True`` when the state appears in :attr:`in_states`.
        """
        state_id = id(state)
        return any(id(s) == state_id for s in self.in_states)

    def has_out_state(self, state: State) -> bool:
        """Return True if ``state`` is produced (but not necessarily reused).

        Parameters
        ----------
        state : State
            State instance to search for.

        Returns
        -------
        bool
            ``True`` when the state appears in :attr:`out_states`.
        """
        state_id = id(state)
        return any(id(s) == state_id for s in self.out_states)

    def has_hidden_state(self, state: State) -> bool:
        """Return True if ``state`` belongs to the group's recurrent set.

        Parameters
        ----------
        state : State
            State instance to test.

        Returns
        -------
        bool
            ``True`` when the state appears in :attr:`hidden_states`.
        """
        state_id = id(state)
        return any(id(s) == state_id for s in self.hidden_states)

    def __repr__(self) -> str:
        """Return a detailed representation showing group configuration."""
        n_eqns = len(self.jaxpr.jaxpr.eqns)
        n_hidden = len(self.hidden_states)
        n_in = len(self.in_states)
        n_out = len(self.out_states)
        n_inputs = len(self.input_vars)

        # Get state names if available
        hidden_names = [get_hidden_name(s) for s in self.hidden_states[:3]]
        if len(self.hidden_states) > 3:
            hidden_names.append('...')
        hidden_str = ', '.join(hidden_names)

        return (
            f"{self.name}("
            f"hidden=[{hidden_str}], "
            f"n_eqns={n_eqns}, "
            f"in_states={n_in}, "
            f"out_states={n_out}, "
            f"inputs={n_inputs})"
        )


@dataclass
class Connection(GraphElem):
    """Describes the primitives that shuttle activity between two groups."""

    def __repr__(self) -> str:
        """Return a representation showing connection signature."""
        n_eqns = len(self.jaxpr.jaxpr.eqns)
        # Get the primitive name from the first equation if available
        prim_name = "unknown"
        if n_eqns > 0:
            first_eqn = self.jaxpr.jaxpr.eqns[0]
            prim_name = str(first_eqn.primitive.name) if hasattr(first_eqn.primitive, 'name') else str(
                first_eqn.primitive)

        n_invars = len(self.jaxpr.jaxpr.invars)
        n_outvars = len(self.jaxpr.jaxpr.outvars)

        return (
            f"Connection("
            f"prim={prim_name}, "
            f"invars={n_invars}, "
            f"outvars={n_outvars})"
        )


@dataclass
class Projection(GraphElem):
    """Connection bundle that transfers activity between two groups."""
    hidden_states: List[State]
    in_states: List[State]
    connections: List[Connection]
    pre_group: Group
    post_group: Group

    @property
    def pre(self):
        """Alias for pre_group for backward compatibility with display functions."""
        return self.pre_group

    @property
    def post(self):
        """Alias for post_group for backward compatibility with display functions."""
        return self.post_group

    def __repr__(self) -> str:
        """Return a representation showing projection path."""
        n_conns = len(self.connections)
        n_eqns = len(self.jaxpr.jaxpr.eqns)
        n_hidden = len(self.hidden_states)
        n_in = len(self.in_states)

        # Get group names
        pre_name = self.pre_group.name if hasattr(self.pre_group, 'name') else 'Group'
        post_name = self.post_group.name if hasattr(self.post_group, 'name') else 'Group'

        return (
            f"Projection("
            f"{pre_name} → {post_name}, "
            f"conns={n_conns}, "
            f"eqns={n_eqns}, "
            f"hidden={n_hidden}, "
            f"in_states={n_in})"
        )


@dataclass
class Output(GraphElem):
    """Description of how values are extracted from a group."""

    hidden_states: List[State]
    in_states: List[State]
    group: Group

    def __repr__(self) -> str:
        """Return a representation showing output extraction details."""
        n_eqns = len(self.jaxpr.jaxpr.eqns)
        n_outvars = len(self.jaxpr.jaxpr.outvars)
        n_hidden = len(self.hidden_states)
        n_in = len(self.in_states)

        # Get group name
        group_name = self.group.name if hasattr(self.group, 'name') else 'Group'

        return (
            f"Output("
            f"from={group_name}, "
            f"outvars={n_outvars}, "
            f"eqns={n_eqns}, "
            f"hidden={n_hidden}, "
            f"in_states={n_in})"
        )


@dataclass
class Input(GraphElem):
    """Description of how external values are injected into a group."""

    group: Group

    def __repr__(self) -> str:
        """Return a representation showing input injection details."""
        n_eqns = len(self.jaxpr.jaxpr.eqns)
        n_invars = len(self.jaxpr.jaxpr.invars)
        n_outvars = len(self.jaxpr.jaxpr.outvars)

        # Get group name
        group_name = self.group.name if hasattr(self.group, 'name') else 'Group'

        return (
            f"Input("
            f"to={group_name}, "
            f"invars={n_invars}, "
            f"outvars={n_outvars}, "
            f"eqns={n_eqns})"
        )


@dataclass
class Spike(GraphElem):
    """Opaque surrogate-gradient spike primitive used by the compiler.

    Parameters
    ----------
    hidden_state : State
        Logical state that emitted the spike surrogate.
    jaxpr : ClosedJaxpr
        ClosedJaxpr that implements the surrogate-gradient primitive.
    """
    hidden_state: State

    def __repr__(self) -> str:
        """Return a representation showing spike surrogate details."""
        n_eqns = len(self.jaxpr.jaxpr.eqns)
        # Get state name if available
        state_name = get_hidden_name(self.hidden_state)

        return (
            f"Spike("
            f"state={state_name}, "
            f"eqns={n_eqns})"
        )


class NeuroGraph:
    """Directed graph capturing dependencies between compiled elements."""

    def __init__(self):
        self._nodes: List[GraphElem] = []
        self._id_to_index: Dict[int, int] = {}
        self._forward_edges: Dict[int, Set[int]] = defaultdict(set)
        self._reverse_edges: Dict[int, Set[int]] = defaultdict(set)

    def _ensure_node(self, node: GraphElem) -> int:
        """Ensure ``node`` exists in internal arrays and return its index.

        Parameters
        ----------
        node : GraphElem
            Element to track.

        Returns
        -------
        int
            Stable index assigned to ``node``.
        """
        node_id = id(node)
        existing = self._id_to_index.get(node_id)
        if existing is not None:
            return existing
        index = len(self._nodes)
        self._nodes.append(node)
        self._id_to_index[node_id] = index
        return index

    def add_node(self, node: GraphElem) -> None:
        """Register ``node`` in insertion order, ignoring duplicates.

        Parameters
        ----------
        node : GraphElem
            Element to insert into the graph.
        """
        self._ensure_node(node)

    def add_edge(self, source: GraphElem, target: GraphElem) -> None:
        """Add a directed edge indicating that ``target`` depends on ``source``.

        Parameters
        ----------
        source : GraphElem
            Upstream element that produces data.
        target : GraphElem
            Downstream element that consumes data.
        """
        source_idx = self._ensure_node(source)
        target_idx = self._ensure_node(target)
        if target_idx not in self._forward_edges[source_idx]:
            self._forward_edges[source_idx].add(target_idx)
            self._reverse_edges[target_idx].add(source_idx)

    def nodes(self) -> Tuple[GraphElem, ...]:
        """Return nodes in their recorded execution order.

        Returns
        -------
        tuple[GraphElem, ...]
            Sequence of every node encountered during compilation.
        """
        return tuple(self._nodes)

    def edges(self) -> Iterable[Tuple[GraphElem, GraphElem]]:
        """Iterate over all directed edges.

        Returns
        -------
        Iterable[tuple[GraphElem, GraphElem]]
            Generator yielding ``(source, target)`` pairs.
        """
        for source_idx, targets in self._forward_edges.items():
            for target_idx in targets:
                yield (self._nodes[source_idx], self._nodes[target_idx])

    def predecessors(self, node: GraphElem) -> Tuple[GraphElem, ...]:
        """Return nodes that must execute before ``node``.

        Parameters
        ----------
        node : GraphElem
            Target element.

        Returns
        -------
        tuple[GraphElem, ...]
            Immediate predecessors of ``node``.
        """
        node_idx = self._id_to_index.get(id(node))
        if node_idx is None:
            return tuple()
        return tuple(self._nodes[i] for i in self._reverse_edges.get(node_idx, ()))

    def successors(self, node: GraphElem) -> Tuple[GraphElem, ...]:
        """Return nodes that depend on ``node``.

        Parameters
        ----------
        node : GraphElem
            Origin element.

        Returns
        -------
        tuple[GraphElem, ...]
            Immediate successors of ``node``.
        """
        node_idx = self._id_to_index.get(id(node))
        if node_idx is None:
            return tuple()
        return tuple(self._nodes[i] for i in self._forward_edges.get(node_idx, ()))

    def edge_count(self) -> int:
        """Return number of directed edges in the graph.

        Returns
        -------
        int
            Total number of edges currently recorded.
        """
        return sum(len(targets) for targets in self._forward_edges.values())

    def __len__(self) -> int:
        return len(self._nodes)

    def __repr__(self) -> str:
        # from ._display import TextDisplayer
        # return TextDisplayer(self).display()

        num_nodes = len(self._nodes)
        num_edges = self.edge_count()
        return f"<Graph with {num_nodes} nodes and {num_edges} edges>"

    def __iter__(self) -> Iterator[GraphElem]:
        return iter(self._nodes)

    def visualize(
        self,
        backend='matplotlib',
        layout='hierarchical',
        interactive=True,
        show_details=True,
        node_size='auto',
        edge_width='auto',
        colorscheme='default',
        export_path=None,
        figsize='auto',
        **kwargs
    ):
        """Visualize the graph using various backends and layout algorithms.

        Parameters
        ----------
        backend : {'matplotlib', 'plotly', 'graphviz', 'networkx'}, optional
            Visualization backend. Default is 'matplotlib'.

            * 'matplotlib': Static publication-quality plots
            * 'plotly': Interactive web-based visualization with hover tooltips, zoom, and pan
            * 'graphviz': Professional hierarchical layouts using Graphviz engine
            * 'networkx': Advanced layout algorithms (spring, spectral, etc.)

        layout : str, optional
            Layout algorithm. Default is 'hierarchical'.

            * For matplotlib/plotly: 'hierarchical' (topological layers)
            * For graphviz: 'dot', 'neato', 'fdp', 'sfdp', 'circo', 'twopi'
            * For networkx: 'spring', 'kamada_kawai', 'spectral', 'circular', 'shell'

        interactive : bool, optional
            Enable interactive features for plotly backend. Default is True.
        show_details : bool, optional
            Show detailed node/edge metadata in tooltips or labels. Default is True.
        node_size : str or dict, optional
            Node sizing strategy. Default is 'auto'.

            * 'auto': Size nodes by complexity (equation/state count)
            * 'uniform': All nodes same size
            * dict: Custom mapping ``{node_idx: size}``

        edge_width : str or dict, optional
            Edge width strategy. Default is 'auto' (uniform).
        colorscheme : str, optional
            Color scheme: 'default', 'pastel', 'vibrant', 'colorblind'. Default is 'default'.
        export_path : str, optional
            Export path. Format inferred from extension (.png, .pdf, .svg, .html).
        figsize : tuple or 'auto', optional
            Figure size as (width, height). Default is 'auto'.
        ax : matplotlib.axes.Axes, optional
            Matplotlib axes (only for matplotlib backend).
        **kwargs
            Additional backend-specific options.

        Returns
        -------
        object
            Backend-specific return value:

            * matplotlib: Figure object
            * plotly: plotly.graph_objects.Figure
            * graphviz: graphviz.Digraph
            * networkx: matplotlib Figure

        Raises
        ------
        ValueError
            If unsupported backend or layout is specified.
        RuntimeError
            If required backend library is not installed.

        Examples
        --------
        Basic matplotlib visualization::

            fig = graph.visualize()
            plt.show()

        Interactive Plotly with export::

            fig = graph.visualize(backend='plotly', export_path='graph.html')
            fig.show()

        Graphviz with force-directed layout::

            dot = graph.visualize(backend='graphviz', layout='fdp')
            dot.render('graph', format='pdf')

        NetworkX spring layout::

            fig = graph.visualize(backend='networkx', layout='spring')
            plt.show()

        Custom colorscheme::

            fig = graph.visualize(colorscheme='colorblind')

        """
        # Use new backend system for non-matplotlib backends
        from ._display import GraphDisplayer
        visualizer = GraphDisplayer(self)
        if backend == 'plotly':
            return visualizer.visualize_plotly(
                layout=layout, interactive=interactive, show_details=show_details,
                node_size=node_size, edge_width=edge_width, colorscheme=colorscheme,
                export_path=export_path, figsize=figsize, **kwargs
            )
        elif backend == 'graphviz':
            return visualizer.visualize_graphviz(
                layout=layout, show_details=show_details, colorscheme=colorscheme,
                export_path=export_path, **kwargs
            )
        elif backend == 'networkx':
            return visualizer.visualize_networkx(
                layout=layout, node_size=node_size, edge_width=edge_width,
                colorscheme=colorscheme, figsize=figsize, show_details=show_details,
                export_path=export_path, **kwargs
            )

        elif backend == 'matplotlib':
            return visualizer.visualzie_matplotlib()

        else:
            raise ValueError(
                f"Unsupported backend: {backend}. "
                f"Choose from 'matplotlib', 'plotly', 'graphviz', or 'networkx'."
            )


class CompiledGraphIR(NamedTuple):
    """Structured result returned by :func:`neuron_ir.compile`.

    Attributes
    ----------
    groups : list[Group]
        Neuron groups that own the state-update subgraphs.
    projections : list[Projection]
        ConnectionIR pipelines between groups.
    inputs : list[Input]
        External inputs traced through the jaxpr.
    outputs : list[Output]
        Observations that should be reported to the caller.
    graph : NeuroGraph
        Execution order for all components.
    """
    # graph IR data
    groups: List[Group]
    projections: List[Projection]
    inputs: List[Input]
    outputs: List[Output]
    graph: NeuroGraph

    # others
    static_argnames: Sequence
    static_argnums: Sequence
    cache_fn: Callable
    cache_key: Hashable
    out_treedef: Any
    jaxpr: ClosedJaxpr
    in_states: Sequence[State]
    out_states: Sequence[State]
    invar_to_state: Dict[Var, State]
    outvar_to_state: Dict[Var, State]
    state_to_invars: Dict[State, Sequence[Var]]
    state_to_outvars: Dict[State, Sequence[Var]]

    def run(self, *args, mode: str = 'compiled', **kwargs) -> Any:
        """Execute the parsed function in the requested mode.

        Parameters
        ----------
        *args, **kwargs
            Runtime arguments forwarded to the original stateful function.
        mode : {'compiled', 'original', 'debug'}, optional
            Execution mode. ``'compiled'`` (default) uses the graph IR,
            ``'original'`` evals the raw ClosedJaxpr, and ``'debug'`` returns a
            tuple with both results.

        Returns
        -------
        Any
            Result of the selected execution path. ``'debug'`` returns a tuple
            ``(original, compiled)``.
        """
        if mode == 'compiled':
            return self.run_compiled_graph(*args, **kwargs, assign_state_val=True)

        elif mode == 'original':
            return self.run_original_jaxpr(*args, **kwargs, assign_state_val=True)

        elif mode == 'debug':
            result = self.run_original_jaxpr(*args, **kwargs, assign_state_val=False)
            compiled = self.run_compiled_graph(*args, **kwargs, assign_state_val=False)
            return result, compiled

        else:
            raise ValueError(f"Unknown mode: {mode}")

    def run_original_jaxpr(self, *args, **kwargs) -> Any:
        """Evaluate the original ClosedJaxpr for comparison/debugging."""
        return self._run_impl(
            lambda *data: jax.core.eval_jaxpr(self.jaxpr.jaxpr, self.jaxpr.consts, *data),
            *args,
            **kwargs,
        )

    def run_compiled_graph(self, *args, **kwargs) -> Any:
        """Execute the compiled graph IR representation."""
        return self._run_impl(self._run_graph, *args, **kwargs)

    def _run_impl(self, impl, *args, assign_state_val: bool = True, **kwargs) -> Any:
        """Shared argument/treedef handling for the execution helpers."""
        # data check
        if self.cache_fn(*args, **kwargs) != self.cache_key:
            raise ValueError("Cache key mismatch. The function has been called with different arguments.")

        # inputs
        in_state_val = [st.value for st in self.in_states]
        kwargs = {k: v for k, v in kwargs.items() if k not in self.static_argnames}  # remove static kwargs
        args = tuple(args[i] for i in range(len(args)) if i not in self.static_argnums)
        args = jax.tree.flatten((args, kwargs, in_state_val))[0]

        # run jaxpr
        jaxpr_outs = impl(*args)

        # outputs
        out, new_state_vals = self.out_treedef.unflatten(jaxpr_outs)
        if len(new_state_vals) != len(self.out_states):
            raise ValueError(f'State length mismatch in output: expected '
                             f'{len(self.out_states)} states, got {len(new_state_vals)}')
        if assign_state_val:
            for st, val in zip(self.out_states, new_state_vals):
                st.restore_value(val)
        return out

    def _run_graph(self, *args) -> Any:
        """Run the compiled graph while maintaining a variable environment.

        Parameters
        ----------
        *args
            Flattened inputs expected by the ClosedJaxpr.

        Returns
        -------
        Any
            Reconstructed model outputs.
        """
        # Build variable environment: Var -> value mapping
        var_env = {}

        # Step 1: Initialize environment with input arguments
        self._initialize_var_env(var_env, args)

        # Step 2: Execute components in graph
        for component in self.graph:
            if isinstance(component, Input):
                self._execute_input(component, var_env)
            elif isinstance(component, Group):
                self._execute_group(component, var_env)
            elif isinstance(component, Projection):
                self._execute_projection(component, var_env)
            elif isinstance(component, Output):
                self._execute_output(component, var_env)

        # Step 3: Collect outputs from environment
        outputs = self._collect_outputs(var_env)

        return outputs

    def _initialize_var_env(self, var_env: Dict[Var, Any], args: Tuple) -> None:
        """Seed the variable environment with the function inputs.

        Parameters
        ----------
        var_env : dict[Var, Any]
            Mutable mapping from variables to runtime values.
        args : tuple
            Positional arguments following the ClosedJaxpr order.
        """
        # Map to jaxpr invars
        assert len(args) == len(self.jaxpr.jaxpr.invars), (
            f"Argument count mismatch: expected {len(self.jaxpr.jaxpr.invars)}, got {len(args)}"
        )
        for var, val in zip(self.jaxpr.jaxpr.invars, args):
            var_env[var] = val
        for var, val in zip(self.jaxpr.constvars, self.jaxpr.consts):
            var_env[var] = val

    def _execute_input(self, input_comp: Input, var_env: Dict[Var, Any]) -> None:
        """Evaluate an :class:`Input` component and store its outputs."""
        # Gather input values from environment
        input_vals = [var_env[var] for var in input_comp.jaxpr.jaxpr.invars]

        # Execute the input jaxpr
        results = jax.core.eval_jaxpr(input_comp.jaxpr.jaxpr, input_comp.jaxpr.consts, *input_vals)

        # Handle single vs multiple outputs
        if not isinstance(results, (tuple, list)):
            results = (results,)

        # Store results in environment
        for var, val in zip(input_comp.jaxpr.jaxpr.outvars, results):
            var_env[var] = val

    def _execute_group(self, group: Group, var_env: Dict[Var, Any]) -> None:
        """Evaluate a :class:`Group` subgraph using values from ``var_env``."""
        # Gather input values from environment
        input_vals = []
        for var in group.jaxpr.jaxpr.invars:
            if var not in var_env:
                raise RuntimeError(
                    f"Variable {var} not found in environment when executing {group.name}"
                )
            input_vals.append(var_env[var])

        # Execute the group jaxpr
        results = jax.core.eval_jaxpr(group.jaxpr.jaxpr, group.jaxpr.consts, *input_vals)

        # Handle single vs multiple outputs
        if not isinstance(results, (tuple, list)):
            results = (results,)

        # Store results in environment
        for var, val in zip(group.jaxpr.jaxpr.outvars, results):
            var_env[var] = val

    def _execute_projection(self, projection: Projection, var_env: Dict[Var, Any]) -> None:
        """Evaluate a :class:`Projection` component, including const fallbacks."""
        # Gather input values from environment
        input_vals = []
        for var in projection.jaxpr.jaxpr.invars:
            if var in var_env:
                input_vals.append(var_env[var])
            else:
                # This might be a constvar, check in the original jaxpr
                if var in self.jaxpr.jaxpr.constvars:
                    # Find the index and use the corresponding const
                    idx = self.jaxpr.jaxpr.constvars.index(var)
                    input_vals.append(self.jaxpr.consts[idx])
                else:
                    raise RuntimeError(f"Variable {var} not found in environment or constvars")

        # Execute the projection jaxpr
        results = jax.core.eval_jaxpr(projection.jaxpr.jaxpr, projection.jaxpr.consts, *input_vals)

        # Handle single vs multiple outputs
        if not isinstance(results, (tuple, list)):
            results = (results,)

        # Store results in environment
        for var, val in zip(projection.jaxpr.jaxpr.outvars, results):
            var_env[var] = val

    def _execute_output(self, output: Output, var_env: Dict[Var, Any]) -> None:
        """Evaluate an :class:`Output` component using values in ``var_env``."""
        # Gather input values from environment
        input_vals = [var_env[var] for var in output.jaxpr.jaxpr.invars]

        # Execute the output jaxpr
        results = jax.core.eval_jaxpr(output.jaxpr.jaxpr, output.jaxpr.consts, *input_vals)

        # Handle single vs multiple outputs
        if not isinstance(results, (tuple, list)):
            results = (results,)

        # Store results in environment
        for var, val in zip(output.jaxpr.jaxpr.outvars, results):
            var_env[var] = val

    def _collect_outputs(self, var_env: Dict[Var, Any]) -> Any:
        """Assemble model outputs from the variable environment.

        Parameters
        ----------
        var_env : dict[Var, Any]
            Environment after graph execution.

        Returns
        -------
        Any
            Single value or tuple that mirrors the original function's outputs.
        """
        output_vals = []
        for var in self.jaxpr.jaxpr.outvars:
            if var not in var_env:
                raise RuntimeError(f"Output variable {var} not found in environment")
            output_vals.append(var_env[var])

        # Return single value or tuple based on output count
        if len(output_vals) == 1:
            return output_vals[0]
        else:
            return tuple(output_vals)

    def __repr__(self) -> str:
        """Return a concise summary of the compiled graph IR.

        Returns
        -------
        str
            Human-readable representation showing compilation statistics.
        """
        n_groups = len(self.groups)
        n_projections = len(self.projections)
        n_inputs = len(self.inputs)
        n_outputs = len(self.outputs)
        n_graph_nodes = len(self.graph)
        n_in_states = len(self.in_states)
        n_out_states = len(self.out_states)

        # Get total equation count
        total_eqns = len(self.jaxpr.jaxpr.eqns)

        # Build group summary
        group_names = [g.name for g in self.groups[:3]]
        if n_groups > 3:
            group_names.append('...')
        groups_str = ', '.join(group_names)

        return (
            f"<CompiledGraphIR: "
            f"groups={n_groups}({groups_str}), "
            f"projs={n_projections}, "
            f"inputs={n_inputs}, "
            f"outputs={n_outputs}, "
            f"states={n_in_states}→{n_out_states}, "
            f"total_eqns={total_eqns}, "
            f"graph_nodes={n_graph_nodes}>"
        )
