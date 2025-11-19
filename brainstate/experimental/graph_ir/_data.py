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
from typing import Dict, Iterable, Iterator, List, NamedTuple, Set, Tuple

from brainstate._compatible_import import ClosedJaxpr, Var
from brainstate._state import State

__all__ = [
    'GraphElem',
    'Graph',
    'Group',
    'Connection',
    'Projection',
    'Output',
    'Input',
    'Spike',
    'CompiledGraph',
]


@dataclass
class GraphElem:
    """Base class for compiled graph elements."""

    jaxpr: ClosedJaxpr


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


@dataclass
class Connection(GraphElem):
    """Describes the primitives that shuttle activity between two groups."""

    pass


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


@dataclass
class Output(GraphElem):
    """Description of how values are extracted from a group."""

    hidden_states: List[State]
    in_states: List[State]
    group: Group


@dataclass
class Input(GraphElem):
    """Description of how external values are injected into a group."""

    group: Group


class Spike(NamedTuple):
    """Opaque surrogate-gradient spike primitive used by the compiler.

    Parameters
    ----------
    state : State
        Logical state that emitted the spike surrogate.
    jaxpr : ClosedJaxpr
        ClosedJaxpr that implements the surrogate-gradient primitive.
    """

    state: State
    jaxpr: ClosedJaxpr


class Graph:
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

        elif backend != 'matplotlib':
            return visualizer.visualzie_matplotlib()

        else:
            raise ValueError(
                f"Unsupported backend: {backend}. "
                f"Choose from 'matplotlib', 'plotly', 'graphviz', or 'networkx'."
            )


class CompiledGraph(NamedTuple):
    """Structured result returned by :func:`graph_ir.compile`.

    Attributes
    ----------
    groups : list[Group]
        Neuron groups that own the state-update subgraphs.
    projections : list[Projection]
        Connection pipelines between groups.
    inputs : list[Input]
        External inputs traced through the jaxpr.
    outputs : list[Output]
        Observations that should be reported to the caller.
    graph : Graph
        Execution order for all components.
    """
    groups: List[Group]
    projections: List[Projection]
    inputs: List[Input]
    outputs: List[Output]
    graph: Graph
