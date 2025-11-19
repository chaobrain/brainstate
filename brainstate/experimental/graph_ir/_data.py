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

from collections import defaultdict, deque
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

    def visualize(self, ax=None):
        """Visualize node dependencies in layered top-down layout.

        Parameters
        ----------
        ax :
            Optional matplotlib Axes to draw on. When ``None`` a new figure and
            axes will be created.

        Returns
        -------
        matplotlib.figure.Figure
            The figure that contains the visualization so callers can further
            customize or display it (e.g. via ``plt.show()``).

        Raises
        ------
        RuntimeError
            If ``matplotlib`` is not installed in the current environment.

        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import FancyBboxPatch
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Graph.visualize requires matplotlib to be installed."
            ) from exc

        if not self._nodes:
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.text(
                0.5,
                0.5,
                "Graph is empty",
                ha='center',
                va='center',
                fontsize=12,
            )
            ax.axis('off')
            fig.tight_layout()
            return fig

        num_nodes = len(self._nodes)
        in_degree = {i: len(self._reverse_edges.get(i, ())) for i in range(num_nodes)}
        levels = {i: 0 for i in range(num_nodes)}
        queue = deque(sorted(idx for idx, deg in in_degree.items() if deg == 0))
        if not queue:
            queue = deque(range(num_nodes))
        enqueued = set(queue)
        processed = set()
        while queue:
            idx = queue.popleft()
            processed.add(idx)
            for succ in sorted(self._forward_edges.get(idx, ())):
                levels[succ] = max(levels.get(succ, 0), levels[idx] + 1)
                in_degree[succ] = in_degree.get(succ, 0) - 1
                if in_degree[succ] <= 0 and succ not in processed and succ not in enqueued:
                    queue.append(succ)
                    enqueued.add(succ)

        if len(processed) != num_nodes:
            remaining = set(range(num_nodes)) - processed
            for idx in remaining:
                preds = self._reverse_edges.get(idx, ())
                if preds:
                    max_pred = max(levels.get(pred, 0) for pred in preds)
                    levels[idx] = max(levels.get(idx, 0), max_pred + 1)
                else:
                    levels[idx] = 0

        layer_map = defaultdict(list)
        for idx, level in levels.items():
            layer_map[level].append(idx)
        normalized_layers = []
        for _, nodes in sorted(layer_map.items(), key=lambda item: item[0]):
            normalized_layers.append(sorted(nodes))

        num_layers = len(normalized_layers)
        max_width = max((len(layer) for layer in normalized_layers), default=1)
        x_gap = 2.5
        y_gap = 2.0
        node_width = 1.8
        node_height = 0.9

        x_positions: Dict[int, float] = {}
        y_positions: Dict[int, float] = {}
        for layer_idx, layer_nodes in enumerate(normalized_layers):
            if not layer_nodes:
                continue
            row_width = len(layer_nodes)
            x_offset = (max_width - row_width) * 0.5 * x_gap
            y_val = (num_layers - layer_idx - 1) * y_gap
            for pos, node_idx in enumerate(layer_nodes):
                x_positions[node_idx] = x_offset + pos * x_gap
                y_positions[node_idx] = y_val

        if ax is None:
            fig_width = max(6.0, max_width * 1.6)
            fig_height = max(4.0, num_layers * 1.5)
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        else:
            fig = ax.figure

        def _node_label(node: GraphElem) -> str:
            if isinstance(node, Group):
                return f"Group\\n{node.name}"
            if isinstance(node, Projection):
                return f"Projection\\n{node.pre_group.name} → {node.post_group.name}"
            if isinstance(node, Input):
                return f"Input\\n→ {node.group.name}"
            if isinstance(node, Output):
                return f"Output\\n{node.group.name} →"
            if isinstance(node, Connection):
                return f"Connection\\n{node.jaxpr.jaxpr.name if hasattr(node.jaxpr, 'jaxpr') else ''}"
            return type(node).__name__

        def _node_style(node: GraphElem) -> Tuple[str, str]:
            if isinstance(node, Input):
                return "#E3F2FD", "#1565C0"
            if isinstance(node, Output):
                return "#FCE4EC", "#AD1457"
            if isinstance(node, Projection):
                return "#FFF3E0", "#E65100"
            if isinstance(node, Connection):
                return "#EDE7F6", "#5E35B1"
            return "#E8F5E9", "#1B5E20"  # Groups and fallbacks

        for idx, node in enumerate(self._nodes):
            x = x_positions.get(idx, 0.0)
            y = y_positions.get(idx, 0.0)
            facecolor, edgecolor = _node_style(node)
            patch = FancyBboxPatch(
                (x - node_width / 2, y - node_height / 2),
                node_width,
                node_height,
                boxstyle='round,pad=0.25',
                linewidth=1.4,
                facecolor=facecolor,
                edgecolor=edgecolor,
            )
            ax.add_patch(patch)
            ax.text(
                x,
                y,
                _node_label(node),
                ha='center',
                va='center',
                fontsize=9,
                color='#263238',
            )

        for source_idx, targets in self._forward_edges.items():
            sx = x_positions.get(source_idx, 0.0)
            sy = y_positions.get(source_idx, 0.0)
            for target_idx in targets:
                tx = x_positions.get(target_idx, 0.0)
                ty = y_positions.get(target_idx, 0.0)
                ax.annotate(
                    '',
                    xy=(tx, ty + node_height / 2),
                    xytext=(sx, sy - node_height / 2),
                    arrowprops=dict(
                        arrowstyle='-|>',
                        color='#546E7A',
                        linewidth=1.2,
                        shrinkA=0,
                        shrinkB=0,
                    ),
                )

        min_x = min(x_positions.values(), default=0.0) - x_gap
        max_x = max(x_positions.values(), default=0.0) + x_gap
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(-y_gap, num_layers * y_gap + y_gap)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Model Dependency Graph", fontsize=12, fontweight='bold')
        ax.axis('off')
        fig.tight_layout()
        return fig

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

    def display(self, mode='text'):
        """Render the compiled graph using the requested presentation mode.

        Parameters
        ----------
        mode : {'text', 'ascii', 'visualization', 'viz'}, optional
            Presentation style. ``'text'`` prints a verbose summary,
            ``'ascii'`` shows a minimal ASCII visualization, and
            ``'visualization'``/``'viz'`` returns a Graphviz ``Digraph``.

        Returns
        -------
        graphviz.Digraph or None
            ``Digraph`` when ``mode`` requests the visualization backend,
            otherwise ``None`` because the information is printed to stdout.

        Raises
        ------
        ValueError
            If an unsupported mode string is provided.
        """
        if mode == 'text':
            return self._text_display()
        elif mode == 'visualization' or mode == 'viz':
            return self._vis_display()
        elif mode == 'ascii':
            return self._ascii_display()
        else:
            raise ValueError(f"Unknown display mode: {mode}. Use 'text', 'ascii', or 'visualization'")

    def _text_display(self):
        """Print a detailed, sectioned summary of the compiled graph."""
        print("=" * 80)
        print("COMPILED GRAPH STRUCTURE")
        print("=" * 80)

        # Summary
        print(f"\nSummary:")
        print(f"  Groups:      {len(self.groups)}")
        print(f"  Projections: {len(self.projections)}")
        print(f"  Inputs:      {len(self.inputs)}")
        print(f"  Outputs:     {len(self.outputs)}")
        print(f"  Call graph:  {len(self.graph)} nodes, {self.graph.edge_count()} edges")

        # Groups
        if self.groups:
            print("\n" + "=" * 80)
            print("GROUPS")
            print("=" * 80)
            for i, group in enumerate(self.groups):
                print(f"\n[{i}] {group.name}")
                print(f"  Hidden states: {len(group.hidden_states)}")
                if group.hidden_states:
                    for state in group.hidden_states:
                        print(f"    - {_format_state(state)}")
                print(f"  In states:     {len(group.in_states)}")
                if group.in_states:
                    for state in group.in_states:
                        print(f"    - {_format_state(state)}")
                print(f"  Out states:    {len(group.out_states)}")
                if group.out_states:
                    for state in group.out_states:
                        print(f"    - {_format_state(state)}")
                print(f"  Input vars:    {len(group.input_vars)}")
                print(f"  Equations:     {len(group.jaxpr.jaxpr.eqns)}")

        # Projections
        if self.projections:
            print("\n" + "=" * 80)
            print("PROJECTIONS")
            print("=" * 80)
            for i, proj in enumerate(self.projections):
                print(f"\n[{i}] {proj.pre_group.name} -> {proj.post_group.name}")
                print(f"  Hidden states: {len(proj.hidden_states)}")
                if proj.hidden_states:
                    for state in proj.hidden_states:
                        print(f"    - {_format_state(state)}")
                print(f"  In states:     {len(proj.in_states)}")
                if proj.in_states:
                    for state in proj.in_states:
                        print(f"    - {_format_state(state)}")
                print(f"  Connections:   {len(proj.connections)}")
                print(f"  Equations:     {len(proj.jaxpr.jaxpr.eqns)}")
                # Show connection primitive names
                if proj.connections:
                    print(f"  Connection ops:")
                    for conn in proj.connections:
                        if conn.jaxpr.jaxpr.eqns:
                            for eqn in conn.jaxpr.jaxpr.eqns:
                                print(f"    - {eqn.primitive.name}")

        # Inputs
        if self.inputs:
            print("\n" + "=" * 80)
            print("INPUTS")
            print("=" * 80)
            for i, inp in enumerate(self.inputs):
                print(f"\n[{i}] Input -> {inp.group.name}")
                print(f"  Input vars:    {len(inp.jaxpr.jaxpr.invars)}")
                print(f"  Equations:     {len(inp.jaxpr.jaxpr.eqns)}")

        # Outputs
        if self.outputs:
            print("\n" + "=" * 80)
            print("OUTPUTS")
            print("=" * 80)
            for i, out in enumerate(self.outputs):
                print(f"\n[{i}] {out.group.name} -> Output")
                print(f"  Hidden states: {len(out.hidden_states)}")
                if out.hidden_states:
                    for state in out.hidden_states:
                        print(f"    - {_format_state(state)}")
                print(f"  Equations:     {len(out.jaxpr.jaxpr.eqns)}")

        # Call graph order
        print("\n" + "=" * 80)
        print("EXECUTION ORDER")
        print("=" * 80)
        print()
        for i, component in enumerate(self.graph):
            comp_type = type(component).__name__
            if isinstance(component, Group):
                print(f"{i:2d}. [Group]      {component.name}")
            elif isinstance(component, Projection):
                print(f"{i:2d}. [Projection] {component.pre_group.name} -> {component.post_group.name}")
            elif isinstance(component, Input):
                print(f"{i:2d}. [Input]      -> {component.group.name}")
            elif isinstance(component, Output):
                print(f"{i:2d}. [Output]     {component.group.name} ->")
            else:
                print(f"{i:2d}. [{comp_type}]")

        print("\n" + "=" * 80)

    def _vis_display(self):
        """Return a Graphviz ``Digraph`` describing the execution graph.

        Returns
        -------
        graphviz.Digraph
            Graph description rendered with colors per component type.

        Raises
        ------
        RuntimeError
            If :mod:`graphviz` is not installed.
        """
        try:
            from graphviz import Digraph
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "CompiledGraph.display(mode='visualization') requires graphviz."
            ) from exc

        dot = Digraph("CompiledGraph")
        dot.attr(rankdir='TB', splines='spline')

        def _label(node: GraphElem) -> str:
            if isinstance(node, Group):
                return f"Group: {node.name}"
            if isinstance(node, Projection):
                return f"Projection: {node.pre_group.name}→{node.post_group.name}"
            if isinstance(node, Input):
                return f"Input → {node.group.name}"
            if isinstance(node, Output):
                return f"{node.group.name} → Output"
            return type(node).__name__

        for node in self.graph.nodes():
            style = {'shape': 'box', 'style': 'rounded,filled', 'fillcolor': '#E8F5E9'}
            if isinstance(node, Projection):
                style['fillcolor'] = '#FFF3E0'
            elif isinstance(node, Input):
                style['fillcolor'] = '#E3F2FD'
            elif isinstance(node, Output):
                style['fillcolor'] = '#FCE4EC'
            dot.node(str(id(node)), _label(node), **style)

        for source, target in self.graph.edges():
            dot.edge(str(id(source)), str(id(target)))

        return dot

    def _ascii_display(self):
        """Render a compact ASCII-art visualization of the compiled graph."""
        print("=" * 80)
        print("COMPILED GRAPH - ASCII VISUALIZATION")
        print("=" * 80)
        print()

        # Build component to ID mapping
        comp_to_id = {}
        for i, group in enumerate(self.groups):
            comp_to_id[id(group)] = (f'G{i}', group.name)
        for i, proj in enumerate(self.projections):
            comp_to_id[id(proj)] = (f'P{i}', f'{proj.pre_group.name}->{proj.post_group.name}')

        # Show legend
        print("Legend:")
        print("  [G#] = Group")
        print("  (P#) = Projection")
        print("  <I#> = Input")
        print("  {O#} = Output")
        print()

        # Show execution order
        print("Execution Order:")
        print()
        for i, component in enumerate(self.graph):
            indent = "  " * min(i, 3)  # Limit indentation
            if isinstance(component, Group):
                comp_id, comp_name = comp_to_id[id(component)]
                print(f"{indent}{i}. [{comp_id}] {comp_name}")
            elif isinstance(component, Projection):
                comp_id, comp_name = comp_to_id[id(component)]
                print(f"{indent}{i}. ({comp_id}) {comp_name}")
            elif isinstance(component, Input):
                print(f"{indent}{i}. <I> → {component.group.name}")
            elif isinstance(component, Output):
                print(f"{indent}{i}. {{O}} {component.group.name} →")

        print()
        print("-" * 80)
        print("Data Flow Diagram:")
        print()

        # Draw data flow
        # First, collect all inputs
        if self.inputs:
            for i, inp in enumerate(self.inputs):
                target = comp_to_id.get(id(inp.group), ('?', 'Unknown'))[0]
                print(f"  <I{i}>  ────inject────>  [{target}]")
            print()

        # Draw groups and their relationships
        drawn_groups = set()
        for proj in self.projections:
            pre_id, pre_name = comp_to_id[id(proj.pre_group)]
            post_id, post_name = comp_to_id[id(proj.post_group)]
            proj_id, proj_name = comp_to_id[id(proj)]

            if id(proj.pre_group) not in drawn_groups:
                print(f"  [{pre_id}] {pre_name}")
                print(
                    f"    | ({len(proj.pre_group.hidden_states)} states, {len(proj.pre_group.jaxpr.jaxpr.eqns)} eqns)")
                drawn_groups.add(id(proj.pre_group))

            # Show projection
            conn_info = f"{len(proj.connections)} conn" if proj.connections else "no conn"
            print(f"    +---spikes---> ({proj_id}) {proj_name}")
            print(f"    |              ({conn_info})")
            print(f"    +---currents-->")
            print()

            if id(proj.post_group) not in drawn_groups:
                print(f"  [{post_id}] {post_name}")
                print(
                    f"    | ({len(proj.post_group.hidden_states)} states, {len(proj.post_group.jaxpr.jaxpr.eqns)} eqns)")
                drawn_groups.add(id(proj.post_group))
            print()

        # Draw remaining groups (not involved in projections)
        for group in self.groups:
            if id(group) not in drawn_groups:
                group_id, group_name = comp_to_id[id(group)]
                print(f"  [{group_id}] {group_name}")
                print(f"    ({len(group.hidden_states)} states, {len(group.jaxpr.jaxpr.eqns)} eqns)")
                print()

        # Draw outputs
        if self.outputs:
            print()
            for i, out in enumerate(self.outputs):
                source = comp_to_id.get(id(out.group), ('?', 'Unknown'))[0]
                print(f"  [{source}]  ────extract────>  {{O{i}}}")

        print()
        print("=" * 80)


def _format_state(state: State) -> str:
    """Return a compact textual summary for ``state``."""
    if hasattr(state, 'value'):
        val = state.value
        if hasattr(val, 'shape'):
            shape_str = f"shape={val.shape}"
        else:
            shape_str = f"value={val}"
        return f"{type(state).__name__}({shape_str})"
    return f"{type(state).__name__}()"
