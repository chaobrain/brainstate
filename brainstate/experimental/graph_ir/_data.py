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
from typing import Dict, Iterable, Iterator, List, NamedTuple, Set, Tuple, Union

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
        """ """
        return self.jaxpr.jaxpr.eqns

    def has_outvar(self, outvar: Var) -> bool:
        """

        Parameters
        ----------
        outvar: Var :
            

        Returns
        -------
        type
            

        """
        return outvar in self.jaxpr.jaxpr.outvars

    def has_invar(self, invar: Var) -> bool:
        """

        Parameters
        ----------
        invar: Var :
            

        Returns
        -------
        type
            

        """
        return invar in self.jaxpr.jaxpr.invars

    def has_in_state(self, state: State) -> bool:
        """

        Parameters
        ----------
        state: State :
            

        Returns
        -------
        type
            

        """
        state_id = id(state)
        return any(id(s) == state_id for s in self.in_states)

    def has_out_state(self, state: State) -> bool:
        """

        Parameters
        ----------
        state: State :
            

        Returns
        -------
        type
            

        """
        state_id = id(state)
        return any(id(s) == state_id for s in self.out_states)

    def has_hidden_state(self, state: State) -> bool:
        """

        Parameters
        ----------
        state: State :
            

        Returns
        -------
        type
            

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
    """Opaque surrogate-gradient spike representation used by the compiler.
    
    Notes:
        The backing JAXPR commonly corresponds to a Heaviside surrogate
        gradient such as ``heaviside_surrogate_gradient``.

    Parameters
    ----------

    Returns
    -------

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
        """Register a node in insertion order, ignoring duplicates."""
        self._ensure_node(node)

    def add_edge(self, source: GraphElem, target: GraphElem) -> None:
        """Add a directed edge indicating that ``target`` depends on ``source``."""
        source_idx = self._ensure_node(source)
        target_idx = self._ensure_node(target)
        if target_idx not in self._forward_edges[source_idx]:
            self._forward_edges[source_idx].add(target_idx)
            self._reverse_edges[target_idx].add(source_idx)

    def nodes(self) -> Tuple[GraphElem, ...]:
        """Return nodes in their recorded execution order."""
        return tuple(self._nodes)

    def edges(self) -> Iterable[Tuple[GraphElem, GraphElem]]:
        """Iterate over all directed edges."""
        for source_idx, targets in self._forward_edges.items():
            for target_idx in targets:
                yield (self._nodes[source_idx], self._nodes[target_idx])

    def predecessors(self, node: GraphElem) -> Tuple[GraphElem, ...]:
        """Return nodes that must execute before ``node``."""
        node_idx = self._id_to_index.get(id(node))
        if node_idx is None:
            return tuple()
        return tuple(self._nodes[i] for i in self._reverse_edges.get(node_idx, ()))

    def successors(self, node: GraphElem) -> Tuple[GraphElem, ...]:
        """Return nodes that depend on ``node``."""
        node_idx = self._id_to_index.get(id(node))
        if node_idx is None:
            return tuple()
        return tuple(self._nodes[i] for i in self._forward_edges.get(node_idx, ()))

    def edge_count(self) -> int:
        """Return number of directed edges in the graph."""
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
        *,
        title: str = "Compiled Call Graph",
        formatter: Optional[Callable[[GraphElem], str]] = None,
        highlight: Optional[Iterable[GraphElem]] = None,
        positions: Optional['np.ndarray'] = None,
        width: int = 900,
        height: int = 600,
        **kwargs,
    ):
        """Visualize the graph using :mod:`braintools.visualize` helpers."""
        if not self._nodes:
            raise ValueError("Graph is empty; nothing to visualize.")

        try:
            from braintools.visualize import interactive_network
        except ImportError as exc:
            raise ImportError(
                "braintools.visualize is required for graph visualization. "
                "Install it with `pip install braintools[visualize]`."
            ) from exc

        import numpy as np

        adjacency = self._build_adjacency_matrix(np)
        labels = [self._format_node_label(node, formatter) for node in self._nodes]
        node_colors = self._build_node_colors(np, highlight)
        if positions is None:
            positions = self._default_positions(np)
        else:
            positions = np.asarray(positions, dtype=float)
            expected = (len(self._nodes), 2)
            if positions.shape != expected:
                raise ValueError(f"positions must have shape {expected}, got {positions.shape}")

        return interactive_network(
            adjacency=adjacency,
            positions=positions,
            node_labels=labels,
            node_colors=node_colors,
            title=title,
            width=width,
            height=height,
            **kwargs,
        )

    def _build_adjacency_matrix(self, np_mod):
        adjacency = np_mod.zeros((len(self._nodes), len(self._nodes)), dtype=float)
        for source_idx, targets in self._forward_edges.items():
            for target_idx in targets:
                adjacency[source_idx, target_idx] = 1.0
        return adjacency

    def _format_node_label(
        self, node: GraphElem, formatter: Optional[Callable[[GraphElem], str]]
    ) -> str:
        if formatter is not None:
            return formatter(node)
        if isinstance(node, Input):
            return f"Input → {node.group.name}"
        if isinstance(node, Group):
            return node.name
        if isinstance(node, Projection):
            return f"{node.pre_group.name} → {node.post_group.name}"
        if isinstance(node, Output):
            return f"{node.group.name} → Output"
        return type(node).__name__

    def _build_node_colors(self, np_mod, highlight: Optional[Iterable[GraphElem]]):
        type_buckets = {
            Input: 0.0,
            Group: 1.0,
            Projection: 2.0,
            Output: 3.0,
        }
        highlight_ids = {id(node) for node in highlight} if highlight else set()
        colors = []
        for node in self._nodes:
            base = type_buckets.get(type(node), 4.0)
            if id(node) in highlight_ids:
                base += 4.0
            colors.append(base)
        return np_mod.asarray(colors, dtype=float)

    def _default_positions(self, np_mod):
        levels = {
            Input: 0.0,
            Group: 1.0,
            Projection: 2.0,
            Output: 3.0,
        }
        y = np_mod.asarray([levels.get(type(node), 4.0) for node in self._nodes], dtype=float)
        if y.size:
            max_level = max(y.max(), 1.0)
            y = 1.0 - y / max_level
        x = np_mod.linspace(0.0, 1.0, len(self._nodes), endpoint=True)
        return np_mod.column_stack((x, y))

    def _ensure_node(self, node: GraphElem) -> int:
        node_id = id(node)
        existing = self._id_to_index.get(node_id)
        if existing is not None:
            return existing
        index = len(self._nodes)
        self._nodes.append(node)
        self._id_to_index[node_id] = index
        return index


class CompiledGraph(NamedTuple):
    """Compiled representation of a Spiking Neural Network.
    
    The object captures the decomposed computation graph emitted by the IR
    compiler, organized into type-safe containers for groups, projections,
    inputs, and outputs. It also tracks the execution order that evaluation
    and visualization utilities reuse.

    Parameters
    ----------

    Returns
    -------

    """
    groups: List[Group]
    projections: List[Projection]
    inputs: List[Input]
    outputs: List[Output]
    graph: Graph

    def display(self, mode='text'):
        """Render the compiled graph with the requested presentation.

        Parameters
        ----------
        mode :
            Display mode to use. Supported values are ``'text'`` for
            the verbose textual summary, ``'ascii'`` for the lightweight
            ASCII-art visualization, and ``'visualization'``/``'viz'`` for
            a Graphviz ``Digraph`` (if the optional dependency is present). (Default value = 'text')

        Returns
        -------
        graphviz.Digraph | None
            A Digraph instance for ``visualization``
        graphviz.Digraph | None
            A Digraph instance for ``visualization``
            mode, or ``None`` for purely textual displays that print to stdout.

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
        """Print a detailed, sectioned summary of the compiled graph.
        
        The output enumerates every group, projection, input, and output along
        with the derived metadata (state counts, primitive names, etc.).

        Parameters
        ----------

        Returns
        -------

        """
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
        """Placeholder for a future Graphviz-based visualization."""
        raise NotImplementedError("Visualization display is not implemented.")

    def _ascii_display(self):
        """Render a compact ASCII-art visualization of the compiled graph.
        
        The output includes a legend, execution order, and a simplified data
        flow diagram that can be consumed directly in a terminal.

        Parameters
        ----------

        Returns
        -------

        """
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
                print(f"    | ({len(proj.pre_group.hidden_states)} states, {len(proj.pre_group.jaxpr.jaxpr.eqns)} eqns)")
                drawn_groups.add(id(proj.pre_group))

            # Show projection
            conn_info = f"{len(proj.connections)} conn" if proj.connections else "no conn"
            print(f"    +---spikes---> ({proj_id}) {proj_name}")
            print(f"    |              ({conn_info})")
            print(f"    +---currents-->")
            print()

            if id(proj.post_group) not in drawn_groups:
                print(f"  [{post_id}] {post_name}")
                print(f"    | ({len(proj.post_group.hidden_states)} states, {len(proj.post_group.jaxpr.jaxpr.eqns)} eqns)")
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
    """

    Parameters
    ----------
    state: State :
        

    Returns
    -------
    type
        Shapes are preferred when present to avoid spamming huge tensors, while
        scalars fall back to their raw value.

    """
    if hasattr(state, 'value'):
        val = state.value
        if hasattr(val, 'shape'):
            shape_str = f"shape={val.shape}"
        else:
            shape_str = f"value={val}"
        return f"{type(state).__name__}({shape_str})"
    return f"{type(state).__name__}()"
