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

"""Advanced visualization backends for Graph IR."""

from collections import defaultdict, deque
from typing import Dict, Tuple, List, Set, Any

import numpy as np

from ._data import NeuroGraph, GraphElem, GroupPrim, ProjectionPrim, InputPrim, OutputPrim, UnknownPrim

__all__ = [
    'GraphDisplayer',
    'TextDisplayer',
]


class GraphDisplayer:
    """Provides multiple visualization backends for Graph objects."""

    def __init__(self, graph: NeuroGraph):
        """Initialize visualizer with a graph instance.

        Parameters
        ----------
        graph : NeuroGraph
            The graph to visualize.
        """
        self.graph = graph
        self._node_positions: Dict[GraphElem, Tuple[float, float]] = {}
        self._highlighted_node_ids: Set[int] = set()  # Store node IDs instead of nodes
        self._fig = None
        self._ax = None

    def _compute_hierarchical_layers(self) -> Dict[GraphElem, int]:
        """Compute hierarchical layers for nodes using topological ordering.

        Handles self-connections by treating them as back-edges that don't affect layer assignment.

        Returns
        -------
        Dict[GraphElem, int]
            Mapping from node to its layer index.
        """
        # Identify self-edges (back-edges) to exclude from layer computation
        self_edges = {(source, target) for source, target in self.graph.edges() if source == target}

        # Compute in-degree for each node, excluding self-edges
        in_degree = {}
        for node in self.graph.nodes():
            # Count predecessors that are not the node itself
            non_self_preds = [pred for pred in self.graph.predecessors(node) if pred != node]
            in_degree[node] = len(non_self_preds)

        # Initialize queue with nodes having zero in-degree
        queue = deque([node for node in self.graph.nodes() if in_degree[node] == 0])

        # Layer assignment
        layers: Dict[GraphElem, int] = {}

        while queue:
            node = queue.popleft()

            # Compute layer as max of predecessors' layers + 1, excluding self-references
            pred_layers = [layers[pred] for pred in self.graph.predecessors(node)
                           if pred in layers and pred != node]
            current_layer = max(pred_layers, default=-1) + 1
            layers[node] = current_layer

            # Update successors (excluding self-loops)
            for succ in self.graph.successors(node):
                if succ != node:  # Skip self-edges
                    in_degree[succ] -= 1
                    if in_degree[succ] == 0:
                        queue.append(succ)

        # Handle any remaining nodes that weren't processed (due to cycles other than self-loops)
        # Place them in the layer based on their processed predecessors
        unprocessed = [node for node in self.graph.nodes() if node not in layers]
        if unprocessed:
            for node in unprocessed:
                pred_layers = [layers[pred] for pred in self.graph.predecessors(node)
                               if pred in layers and pred != node]
                current_layer = max(pred_layers, default=0) + 1
                layers[node] = current_layer

        return layers

    def _layout_hierarchical_lr(self) -> Dict[GraphElem, Tuple[float, float]]:
        """Compute left-to-right hierarchical layout.

        Returns
        -------
        Dict[GraphElem, Tuple[float, float]]
            Mapping from node to (x, y) position.
        """
        layers = self._compute_hierarchical_layers()

        # GroupPrim nodes by layer
        layer_nodes: Dict[int, List[GraphElem]] = defaultdict(list)
        for node, layer in layers.items():
            layer_nodes[layer].append(node)

        positions = {}
        x_spacing = 2.0
        y_spacing = 1.5

        for layer_idx, nodes in layer_nodes.items():
            x = layer_idx * x_spacing
            num_nodes = len(nodes)

            # Sort nodes for consistent positioning (InputPrim, GroupPrim, ProjectionPrim, OutputPrim)
            def node_sort_key(n):
                if isinstance(n, InputPrim):
                    return (0, n.name)
                elif isinstance(n, GroupPrim):
                    return (1, n.name)
                elif isinstance(n, ProjectionPrim):
                    return (2, n.name)
                else:  # OutputPrim or UnknownPrim
                    return (3, n.name)

            sorted_nodes = sorted(nodes, key=node_sort_key)

            for i, node in enumerate(sorted_nodes):
                y = (i - num_nodes / 2.0) * y_spacing
                positions[node] = (x, y)

        return positions

    def _layout_hierarchical_tb(self) -> Dict[GraphElem, Tuple[float, float]]:
        """Compute top-to-bottom hierarchical layout.

        Returns
        -------
        Dict[GraphElem, Tuple[float, float]]
            Mapping from node to (x, y) position.
        """
        # Reuse left-right layout but swap x and y, and negate y
        lr_positions = self._layout_hierarchical_lr()
        return {node: (y, -x) for node, (x, y) in lr_positions.items()}

    def _layout_force_directed(self, iterations: int = 100) -> Dict[GraphElem, Tuple]:
        """Compute force-directed layout using simplified spring algorithm.

        Parameters
        ----------
        iterations : int
            Number of iterations for force-directed algorithm.

        Returns
        -------
        Dict[GraphElem, Tuple[float, float]]
            Mapping from node to (x, y) position.
        """
        # Start with hierarchical layout as initial positions
        positions = self._layout_hierarchical_lr()

        nodes = list(self.graph.nodes())
        node_to_idx = {node: i for i, node in enumerate(nodes)}

        # Convert to numpy arrays for efficient computation
        pos_array = np.array([positions[node] for node in nodes], dtype=float)

        # Parameters
        k = 1.0  # Optimal distance
        c_spring = 0.1  # Spring constant
        c_repel = 0.5  # Repulsion constant
        damping = 0.9

        for iteration in range(iterations):
            forces = np.zeros_like(pos_array)

            # Repulsive forces between all pairs
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    delta = pos_array[i] - pos_array[j]
                    dist = np.linalg.norm(delta)
                    if dist > 0:
                        force = c_repel * k * k / (dist * dist) * (delta / dist)
                        forces[i] += force
                        forces[j] -= force

            # Attractive forces for edges
            for source, target in self.graph.edges():
                i = node_to_idx[source]
                j = node_to_idx[target]
                delta = pos_array[j] - pos_array[i]
                dist = np.linalg.norm(delta)
                if dist > 0:
                    force = c_spring * (dist - k) * (delta / dist)
                    forces[i] += force
                    forces[j] -= force

            # Update positions
            pos_array += forces * damping
            damping *= 0.99

        # Convert back to dictionary
        return {node: tuple(pos_array[i]) for i, node in enumerate(nodes)}

    def _get_node_style(self, node: GraphElem) -> Dict[str, Any]:
        """Get visual style for a node based on its type.

        Parameters
        ----------
        node : GraphElem
            The node to style.

        Returns
        -------
        Dict[str, Any]
            Style dictionary with keys: shape, color, size, edge_color, edge_width.
        """
        is_highlighted = id(node) in self._highlighted_node_ids

        if isinstance(node, GroupPrim):
            return {
                'shape': 'circle',
                'color': '#3498db' if not is_highlighted else '#e74c3c',
                'size': 1200,
                'edge_color': '#2c3e50',
                'edge_width': 3 if is_highlighted else 2,
                'alpha': 1.0 if is_highlighted else 0.9,
            }
        elif isinstance(node, InputPrim):
            return {
                'shape': 'roundbox',
                'color': '#2ecc71' if not is_highlighted else '#e74c3c',
                'size': 600,
                'edge_color': '#27ae60',
                'edge_width': 2 if is_highlighted else 1,
                'alpha': 1.0 if is_highlighted else 0.7,
            }
        elif isinstance(node, OutputPrim):
            return {
                'shape': 'roundbox',
                'color': '#f39c12' if not is_highlighted else '#e74c3c',
                'size': 600,
                'edge_color': '#e67e22',
                'edge_width': 2 if is_highlighted else 1,
                'alpha': 1.0 if is_highlighted else 0.7,
            }
        elif isinstance(node, ProjectionPrim):
            # ProjectionPrim nodes are shown as small diamonds on edges
            return {
                'shape': 'diamond',
                'color': '#9b59b6' if not is_highlighted else '#e74c3c',
                'size': 300,
                'edge_color': '#8e44ad',
                'edge_width': 2 if is_highlighted else 1,
                'alpha': 1.0 if is_highlighted else 0.8,
            }
        elif isinstance(node, UnknownPrim):
            # UnknownPrim nodes shown as gray squares
            return {
                'shape': 'square',
                'color': '#95a5a6' if not is_highlighted else '#e74c3c',
                'size': 500,
                'edge_color': '#7f8c8d',
                'edge_width': 2 if is_highlighted else 1,
                'alpha': 1.0 if is_highlighted else 0.7,
            }
        else:
            return {
                'shape': 'circle',
                'color': '#bdc3c7',
                'size': 400,
                'edge_color': '#95a5a6',
                'edge_width': 1,
                'alpha': 0.6,
            }

    def _get_node_label(self, node: GraphElem) -> str:
        """Get label text for a node.

        Parameters
        ----------
        node : GraphElem
            The node to label.

        Returns
        -------
        str
            Label text.
        """
        if isinstance(node, GroupPrim):
            # Show group name with number of hidden states (neurons)
            num_hidden = len(node.hidden_states) if hasattr(node, 'hidden_states') else 0
            return f"{node.name}\n#{num_hidden}"
        elif isinstance(node, InputPrim):
            # Count number of input variables from jaxpr
            num_inputs = len(node.jaxpr.jaxpr.invars) if hasattr(node, 'jaxpr') else 0
            return f"{node.name}\n#{num_inputs}"
        elif isinstance(node, OutputPrim):
            # Count number of outputs (from jaxpr outvars)
            num_outputs = len(node.jaxpr.jaxpr.outvars) if hasattr(node, 'jaxpr') else 0
            return f"{node.name}\n#{num_outputs}"
        elif isinstance(node, ProjectionPrim):
            # Count connections
            num_conns = len(node.connections) if hasattr(node, 'connections') else 0
            return f"{node.name}\n#{num_conns}"
        elif isinstance(node, UnknownPrim):
            # Show unknown block with equation count
            num_eqns = len(node.jaxpr.jaxpr.eqns) if hasattr(node, 'jaxpr') else 0
            return f"{node.name}\n#{num_eqns}"
        else:
            return node.name if hasattr(node, 'name') else str(type(node).__name__)

    def _draw_node(self, ax, node: GraphElem, pos: Tuple[float, float], style: Dict[str, Any]):
        """Draw a single node on the axes.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to draw on.
        node : GraphElem
            The node to draw.
        pos : Tuple[float, float]
            The (x, y) position.
        style : Dict[str, Any]
            Visual style dictionary.
        """
        import matplotlib.patches as mpatches

        x, y = pos
        shape = style['shape']
        size = style['size']
        radius = np.sqrt(size / np.pi) * 0.01  # Scale size to radius

        if shape == 'circle':
            patch = mpatches.Circle(
                (x, y), radius,
                facecolor=style['color'],
                edgecolor=style['edge_color'],
                linewidth=style['edge_width'],
                alpha=style['alpha'],
                picker=True,
                zorder=2
            )
        elif shape == 'roundbox':
            patch = mpatches.FancyBboxPatch(
                (x - radius, y - radius * 0.6),
                radius * 2,
                radius * 1.2,
                boxstyle="round,pad=0.05",
                facecolor=style['color'],
                edgecolor=style['edge_color'],
                linewidth=style['edge_width'],
                alpha=style['alpha'],
                picker=True,
                zorder=2
            )
        elif shape == 'diamond':
            # Diamond shape using polygon
            points = np.array([
                [x, y + radius],
                [x + radius, y],
                [x, y - radius],
                [x - radius, y]
            ])
            patch = mpatches.Polygon(
                points,
                facecolor=style['color'],
                edgecolor=style['edge_color'],
                linewidth=style['edge_width'],
                alpha=style['alpha'],
                picker=True,
                zorder=2
            )
        else:
            # Default to circle
            patch = mpatches.Circle(
                (x, y),
                radius,
                facecolor=style['color'],
                edgecolor=style['edge_color'],
                linewidth=style['edge_width'],
                alpha=style['alpha'],
                picker=True,
                zorder=2
            )

        patch.set_gid(str(id(node)))  # Store node ID for click handling
        ax.add_patch(patch)

        # Add label
        label = self._get_node_label(node)
        fontsize = 10 if isinstance(node, GroupPrim) else 8
        fontweight = 'bold' if isinstance(node, GroupPrim) else 'normal'
        ax.text(
            x, y, label,
            ha='center',
            va='center',
            fontsize=fontsize,
            fontweight=fontweight,
            color='white' if isinstance(node, GroupPrim) else 'black'
        )

    def _draw_edge(
        self, ax, source: GraphElem, target: GraphElem,
        source_pos: Tuple[float, float], target_pos: Tuple[float, float],
        is_projection: bool = False
    ):
        """Draw an edge between two nodes.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to draw on.
        source : GraphElem
            Source node.
        target : GraphElem
            Target node.
        source_pos : Tuple[float, float]
            Source position.
        target_pos : Tuple[float, float]
            Target position.
        is_projection : bool
            Whether this edge represents a ProjectionPrim connection.
        """
        import matplotlib.patches as mpatches

        is_highlighted = id(source) in self._highlighted_node_ids or id(target) in self._highlighted_node_ids

        # Check if this is a self-loop
        is_self_loop = (source == target)

        if is_projection:
            # Solid thick arrow for ProjectionPrim - more prominent for data flow
            color = '#e74c3c' if is_highlighted else '#9b59b6'
            linewidth = 3.5 if is_highlighted else 2.5
            linestyle = '-'
            alpha = 1.0 if is_highlighted else 0.85
            # Larger arrow head for projection connections
            arrowstyle = '->,head_width=0.6,head_length=1.0'
        else:
            # Dashed arrow for InputPrim/OutputPrim connections - still visible but distinct
            color = '#e74c3c' if is_highlighted else '#7f8c8d'
            linewidth = 2.5 if is_highlighted else 2.0
            linestyle = '--'
            alpha = 1.0 if is_highlighted else 0.7
            # Medium arrow head for input/output connections
            arrowstyle = '->,head_width=0.5,head_length=0.9'

        if is_self_loop:
            # Draw self-loop as a circular arc above the node
            x, y = source_pos
            # Get node style to determine size
            style = self._get_node_style(source)
            radius = np.sqrt(style['size'] / np.pi) * 0.01

            # Create a curved path that loops back to the same node
            # Position the loop above and to the right of the node
            loop_radius = radius * 1.5
            loop_offset_x = radius * 0.3
            loop_offset_y = radius * 1.8

            # Start and end points on the edge of the node
            start_angle = 45  # degrees
            end_angle = 135  # degrees
            start_x = x + radius * np.cos(np.radians(start_angle)) + loop_offset_x
            start_y = y + radius * np.sin(np.radians(start_angle))
            end_x = x + radius * np.cos(np.radians(end_angle)) + loop_offset_x
            end_y = y + radius * np.sin(np.radians(end_angle))

            # Draw the self-loop with a large arc
            arrow = mpatches.FancyArrowPatch(
                (start_x, start_y), (end_x, end_y),
                arrowstyle=arrowstyle,
                connectionstyle=f'arc3,rad=1.5',  # Large arc for visibility
                color=color,
                linewidth=linewidth,
                linestyle=linestyle,
                alpha=alpha,
                zorder=3
            )
        else:
            # Draw regular curved arrow with prominent head
            arrow = mpatches.FancyArrowPatch(
                source_pos, target_pos,
                arrowstyle=arrowstyle,
                connectionstyle='arc3,rad=0.1',
                color=color,
                linewidth=linewidth,
                linestyle=linestyle,
                alpha=alpha,
                zorder=3  # Draw arrows on top of nodes
            )
        ax.add_patch(arrow)

    def _on_click(self, event):
        """Handle click events on nodes for highlighting.

        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
            The click event.
        """
        if event.inaxes != self._ax:
            return

        # Find clicked node
        clicked_node = None
        for artist in self._ax.patches:
            if artist.contains(event)[0]:
                gid = artist.get_gid()
                if gid:
                    node_id = int(gid)
                    for node in self.graph.nodes():
                        if id(node) == node_id:
                            clicked_node = node
                            break
                break

        if clicked_node is None:
            # Clear highlights
            if self._highlighted_node_ids:
                self._highlighted_node_ids.clear()
                self._redraw()
        else:
            # Toggle highlight
            if id(clicked_node) in self._highlighted_node_ids:
                self._highlighted_node_ids.clear()
            else:
                # Highlight clicked node and its neighbors
                self._highlighted_node_ids.clear()
                self._highlighted_node_ids.add(id(clicked_node))
                # Add IDs of predecessors and successors
                for pred in self.graph.predecessors(clicked_node):
                    self._highlighted_node_ids.add(id(pred))
                for succ in self.graph.successors(clicked_node):
                    self._highlighted_node_ids.add(id(succ))

            self._redraw()

    def _redraw(self):
        """Redraw the graph with current highlight state."""
        if self._ax is None or self._fig is None:
            return

        self._ax.clear()
        self._draw_graph_elements()
        self._fig.canvas.draw()

    def _draw_graph_elements(self):
        """Draw all graph elements (nodes and edges) on the current axes."""
        # Draw edges first (so they appear behind nodes)
        projection_edges = set()

        # Identify which edges connect to Projections
        for node in self.graph.nodes():
            if isinstance(node, ProjectionPrim):
                # Edges from pre_group to projection and projection to post_group
                if hasattr(node, 'pre_group') and hasattr(node, 'post_group'):
                    projection_edges.add((node.pre_group, node))
                    projection_edges.add((node, node.post_group))

        for source, target in self.graph.edges():
            is_proj = (source, target) in projection_edges
            self._draw_edge(self._ax,
                            source,
                            target,
                            self._node_positions[source],
                            self._node_positions[target],
                            is_projection=is_proj)

        # Draw nodes
        for node in self.graph.nodes():
            style = self._get_node_style(node)
            self._draw_node(self._ax, node, self._node_positions[node], style)

        # Set axis properties
        self._ax.set_aspect('equal')
        self._ax.axis('off')

        # Set appropriate limits with padding
        if self._node_positions:
            positions = list(self._node_positions.values())
            xs, ys = zip(*positions)
            margin = 1.0
            self._ax.set_xlim(min(xs) - margin, max(xs) + margin)
            self._ax.set_ylim(min(ys) - margin, max(ys) + margin)

    def display(self, layout: str = 'auto', figsize: Tuple[float, float] = (12, 8), **kwargs):
        """Display the graph using matplotlib.

        Parameters
        ----------
        layout : str
            Layout algorithm to use:
            - 'lr' or 'left-right': Left-to-right hierarchical layout
            - 'tb' or 'top-bottom': Top-to-bottom hierarchical layout
            - 'auto' or 'force': Force-directed layout
        figsize : Tuple[float, float]
            Figure size (width, height) in inches.
        **kwargs
            Additional arguments passed to layout algorithm.

        Returns
        -------
        matplotlib.figure.Figure
            The created figure.
        """
        import matplotlib.pyplot as plt

        # Compute layout
        if layout in ('lr', 'left-right'):
            self._node_positions = self._layout_hierarchical_lr()
        elif layout in ('tb', 'top-bottom'):
            self._node_positions = self._layout_hierarchical_tb()
        elif layout in ('auto', 'force'):
            iterations = kwargs.get('iterations', 100)
            self._node_positions = self._layout_force_directed(iterations=iterations)
        else:
            raise ValueError(f"UnknownPrim layout: {layout}. Use 'lr', 'tb', or 'auto'.")

        # Create figure
        self._fig, self._ax = plt.subplots(figsize=figsize)

        # Draw graph
        self._draw_graph_elements()

        # Connect click handler
        self._fig.canvas.mpl_connect('button_press_event', self._on_click)

        # Add title and legend
        self._ax.set_title(
            'NeuroGraph Visualization\n(Click nodes to highlight connections)',
            fontsize=14,
            fontweight='bold',
            pad=20
        )

        # Create legend
        import matplotlib.patches as mpatches
        import matplotlib.lines as mlines
        legend_elements = [
            mpatches.Patch(facecolor='#3498db', edgecolor='#2c3e50', label='GroupPrim (Neurons)'),
            mpatches.Patch(facecolor='#2ecc71', edgecolor='#27ae60', label='InputPrim'),
            mpatches.Patch(facecolor='#f39c12', edgecolor='#e67e22', label='OutputPrim'),
            mpatches.Patch(facecolor='#9b59b6', edgecolor='#8e44ad', label='ProjectionPrim'),
            mlines.Line2D(
                [], [], color='#9b59b6', marker='>', markersize=8,
                linestyle='-', linewidth=2.5, label='Data Flow (ProjectionPrim)'
            ),
            mlines.Line2D(
                [], [], color='#7f8c8d', marker='>', markersize=7,
                linestyle='--', linewidth=2.0, label='Data Flow (InputPrim/OutputPrim)'
            ),
        ]
        self._ax.legend(
            handles=legend_elements,
            loc='upper left',
            bbox_to_anchor=(1.02, 1),
            fontsize=9
        )

        plt.tight_layout()

        return self._fig


class TextDisplayer:
    """Text-based visualization for NeuroGraph structures."""

    def __init__(self, graph: NeuroGraph):
        """Initialize the text displayer.

        Parameters
        ----------
        graph : NeuroGraph
            The graph to visualize.
        """
        self.graph = graph

    def display(self, verbose: bool = False, show_jaxpr: bool = False) -> str:
        """Generate a text-based visualization of the graph.

        Parameters
        ----------
        verbose : bool
            If True, show detailed node information. Default: False.
        show_jaxpr : bool
            If True, show jaxpr equations for each node. Default: False.

        Returns
        -------
        str
            Formatted text representation of the graph.
        """
        sections = []

        # 1. Summary section
        sections.append(self._build_summary())

        # 2. Nodes section
        sections.append(self._build_nodes_section(verbose, show_jaxpr))

        # 3. Dependencies section
        sections.append(self._build_edges_section())

        return '\n\n'.join(sections)

    def _get_node_type_counts(self) -> Dict[str, int]:
        """Count nodes by type.

        Returns
        -------
        Dict[str, int]
            Mapping from node type name to count.
        """
        counts = {'GroupPrim': 0, 'ProjectionPrim': 0, 'InputPrim': 0, 'OutputPrim': 0, 'UnknownPrim': 0, 'Other': 0}
        for node in self.graph.nodes():
            if isinstance(node, GroupPrim):
                counts['GroupPrim'] += 1
            elif isinstance(node, ProjectionPrim):
                counts['ProjectionPrim'] += 1
            elif isinstance(node, InputPrim):
                counts['InputPrim'] += 1
            elif isinstance(node, OutputPrim):
                counts['OutputPrim'] += 1
            elif isinstance(node, UnknownPrim):
                counts['UnknownPrim'] += 1
            else:
                counts['Other'] += 1
        return counts

    def _build_summary(self) -> str:
        """Build the summary section showing graph statistics.

        Returns
        -------
        str
            Formatted summary string.
        """
        num_nodes = len(self.graph)
        num_edges = self.graph.edge_count()
        type_counts = self._get_node_type_counts()

        # Build type summary
        type_parts = []
        if type_counts['GroupPrim'] > 0:
            type_parts.append(f"{type_counts['GroupPrim']} GroupPrim{'s' if type_counts['GroupPrim'] > 1 else ''}")
        if type_counts['ProjectionPrim'] > 0:
            type_parts.append(f"{type_counts['ProjectionPrim']} ProjectionPrim{'s' if type_counts['ProjectionPrim'] > 1 else ''}")
        if type_counts['InputPrim'] > 0:
            type_parts.append(f"{type_counts['InputPrim']} InputPrim{'s' if type_counts['InputPrim'] > 1 else ''}")
        if type_counts['OutputPrim'] > 0:
            type_parts.append(f"{type_counts['OutputPrim']} OutputPrim{'s' if type_counts['OutputPrim'] > 1 else ''}")
        if type_counts['Other'] > 0:
            type_parts.append(f"{type_counts['Other']} Other")

        type_summary = ', '.join(type_parts) if type_parts else 'empty'

        return f"NeuroGraph Summary:\n  Nodes: {num_nodes} ({type_summary})\n  Edges: {num_edges}"

    def _build_nodes_section(self, verbose: bool, show_jaxpr: bool) -> str:
        """Build the nodes section showing each node with details.

        Parameters
        ----------
        verbose : bool
            Show detailed node information.
        show_jaxpr : bool
            Show jaxpr equations.

        Returns
        -------
        str
            Formatted nodes section.
        """
        lines = ["Nodes (execution order):"]

        nodes = self.graph.nodes()
        for idx, node in enumerate(nodes):
            node_str = self._format_node(node, idx, verbose, show_jaxpr)
            lines.append(node_str)

            # Show successors with tree structure
            successors = self.graph.successors(node)
            if successors:
                for i, succ in enumerate(successors):
                    # Check if this is a self-connection
                    if succ == node:
                        # Self-connection indicator
                        if i == len(successors) - 1:
                            lines.append(f"      └─> [self] (self-connection)")
                        else:
                            lines.append(f"      ├─> [self] (self-connection)")
                    else:
                        # Find successor index
                        succ_idx = nodes.index(succ) if succ in nodes else -1
                        if i == len(successors) - 1:
                            lines.append(f"      └─> [{succ_idx}] {self._get_node_short_name(succ)}")
                        else:
                            lines.append(f"      ├─> [{succ_idx}] {self._get_node_short_name(succ)}")

        return '\n'.join(lines)

    def _format_node(self, node: GraphElem, index: int, verbose: bool, show_jaxpr: bool) -> str:
        """Format a single node for display.

        Parameters
        ----------
        node : GraphElem
            The node to format.
        index : int
            The node's index in execution order.
        verbose : bool
            Show detailed information.
        show_jaxpr : bool
            Show jaxpr equations.

        Returns
        -------
        str
            Formatted node string.
        """
        lines = []

        # Main node line
        if isinstance(node, GroupPrim):
            num_hidden = len(node.hidden_states)
            main_line = f"  [{index}] {node.name} #{num_hidden} states"
            if verbose:
                num_in = len(node.in_states)
                num_out = len(node.out_states)
                num_eqns = len(node.jaxpr.jaxpr.eqns)
                lines.append(main_line)
                lines.append(f"      Hidden States: {num_hidden}, In States: {num_in}, Out States: {num_out}")
                lines.append(f"      Equations: {num_eqns}")
            else:
                lines.append(main_line)

        elif isinstance(node, ProjectionPrim):
            pre_name = node.pre_group.name
            post_name = node.post_group.name
            num_conns = len(node.connections)
            main_line = f"  [{index}] {node.name} ({pre_name} → {post_name}) #{num_conns} connections"
            if verbose:
                num_hidden = len(node.hidden_states)
                num_in = len(node.in_states)
                num_eqns = len(node.jaxpr.jaxpr.eqns)
                lines.append(main_line)
                lines.append(f"      Hidden States: {num_hidden}, In States: {num_in}")
                lines.append(f"      Equations: {num_eqns}")
            else:
                lines.append(main_line)

        elif isinstance(node, InputPrim):
            group_name = node.group.name
            num_invars = len(node.jaxpr.jaxpr.invars)
            main_line = f"  [{index}] {node.name} to {group_name} (#{num_invars} vars)"
            if verbose:
                num_outvars = len(node.jaxpr.jaxpr.outvars)
                num_eqns = len(node.jaxpr.jaxpr.eqns)
                lines.append(main_line)
                lines.append(f"      In Vars: {num_invars}, Out Vars: {num_outvars}, Equations: {num_eqns}")
            else:
                lines.append(main_line)

        elif isinstance(node, OutputPrim):
            group_name = node.group.name
            num_outvars = len(node.jaxpr.jaxpr.outvars)
            main_line = f"  [{index}] {node.name} from {group_name} (#{num_outvars} vars)"
            if verbose:
                num_hidden = len(node.hidden_states)
                num_in = len(node.in_states)
                num_eqns = len(node.jaxpr.jaxpr.eqns)
                lines.append(main_line)
                lines.append(f"      Hidden States: {num_hidden}, In States: {num_in}, Equations: {num_eqns}")
            else:
                lines.append(main_line)

        elif isinstance(node, UnknownPrim):
            num_eqns = len(node.jaxpr.jaxpr.eqns)
            main_line = f"  [{index}] {node.name} #{num_eqns} equations"
            if verbose:
                num_invars = len(node.jaxpr.jaxpr.invars)
                num_outvars = len(node.jaxpr.jaxpr.outvars)
                indices_str = f"{node.eqn_indices[0]}..{node.eqn_indices[-1]}" if len(node.eqn_indices) > 1 else str(node.eqn_indices[0])
                lines.append(main_line)
                lines.append(f"      In Vars: {num_invars}, Out Vars: {num_outvars}")
                lines.append(f"      Original Indices: {indices_str}")
            else:
                lines.append(main_line)

        else:
            lines.append(f"  [{index}] {node.name}")

        # Add jaxpr details if requested
        if show_jaxpr:
            jaxpr_str = self._format_jaxpr(node)
            if jaxpr_str:
                lines.append(jaxpr_str)

        return '\n'.join(lines)

    def _get_node_short_name(self, node: GraphElem) -> str:
        """Get a short name for a node for use in tree display.

        Parameters
        ----------
        node : GraphElem
            The node.

        Returns
        -------
        str
            Short name string.
        """
        if isinstance(node, GroupPrim):
            return node.name
        elif isinstance(node, ProjectionPrim):
            return node.name
        elif isinstance(node, InputPrim):
            group_name = node.group.name
            return f"{node.name} to {group_name}"
        elif isinstance(node, OutputPrim):
            group_name = node.group.name
            return f"{node.name} from {group_name}"
        elif isinstance(node, UnknownPrim):
            return node.name
        else:
            return node.name if hasattr(node, 'name') else type(node).__name__

    def _format_jaxpr(self, node: GraphElem) -> str:
        """Format jaxpr equations for a node.

        Parameters
        ----------
        node : GraphElem
            The node.

        Returns
        -------
        str
            Formatted jaxpr string, or empty if no equations.
        """
        if not node.jaxpr:
            return ""

        eqns = node.jaxpr.jaxpr.eqns
        if not eqns:
            return ""

        lines = ["      JAXPR Equations:"]
        for i, eqn in enumerate(eqns[:10]):  # Limit to first 10 equations
            # Format equation as: outvars = primitive invars
            outvars_str = ', '.join(str(v) for v in eqn.outvars)
            invars_str = ', '.join(str(v) for v in eqn.invars)
            lines.append(f"        [{i}] {outvars_str} = {eqn.primitive.name}({invars_str})")

        if len(eqns) > 10:
            lines.append(f"        ... ({len(eqns) - 10} more equations)")

        return '\n'.join(lines)

    def _build_edges_section(self) -> str:
        """Build the edges/dependencies section.

        Returns
        -------
        str
            Formatted edges section.
        """
        lines = [f"Dependencies ({self.graph.edge_count()} edges):"]

        edges = list(self.graph.edges())
        if not edges:
            lines.append("  (no edges)")
            return '\n'.join(lines)

        for source, target in edges:
            source_name = self._get_node_short_name(source)
            target_name = self._get_node_short_name(target)
            # Mark self-connections clearly
            if source == target:
                lines.append(f"  {source_name} → self (self-connection)")
            else:
                lines.append(f"  {source_name} → {target_name}")

        return '\n'.join(lines)
