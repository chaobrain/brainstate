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
from typing import Dict, Tuple, List, Set, Optional, Any
import numpy as np

from ._data import NeuroGraph, GraphElem, Group, Projection, Input, Output, Connection

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
        self._highlighted_nodes: Set[GraphElem] = set()
        self._fig = None
        self._ax = None

    def _compute_hierarchical_layers(self) -> Dict[GraphElem, int]:
        """Compute hierarchical layers for nodes using topological ordering.

        Returns
        -------
        Dict[GraphElem, int]
            Mapping from node to its layer index.
        """
        # Compute in-degree for each node
        in_degree = {node: len(self.graph.predecessors(node)) for node in self.graph.nodes()}

        # Initialize queue with nodes having zero in-degree
        queue = deque([node for node in self.graph.nodes() if in_degree[node] == 0])

        # Layer assignment
        layers: Dict[GraphElem, int] = {}

        while queue:
            node = queue.popleft()

            # Compute layer as max of predecessors' layers + 1
            pred_layers = [layers[pred] for pred in self.graph.predecessors(node) if pred in layers]
            current_layer = max(pred_layers, default=-1) + 1
            layers[node] = current_layer

            # Update successors
            for succ in self.graph.successors(node):
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    queue.append(succ)

        return layers

    def _layout_hierarchical_lr(self) -> Dict[GraphElem, Tuple[float, float]]:
        """Compute left-to-right hierarchical layout.

        Returns
        -------
        Dict[GraphElem, Tuple[float, float]]
            Mapping from node to (x, y) position.
        """
        layers = self._compute_hierarchical_layers()

        # Group nodes by layer
        layer_nodes: Dict[int, List[GraphElem]] = defaultdict(list)
        for node, layer in layers.items():
            layer_nodes[layer].append(node)

        positions = {}
        x_spacing = 2.0
        y_spacing = 1.5

        for layer_idx, nodes in layer_nodes.items():
            x = layer_idx * x_spacing
            num_nodes = len(nodes)

            # Sort nodes for consistent positioning (Input, Group, Projection, Output)
            def node_sort_key(n):
                if isinstance(n, Input):
                    return (0, n.name if hasattr(n, 'name') else '')
                elif isinstance(n, Group):
                    return (1, n.name)
                elif isinstance(n, Projection):
                    return (2, n.name)
                else:  # Output
                    return (3, n.name if hasattr(n, 'name') else '')

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

    def _layout_force_directed(self, iterations: int = 100) -> Dict[GraphElem, Tuple[float, float]]:
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
        is_highlighted = node in self._highlighted_nodes

        if isinstance(node, Group):
            return {
                'shape': 'circle',
                'color': '#3498db' if not is_highlighted else '#e74c3c',
                'size': 1200,
                'edge_color': '#2c3e50',
                'edge_width': 3 if is_highlighted else 2,
                'alpha': 1.0 if is_highlighted else 0.9,
            }
        elif isinstance(node, Input):
            return {
                'shape': 'roundbox',
                'color': '#2ecc71' if not is_highlighted else '#e74c3c',
                'size': 600,
                'edge_color': '#27ae60',
                'edge_width': 2 if is_highlighted else 1,
                'alpha': 1.0 if is_highlighted else 0.7,
            }
        elif isinstance(node, Output):
            return {
                'shape': 'roundbox',
                'color': '#f39c12' if not is_highlighted else '#e74c3c',
                'size': 600,
                'edge_color': '#e67e22',
                'edge_width': 2 if is_highlighted else 1,
                'alpha': 1.0 if is_highlighted else 0.7,
            }
        elif isinstance(node, Projection):
            # Projection nodes are shown as small diamonds on edges
            return {
                'shape': 'diamond',
                'color': '#9b59b6' if not is_highlighted else '#e74c3c',
                'size': 300,
                'edge_color': '#8e44ad',
                'edge_width': 2 if is_highlighted else 1,
                'alpha': 1.0 if is_highlighted else 0.8,
            }
        else:
            return {
                'shape': 'circle',
                'color': '#95a5a6',
                'size': 400,
                'edge_color': '#7f8c8d',
                'edge_width': 1,
                'alpha': 0.7,
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
        if isinstance(node, Group):
            return node.name
        elif isinstance(node, Input):
            # Count number of input variables
            num_inputs = len(node.input_vars) if hasattr(node, 'input_vars') else 0
            return f"Input\n#{num_inputs}"
        elif isinstance(node, Output):
            # Count number of outputs (from jaxpr outvars)
            num_outputs = len(node.jaxpr.jaxpr.outvars) if hasattr(node, 'jaxpr') else 0
            return f"Output\n#{num_outputs}"
        elif isinstance(node, Projection):
            # Count connections
            num_conns = len(node.connections) if hasattr(node, 'connections') else 0
            return f"{num_conns}"
        else:
            return str(type(node).__name__)

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
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        x, y = pos
        shape = style['shape']
        size = style['size']
        radius = np.sqrt(size / np.pi) * 0.01  # Scale size to radius

        if shape == 'circle':
            patch = mpatches.Circle((x, y), radius,
                                   facecolor=style['color'],
                                   edgecolor=style['edge_color'],
                                   linewidth=style['edge_width'],
                                   alpha=style['alpha'],
                                   picker=True)
        elif shape == 'roundbox':
            patch = mpatches.FancyBboxPatch((x - radius, y - radius * 0.6),
                                           radius * 2, radius * 1.2,
                                           boxstyle="round,pad=0.05",
                                           facecolor=style['color'],
                                           edgecolor=style['edge_color'],
                                           linewidth=style['edge_width'],
                                           alpha=style['alpha'],
                                           picker=True)
        elif shape == 'diamond':
            # Diamond shape using polygon
            points = np.array([
                [x, y + radius],
                [x + radius, y],
                [x, y - radius],
                [x - radius, y]
            ])
            patch = mpatches.Polygon(points,
                                    facecolor=style['color'],
                                    edgecolor=style['edge_color'],
                                    linewidth=style['edge_width'],
                                    alpha=style['alpha'],
                                    picker=True)
        else:
            # Default to circle
            patch = mpatches.Circle((x, y), radius,
                                   facecolor=style['color'],
                                   edgecolor=style['edge_color'],
                                   linewidth=style['edge_width'],
                                   alpha=style['alpha'],
                                   picker=True)

        patch.set_gid(str(id(node)))  # Store node ID for click handling
        ax.add_patch(patch)

        # Add label
        label = self._get_node_label(node)
        fontsize = 10 if isinstance(node, Group) else 8
        fontweight = 'bold' if isinstance(node, Group) else 'normal'
        ax.text(x, y, label,
               ha='center', va='center',
               fontsize=fontsize, fontweight=fontweight,
               color='white' if isinstance(node, Group) else 'black')

    def _draw_edge(self, ax, source: GraphElem, target: GraphElem,
                   source_pos: Tuple[float, float], target_pos: Tuple[float, float],
                   is_projection: bool = False):
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
            Whether this edge represents a Projection connection.
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        is_highlighted = source in self._highlighted_nodes or target in self._highlighted_nodes

        if is_projection:
            # Solid thick arrow for Projection
            color = '#e74c3c' if is_highlighted else '#9b59b6'
            linewidth = 3 if is_highlighted else 2
            linestyle = '-'
            alpha = 1.0 if is_highlighted else 0.7
        else:
            # Dashed thinner arrow for Input/Output connections
            color = '#e74c3c' if is_highlighted else '#95a5a6'
            linewidth = 2 if is_highlighted else 1.5
            linestyle = '--'
            alpha = 1.0 if is_highlighted else 0.5

        # Draw curved arrow
        arrow = mpatches.FancyArrowPatch(
            source_pos, target_pos,
            arrowstyle='->,head_width=0.4,head_length=0.8',
            connectionstyle='arc3,rad=0.1',
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            alpha=alpha,
            zorder=1
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
            if self._highlighted_nodes:
                self._highlighted_nodes.clear()
                self._redraw()
        else:
            # Toggle highlight
            if clicked_node in self._highlighted_nodes:
                self._highlighted_nodes.clear()
            else:
                # Highlight clicked node and its neighbors
                self._highlighted_nodes.clear()
                self._highlighted_nodes.add(clicked_node)
                self._highlighted_nodes.update(self.graph.predecessors(clicked_node))
                self._highlighted_nodes.update(self.graph.successors(clicked_node))

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
            if isinstance(node, Projection):
                # Edges from pre_group to projection and projection to post_group
                if hasattr(node, 'pre_group') and hasattr(node, 'post_group'):
                    projection_edges.add((node.pre_group, node))
                    projection_edges.add((node, node.post_group))

        for source, target in self.graph.edges():
            is_proj = (source, target) in projection_edges
            self._draw_edge(self._ax, source, target,
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
            raise ValueError(f"Unknown layout: {layout}. Use 'lr', 'tb', or 'auto'.")

        # Create figure
        self._fig, self._ax = plt.subplots(figsize=figsize)

        # Draw graph
        self._draw_graph_elements()

        # Connect click handler
        self._fig.canvas.mpl_connect('button_press_event', self._on_click)

        # Add title and legend
        self._ax.set_title('NeuroGraph Visualization\n(Click nodes to highlight connections)',
                          fontsize=14, fontweight='bold', pad=20)

        # Create legend
        import matplotlib.patches as mpatches
        legend_elements = [
            mpatches.Patch(facecolor='#3498db', edgecolor='#2c3e50', label='Group (Neurons)'),
            mpatches.Patch(facecolor='#2ecc71', edgecolor='#27ae60', label='Input'),
            mpatches.Patch(facecolor='#f39c12', edgecolor='#e67e22', label='Output'),
            mpatches.Patch(facecolor='#9b59b6', edgecolor='#8e44ad', label='Projection'),
            mpatches.Patch(facecolor='none', edgecolor='#9b59b6',
                          linestyle='-', linewidth=2, label='Projection Connection'),
            mpatches.Patch(facecolor='none', edgecolor='#95a5a6',
                          linestyle='--', linewidth=1.5, label='Input/Output Connection'),
        ]
        self._ax.legend(handles=legend_elements, loc='upper left',
                       bbox_to_anchor=(1.02, 1), fontsize=9)

        plt.tight_layout()

        return self._fig



class TextDisplayer:
    def __init__(self, graph):
        self.graph = graph
