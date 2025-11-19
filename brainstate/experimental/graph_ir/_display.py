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
from typing import Dict, Tuple

from ._data import Graph, GraphIRElem, GroupIR, ProjectionIR, InputIR, OutputIR, ConnectionIR

__all__ = [
    'GraphDisplayer',
    'TextDisplayer',
]


class GraphDisplayer:
    """Provides multiple visualization backends for Graph objects."""

    def __init__(self, graph: 'Graph'):
        """Initialize visualizer with a graph instance.

        Parameters
        ----------
        graph : Graph
            The graph to visualize.
        """
        self.graph = graph

    def visualize_plotly(
        self,
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
        """Create interactive Plotly visualization.

        Parameters
        ----------
        layout : str
            Layout algorithm ('hierarchical').
        interactive : bool
            Enable interactive features.
        show_details : bool
            Show tooltips with node metadata.
        node_size : str or dict
            Node sizing strategy.
        edge_width : str or dict
            Edge width strategy.
        colorscheme : str
            Color scheme name.
        export_path : str, optional
            Path to export HTML file.
        figsize : tuple or 'auto'
            Figure size.
        **kwargs
            Additional Plotly-specific options.

        Returns
        -------
        plotly.graph_objects.Figure
            Interactive Plotly figure.

        Raises
        ------
        RuntimeError
            If plotly is not installed.
        """
        try:
            import plotly.graph_objects as go
        except ImportError as exc:
            raise RuntimeError(
                "Plotly backend requires plotly to be installed. "
                "Install with: pip install plotly"
            ) from exc

        from ._data import GroupIR, ProjectionIR

        if not self.graph._nodes:
            # Empty graph
            fig = go.Figure()
            fig.add_annotation(
                text="Graph is empty",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=14)
            )
            fig.update_layout(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                showlegend=False,
            )
            return fig

        # Compute layout
        if layout == 'hierarchical':
            x_pos, y_pos, num_layers, max_width = self._compute_hierarchical_layout()
        else:
            raise ValueError(f"Unsupported layout for Plotly: {layout}")

        # Get color scheme
        colors = self._get_color_scheme(colorscheme)

        # Prepare node traces
        node_x = []
        node_y = []
        node_colors = []
        node_text = []
        node_hovertext = []
        node_sizes = []

        for idx, node in enumerate(self.graph._nodes):
            node_x.append(x_pos.get(idx, 0.0))
            node_y.append(y_pos.get(idx, 0.0))

            # Color
            fill_color, _ = self._get_node_style(node, colorscheme)
            node_colors.append(fill_color)

            # Label
            node_text.append(self._get_node_label(node, show_details))

            # Hover text
            if show_details:
                metadata = self._get_node_metadata(node)
                hover_lines = [f"<b>{self._get_node_label(node, False).replace(chr(10), ' ')}</b>"]
                for key, value in metadata.items():
                    if key != 'Type':  # Type is already in label
                        hover_lines.append(f"{key}: {value}")
                node_hovertext.append("<br>".join(hover_lines))
            else:
                node_hovertext.append(self._get_node_label(node, False))

            # Size
            if node_size == 'auto':
                # Size by complexity
                if isinstance(node, (GroupIR, ProjectionIR)):
                    complexity = len(node.jaxpr.jaxpr.eqns) + len(getattr(node, 'hidden_states', []))
                    node_sizes.append(20 + min(complexity * 2, 40))
                else:
                    node_sizes.append(25)
            elif isinstance(node_size, dict):
                node_sizes.append(node_size.get(idx, 25))
            else:
                node_sizes.append(25)

        # Create edge traces
        edge_x = []
        edge_y = []

        for source_idx, targets in self.graph._forward_edges.items():
            sx = x_pos.get(source_idx, 0.0)
            sy = y_pos.get(source_idx, 0.0)
            for target_idx in targets:
                tx = x_pos.get(target_idx, 0.0)
                ty = y_pos.get(target_idx, 0.0)

                # Add arrow line
                edge_x.extend([sx, tx, None])
                edge_y.extend([sy, ty, None])

        # Create edge trace
        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            mode='lines',
            line=dict(width=2 if edge_width == 'auto' else edge_width, color='#546E7A'),
            hoverinfo='none',
            showlegend=False,
        )

        # Create node trace
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=2, color='#263238'),
                symbol='square',
            ),
            text=[label.replace('\\n', '<br>') for label in node_text],
            textposition="middle center",
            textfont=dict(size=9, color='#263238'),
            hovertext=node_hovertext,
            hoverinfo='text',
            showlegend=False,
        )

        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace])

        # Update layout
        if figsize == 'auto':
            width = max(600, max_width * 200)
            height = max(400, num_layers * 150)
        else:
            width, height = figsize

        fig.update_layout(
            title=dict(
                text="Model Dependency Graph",
                x=0.5,
                xanchor='center',
                font=dict(size=16, color='#263238')
            ),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=20, r=20, t=60),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            width=width,
            height=height,
        )

        # Export if requested
        if export_path:
            if export_path.endswith('.html'):
                fig.write_html(export_path)
            elif export_path.endswith('.png'):
                fig.write_image(export_path)
            elif export_path.endswith('.pdf'):
                fig.write_image(export_path)
            elif export_path.endswith('.svg'):
                fig.write_image(export_path)

        return fig

    def visualize_graphviz(
        self,
        layout='dot',
        show_details=True,
        colorscheme='default',
        export_path=None,
        **kwargs
    ):
        """Create Graphviz visualization with professional layouts.

        Parameters
        ----------
        layout : str
            Graphviz layout engine: 'dot', 'neato', 'fdp', 'sfdp', 'circo', 'twopi'.
        show_details : bool
            Include detailed node information.
        colorscheme : str
            Color scheme name.
        export_path : str, optional
            Path to export (format inferred from extension).
        **kwargs
            Additional Graphviz attributes.

        Returns
        -------
        graphviz.Digraph
            Graphviz diagram object.

        Raises
        ------
        RuntimeError
            If graphviz is not installed.
        """
        try:
            from graphviz import Digraph
        except ImportError as exc:
            raise RuntimeError(
                "Graphviz backend requires graphviz to be installed. "
                "Install with: pip install graphviz"
            ) from exc

        # Create digraph with specified engine
        valid_engines = ['dot', 'neato', 'fdp', 'sfdp', 'circo', 'twopi']
        if layout not in valid_engines:
            raise ValueError(f"Invalid Graphviz layout: {layout}. Choose from {valid_engines}")

        dot = Digraph(engine=layout, comment='Model Dependency Graph')
        dot.attr(rankdir='TB', splines='spline')
        dot.attr('node', shape='box', style='rounded,filled', fontname='Arial')
        dot.attr('edge', fontname='Arial')

        # Apply custom attributes from kwargs
        for key, value in kwargs.items():
            dot.attr(key, value)

        colors = self._get_color_scheme(colorscheme)

        # Add nodes
        for idx, node in enumerate(self.graph._nodes):
            node_id = str(idx)
            label = self._get_node_label(node, False).replace('\\n', '\n')

            if show_details:
                metadata = self._get_node_metadata(node)
                detail_lines = [label]
                for key, value in metadata.items():
                    if key != 'Type':
                        detail_lines.append(f"{key}: {value}")
                label = '\n'.join(detail_lines)

            fill_color, edge_color = self._get_node_style(node, colorscheme)

            dot.node(
                node_id,
                label,
                fillcolor=fill_color,
                color=edge_color,
                fontsize='10',
            )

        # Add edges
        for source_idx, targets in self.graph._forward_edges.items():
            for target_idx in targets:
                dot.edge(str(source_idx), str(target_idx), color='#546E7A')

        # Export if requested
        if export_path:
            # Infer format from extension
            if '.' in export_path:
                fmt = export_path.rsplit('.', 1)[1]
                base = export_path.rsplit('.', 1)[0]
                dot.render(base, format=fmt, cleanup=True)
            else:
                dot.render(export_path, cleanup=True)

        return dot

    def visualize_networkx(
        self,
        layout='spring',
        node_size='auto',
        edge_width='auto',
        colorscheme='default',
        figsize='auto',
        show_details=True,
        export_path=None,
        **kwargs
    ):
        """Visualize using NetworkX with advanced layout algorithms.

        Parameters
        ----------
        layout : str
            NetworkX layout: 'spring', 'kamada_kawai', 'spectral', 'circular', 'shell'.
        node_size : str or dict
            Node sizing strategy.
        edge_width : str or dict
            Edge width strategy.
        colorscheme : str
            Color scheme name.
        figsize : tuple or 'auto'
            Figure size.
        show_details : bool
            Show node labels.
        export_path : str, optional
            Path to export image.
        **kwargs
            Additional NetworkX layout parameters.

        Returns
        -------
        matplotlib.figure.Figure
            Matplotlib figure.

        Raises
        ------
        RuntimeError
            If networkx or matplotlib is not installed.
        """
        try:
            import networkx as nx
            import matplotlib.pyplot as plt
            from matplotlib.patches import FancyBboxPatch
        except ImportError as exc:
            raise RuntimeError(
                "NetworkX backend requires networkx and matplotlib. "
                "Install with: pip install networkx matplotlib"
            ) from exc

        from ._data import GroupIR, ProjectionIR

        # Build NetworkX graph
        G = nx.DiGraph()
        for idx in range(len(self.graph._nodes)):
            G.add_node(idx)

        for source_idx, targets in self.graph._forward_edges.items():
            for target_idx in targets:
                G.add_edge(source_idx, target_idx)

        # Compute layout
        layout_funcs = {
            'spring': nx.spring_layout,
            'kamada_kawai': nx.kamada_kawai_layout,
            'spectral': nx.spectral_layout,
            'circular': nx.circular_layout,
            'shell': nx.shell_layout,
        }

        if layout not in layout_funcs:
            raise ValueError(f"Unsupported NetworkX layout: {layout}. Choose from {list(layout_funcs.keys())}")

        pos = layout_funcs[layout](G, **kwargs)

        # Create figure
        if figsize == 'auto':
            num_nodes = len(self.graph._nodes)
            fig_size = max(8, min(num_nodes * 0.8, 20))
            fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        else:
            fig, ax = plt.subplots(figsize=figsize)

        # Prepare node sizes
        if node_size == 'auto':
            sizes = []
            for node in self.graph._nodes:
                if isinstance(node, (GroupIR, ProjectionIR)):
                    complexity = len(node.jaxpr.jaxpr.eqns) + len(getattr(node, 'hidden_states', []))
                    sizes.append(2000 + min(complexity * 100, 3000))
                else:
                    sizes.append(2000)
        elif isinstance(node_size, dict):
            sizes = [node_size.get(i, 2000) for i in range(len(self.graph._nodes))]
        else:
            sizes = 2000

        # Prepare node colors
        colors = []
        for node in self.graph._nodes:
            fill_color, _ = self._get_node_style(node, colorscheme)
            colors.append(fill_color)

        # Draw edges
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edge_color='#546E7A',
            width=2 if edge_width == 'auto' else edge_width,
            arrowsize=20,
            arrowstyle='-|>',
            node_size=sizes if isinstance(sizes, list) else [sizes] * len(G.nodes()),
        )

        # Draw nodes
        nx.draw_networkx_nodes(
            G,
            pos,
            ax=ax,
            node_color=colors,
            node_size=sizes,
            node_shape='s',
            edgecolors='#263238',
            linewidths=2,
        )

        # Draw labels
        if show_details:
            labels = {}
            for idx, node in enumerate(self.graph._nodes):
                labels[idx] = self._get_node_label(node, False).replace('\\n', '\n')

            nx.draw_networkx_labels(
                G, pos, labels, ax=ax,
                font_size=8,
                font_color='#263238',
            )

        ax.set_title("Model Dependency Graph", fontsize=14, fontweight='bold')
        ax.axis('off')
        fig.tight_layout()

        # Export if requested
        if export_path:
            fig.savefig(export_path, dpi=300, bbox_inches='tight')

        return fig

    def visualzie_matplotlib(
        self,
        ax=None,
    ):
        # Original matplotlib implementation
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import FancyBboxPatch
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Matplotlib backend requires matplotlib to be installed."
            ) from exc

        if not self.graph._nodes:
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

        num_nodes = len(self.graph._nodes)
        in_degree = {i: len(self.graph._reverse_edges.get(i, ())) for i in range(num_nodes)}
        levels = {i: 0 for i in range(num_nodes)}
        queue = deque(sorted(idx for idx, deg in in_degree.items() if deg == 0))
        if not queue:
            queue = deque(range(num_nodes))
        enqueued = set(queue)
        processed = set()
        while queue:
            idx = queue.popleft()
            processed.add(idx)
            for succ in sorted(self.graph._forward_edges.get(idx, ())):
                levels[succ] = max(levels.get(succ, 0), levels[idx] + 1)
                in_degree[succ] = in_degree.get(succ, 0) - 1
                if in_degree[succ] <= 0 and succ not in processed and succ not in enqueued:
                    queue.append(succ)
                    enqueued.add(succ)

        if len(processed) != num_nodes:
            remaining = set(range(num_nodes)) - processed
            for idx in remaining:
                preds = self.graph._reverse_edges.get(idx, ())
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

        def _node_label(node: GraphIRElem) -> str:
            if isinstance(node, GroupIR):
                return f"GroupIR\\n{node.name}"
            if isinstance(node, ProjectionIR):
                return f"ProjectionIR\\n{node.pre_group.name} → {node.post_group.name}"
            if isinstance(node, InputIR):
                return f"InputIR\\n→ {node.group.name}"
            if isinstance(node, OutputIR):
                return f"Output\\n{node.group.name} →"
            if isinstance(node, ConnectionIR):
                return f"ConnectionIR\\n{node.jaxpr.jaxpr.name if hasattr(node.jaxpr, 'jaxpr') else ''}"
            return type(node).__name__

        def _node_style(node: GraphIRElem) -> Tuple[str, str]:
            if isinstance(node, InputIR):
                return "#E3F2FD", "#1565C0"
            if isinstance(node, OutputIR):
                return "#FCE4EC", "#AD1457"
            if isinstance(node, ProjectionIR):
                return "#FFF3E0", "#E65100"
            if isinstance(node, ConnectionIR):
                return "#EDE7F6", "#5E35B1"
            return "#E8F5E9", "#1B5E20"  # Groups and fallbacks

        for idx, node in enumerate(self.graph._nodes):
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

        for source_idx, targets in self.graph._forward_edges.items():
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

    def _compute_hierarchical_layout(self):
        """Compute hierarchical layout positions."""
        num_nodes = len(self.graph._nodes)
        if num_nodes == 0:
            return {}, {}, 0, 0

        # Compute topological levels
        in_degree = {i: len(self.graph._reverse_edges.get(i, ())) for i in range(num_nodes)}
        levels = {i: 0 for i in range(num_nodes)}
        queue = deque(sorted(idx for idx, deg in in_degree.items() if deg == 0))
        if not queue:
            queue = deque(range(num_nodes))
        enqueued = set(queue)
        processed = set()

        while queue:
            idx = queue.popleft()
            processed.add(idx)
            for succ in sorted(self.graph._forward_edges.get(idx, ())):
                levels[succ] = max(levels.get(succ, 0), levels[idx] + 1)
                in_degree[succ] = in_degree.get(succ, 0) - 1
                if in_degree[succ] <= 0 and succ not in processed and succ not in enqueued:
                    queue.append(succ)
                    enqueued.add(succ)

        # Handle cycles
        if len(processed) != num_nodes:
            remaining = set(range(num_nodes)) - processed
            for idx in remaining:
                preds = self.graph._reverse_edges.get(idx, ())
                if preds:
                    max_pred = max(levels.get(pred, 0) for pred in preds)
                    levels[idx] = max(levels.get(idx, 0), max_pred + 1)
                else:
                    levels[idx] = 0

        # Group by level
        layer_map = defaultdict(list)
        for idx, level in levels.items():
            layer_map[level].append(idx)
        normalized_layers = []
        for _, nodes in sorted(layer_map.items()):
            normalized_layers.append(sorted(nodes))

        num_layers = len(normalized_layers)
        max_width = max((len(layer) for layer in normalized_layers), default=1)

        # Compute positions
        x_gap = 2.5
        y_gap = 2.0
        x_positions = {}
        y_positions = {}

        for layer_idx, layer_nodes in enumerate(normalized_layers):
            if not layer_nodes:
                continue
            row_width = len(layer_nodes)
            x_offset = (max_width - row_width) * 0.5 * x_gap
            y_val = (num_layers - layer_idx - 1) * y_gap
            for pos, node_idx in enumerate(layer_nodes):
                x_positions[node_idx] = x_offset + pos * x_gap
                y_positions[node_idx] = y_val

        return x_positions, y_positions, num_layers, max_width

    def _get_node_label(self, node, show_details=True):
        """Get display label for a node."""
        from ._data import GroupIR, ProjectionIR, InputIR, OutputIR, ConnectionIR

        if isinstance(node, GroupIR):
            return f"GroupIR\\n{node.name}"
        if isinstance(node, ProjectionIR):
            return f"ProjectionIR\\n{node.pre_group.name} → {node.post_group.name}"
        if isinstance(node, InputIR):
            return f"InputIR\\n→ {node.group.name}"
        if isinstance(node, OutputIR):
            return f"Output\\n{node.group.name} →"
        if isinstance(node, ConnectionIR):
            return f"ConnectionIR\\n{node.jaxpr.jaxpr.name if hasattr(node.jaxpr, 'jaxpr') else ''}"
        return type(node).__name__

    def _get_node_metadata(self, node):
        """Extract metadata from a node."""
        from ._data import GroupIR, ProjectionIR, InputIR, OutputIR

        metadata = {'Type': type(node).__name__}

        if isinstance(node, GroupIR):
            metadata['Name'] = node.name
            metadata['Hidden States'] = str(len(node.hidden_states))
            metadata['In States'] = str(len(node.in_states))
            metadata['Out States'] = str(len(node.out_states))
            metadata['Equations'] = str(len(node.jaxpr.jaxpr.eqns))
            metadata['InputIR Vars'] = str(len(node.input_vars))
        elif isinstance(node, ProjectionIR):
            metadata['From'] = node.pre_group.name
            metadata['To'] = node.post_group.name
            metadata['Hidden States'] = str(len(node.hidden_states))
            metadata['In States'] = str(len(node.in_states))
            metadata['Connections'] = str(len(node.connections))
            metadata['Equations'] = str(len(node.jaxpr.jaxpr.eqns))
        elif isinstance(node, InputIR):
            metadata['Target Group'] = node.group.name
            metadata['Equations'] = str(len(node.jaxpr.jaxpr.eqns))
        elif isinstance(node, OutputIR):
            metadata['Source Group'] = node.group.name
            metadata['Hidden States'] = str(len(node.hidden_states))
            metadata['Equations'] = str(len(node.jaxpr.jaxpr.eqns))

        return metadata

    def _get_color_scheme(self, colorscheme='default'):
        """Get color scheme for node types."""
        schemes = {
            'default': {
                'InputIR': ("#E3F2FD", "#1565C0"),
                'Output': ("#FCE4EC", "#AD1457"),
                'ProjectionIR': ("#FFF3E0", "#E65100"),
                'ConnectionIR': ("#EDE7F6", "#5E35B1"),
                'GroupIR': ("#E8F5E9", "#1B5E20"),
            },
            'pastel': {
                'InputIR': ("#BBDEFB", "#2196F3"),
                'Output': ("#F8BBD0", "#E91E63"),
                'ProjectionIR': ("#FFE0B2", "#FF9800"),
                'ConnectionIR': ("#D1C4E9", "#673AB7"),
                'GroupIR': ("#C8E6C9", "#4CAF50"),
            },
            'vibrant': {
                'InputIR': ("#2196F3", "#0D47A1"),
                'Output': ("#E91E63", "#880E4F"),
                'ProjectionIR': ("#FF9800", "#E65100"),
                'ConnectionIR': ("#673AB7", "#311B92"),
                'GroupIR': ("#4CAF50", "#1B5E20"),
            },
            'colorblind': {
                'InputIR': ("#0173B2", "#023858"),
                'Output': ("#DE8F05", "#7C4F00"),
                'ProjectionIR': ("#CC78BC", "#6E3F64"),
                'ConnectionIR': ("#029E73", "#015040"),
                'GroupIR': ("#ECE133", "#7A6F1A"),
            },
        }
        return schemes.get(colorscheme, schemes['default'])

    def _get_node_style(self, node, colorscheme='default'):
        """Get fill and edge colors for a node."""
        colors = self._get_color_scheme(colorscheme)
        node_type = type(node).__name__
        return colors.get(node_type, colors['GroupIR'])


class TextDisplayer:
    def __init__(self, graph):
        self.graph = graph
