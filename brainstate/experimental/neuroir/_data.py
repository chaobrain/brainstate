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
from typing import Iterable, Iterator, Set, Tuple, Dict, NamedTuple, Sequence, Any, Hashable, List

import jax

from brainstate._compatible_import import ClosedJaxpr, Var
from brainstate._state import State, DelayState
from brainstate.random._state import RandomState
from brainstate.transform._make_jaxpr import get_arg_cache_key
from ._utils import get_hidden_name

__all__ = [
    'GraphElem',
    'GroupPrim',
    'ConnectionPrim',
    'ProjectionPrim',
    'OutputPrim',
    'InputPrim',
    'UnknownPrim',
    'SpikePrim',
    'RandomPrim',
    'DelayPrim',
    'NeuroGraph',
    'CompiledGraphIR',
]


@dataclass
class GraphElem:
    """Base class for compiled graph IR elements."""

    jaxpr: ClosedJaxpr
    name: str  # Add name field with default value

    def jaxpr2hlo(self):
        pass


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

    def __hash__(self) -> int:
        """Return hash based on object identity for use in sets and dicts."""
        return hash(id(self))

    def __eq__(self, other) -> bool:
        """Check equality based on object identity."""
        return self is other

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
        """Return True if ``invar`` is consumed by this element.

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


class IntermediateGraphElem(GraphElem):
    pass


@dataclass(eq=False)
class ConnectionPrim(IntermediateGraphElem):
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
            f"{self.name}("
            f"prim={prim_name}, "
            f"invars={n_invars}, "
            f"outvars={n_outvars})"
        )


class FinalGraphElem(GraphElem):
    pass


@dataclass(eq=False)
class SpikePrim(GraphElem):
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

        return f"{self.name}(state={state_name}, eqns={n_eqns})"


@dataclass(eq=False)
class GroupPrim(FinalGraphElem):
    """Logical container for a compiled neuron group."""

    hidden_states: List[State]
    in_states: List[State]
    out_states: List[State]
    input_vars: List[Var]

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


@dataclass(eq=False)
class ProjectionPrim(FinalGraphElem):
    """ConnectionPrim bundle that transfers activity between two groups."""

    hidden_states: List[State]
    in_states: List[State]
    connections: List[ConnectionPrim]
    pre_group: GroupPrim
    post_group: GroupPrim

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
        pre_name = self.pre_group.name if hasattr(self.pre_group, 'name') else 'GroupPrim'
        post_name = self.post_group.name if hasattr(self.post_group, 'name') else 'GroupPrim'

        return (
            f"{self.name}("
            f"{pre_name} → {post_name}, "
            f"conns={n_conns}, "
            f"eqns={n_eqns}, "
            f"hidden={n_hidden}, "
            f"in_states={n_in})"
        )


@dataclass(eq=False)
class OutputPrim(FinalGraphElem):
    """Description of how values are extracted from a group."""

    hidden_states: List[State]
    in_states: List[State]
    group: GroupPrim

    def __repr__(self) -> str:
        """Return a representation showing output extraction details."""
        n_eqns = len(self.jaxpr.jaxpr.eqns)
        n_outvars = len(self.jaxpr.jaxpr.outvars)
        n_hidden = len(self.hidden_states)
        n_in = len(self.in_states)

        # Get group name
        group_name = self.group.name if hasattr(self.group, 'name') else 'GroupPrim'

        return (
            f"{self.name}("
            f"from={group_name}, "
            f"outvars={n_outvars}, "
            f"eqns={n_eqns}, "
            f"hidden={n_hidden}, "
            f"in_states={n_in})"
        )


@dataclass(eq=False)
class InputPrim(FinalGraphElem):
    """Description of how external values are injected into a group."""

    group: GroupPrim

    def __repr__(self) -> str:
        """Return a representation showing input injection details."""
        n_eqns = len(self.jaxpr.jaxpr.eqns)
        n_invars = len(self.jaxpr.jaxpr.invars)
        n_outvars = len(self.jaxpr.jaxpr.outvars)

        # Get group name
        group_name = self.group.name if hasattr(self.group, 'name') else 'GroupPrim'

        return (
            f"{self.name}("
            f"to={group_name}, "
            f"invars={n_invars}, "
            f"outvars={n_outvars}, "
            f"eqns={n_eqns})"
        )


@dataclass(eq=False)
class UnknownPrim(GraphElem):
    """UnknownPrim computation block containing unrecognized equations.

    This represents equations that were not assigned to any known component
    (GroupPrim, ProjectionPrim, InputPrim, or OutputPrim) during compilation. Consecutive
    unassigned equations are packaged into UnknownPrim blocks.

    UnknownPrim blocks are automatically integrated into the computation graph
    with proper dependency tracking:

    - If an UnknownPrim block consumes variables produced by a GroupPrim or ProjectionPrim,
      a dependency edge is created from the producer to the UnknownPrim block
    - If a GroupPrim or ProjectionPrim consumes variables produced by an UnknownPrim block,
      a dependency edge is created from the UnknownPrim block to the consumer
    - UnknownPrim blocks appear in the execution graph at their natural position
      based on the original equation order

    This ensures that UnknownPrim computations execute at the correct point in
    the computation sequence, respecting all data dependencies.

    Parameters
    ----------
    eqn_indices : Tuple[int, ...]
        Original equation indices in the source jaxpr, used for debugging
        and error reporting.

    Notes
    -----
    UnknownPrim blocks typically arise when the compiler encounters:

    - Custom operations not recognized as neuron groups, projections, or I/O
    - Intermediate computations that don't fit the standard SNN patterns
    - Complex control flow or data transformations

    The presence of UnknownPrim blocks doesn't indicate an error - they represent
    valid computations that simply don't match the expected neuron/synapse patterns.
    """

    eqn_indices: Tuple[int, ...]

    def __repr__(self) -> str:
        """Return a representation showing unknown block details."""
        n_eqns = len(self.jaxpr.jaxpr.eqns)
        if len(self.eqn_indices) <= 5:
            indices_str = str(self.eqn_indices)
        else:
            indices_str = f"({self.eqn_indices[0]}..{self.eqn_indices[-1]})"
        return f"{self.name}(eqns={n_eqns}, indices={indices_str})"


class RandomPrim(FinalGraphElem):
    """Random number generator."""
    state: RandomState


class DelayPrim(FinalGraphElem):
    state: DelayState


class NeuroGraph:
    """Directed graph capturing dependencies between compiled elements."""

    def __init__(self):
        self._nodes: List[GraphElem] = []
        self._id_to_index: Dict[int, int] = {}
        self._forward_edges: Dict[int, Set[int]] = defaultdict(set)
        self._reverse_edges: Dict[int, Set[int]] = defaultdict(set)

        # Categorized element collections
        self._groups: List['GroupPrim'] = []
        self._projections: List['ProjectionPrim'] = []
        self._inputs: List['InputPrim'] = []
        self._outputs: List['OutputPrim'] = []
        self._unknowns: List['UnknownPrim'] = []

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

        # Automatically categorize the node by type
        if isinstance(node, GroupPrim):
            self._groups.append(node)
        elif isinstance(node, ProjectionPrim):
            self._projections.append(node)
        elif isinstance(node, InputPrim):
            self._inputs.append(node)
        elif isinstance(node, OutputPrim):
            self._outputs.append(node)
        elif isinstance(node, UnknownPrim):
            self._unknowns.append(node)

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

    @property
    def groups(self) -> Tuple['GroupPrim', ...]:
        """Return all GroupPrim nodes in the graph.

        Returns
        -------
        tuple[GroupPrim, ...]
            All GroupPrim elements added to the graph.
        """
        return tuple(self._groups)

    @property
    def projections(self) -> Tuple['ProjectionPrim', ...]:
        """Return all ProjectionPrim nodes in the graph.

        Returns
        -------
        tuple[ProjectionPrim, ...]
            All ProjectionPrim elements added to the graph.
        """
        return tuple(self._projections)

    @property
    def inputs(self) -> Tuple['InputPrim', ...]:
        """Return all InputPrim nodes in the graph.

        Returns
        -------
        tuple[InputPrim, ...]
            All InputPrim elements added to the graph.
        """
        return tuple(self._inputs)

    @property
    def outputs(self) -> Tuple['OutputPrim', ...]:
        """Return all OutputPrim nodes in the graph.

        Returns
        -------
        tuple[OutputPrim, ...]
            All OutputPrim elements added to the graph.
        """
        return tuple(self._outputs)

    @property
    def unknowns(self) -> Tuple['UnknownPrim', ...]:
        """Return all UnknownPrim nodes in the graph.

        Returns
        -------
        tuple[UnknownPrim, ...]
            All UnknownPrim elements added to the graph.
        """
        return tuple(self._unknowns)

    def __len__(self) -> int:
        return len(self._nodes)

    def __iter__(self) -> Iterator[GraphElem]:
        return iter(self._nodes)

    def __repr__(self) -> str:
        """Return a text-based visualization of the graph structure.

        Returns
        -------
        str
            Formatted text representation showing nodes, edges, and structure.
        """
        from ._display import TextDisplayer
        return TextDisplayer(self).display()

    def text(self, verbose: bool = False, show_jaxpr: bool = False) -> str:
        """Return a text-based visualization of the graph structure.

        Parameters
        ----------
        verbose : bool, optional
            If True, include detailed node information. Default is False.
        show_jaxpr : bool, optional
            If True, display JAXPR equations for each node. Default is False.

        Returns
        -------
        str
            Formatted text representation showing nodes, edges, and structure.
        """
        from ._display import TextDisplayer
        return TextDisplayer(self).display(verbose=verbose, show_jaxpr=show_jaxpr)

    def visualize(
        self,
        layout: str = 'auto',
        figsize: Tuple[float, float] = (12, 8),
        **kwargs
    ):
        """Visualize the graph structure using matplotlib.

        This method creates an interactive visualization of the graph showing:
        - Different node types (GroupPrim, InputPrim, OutputPrim, ProjectionPrim) with distinct styles
        - Hierarchical relationships between nodes
        - Different edge styles for ProjectionPrim vs InputPrim/OutputPrim connections
        - Click-to-highlight functionality for exploring connections

        Parameters
        ----------
        layout : str, optional
            Layout algorithm to use for positioning nodes:

            - 'lr' or 'left-right': Left-to-right hierarchical layout (InputPrim on left, OutputPrim on right)
            - 'tb' or 'top-bottom': Top-to-bottom hierarchical layout (InputPrim on top, OutputPrim on bottom)
            - 'auto' or 'force': Force-directed layout with automatic node positioning

            Default is 'auto'.
        figsize : Tuple[float, float], optional
            Figure size as (width, height) in inches. Default is (12, 8).
        **kwargs
            Additional keyword arguments passed to the layout algorithm.
            For force-directed layout, you can specify 'iterations' (default: 100).

        Returns
        -------
        matplotlib.figure.Figure
            The matplotlib figure containing the visualization. You can display it
            with ``plt.show()`` or save it with ``fig.savefig(filename)``.

        Examples
        --------
        >>> graph.visualize(layout='lr')  # Left-to-right layout
        >>> plt.show()

        >>> graph.visualize(layout='tb', figsize=(16, 10))  # Top-to-bottom with custom size
        >>> plt.show()

        >>> graph.visualize(layout='auto', iterations=200)  # Force-directed with more iterations
        >>> plt.show()

        Notes
        -----

        - **GroupPrim nodes** are shown as large blue circles with their names displayed prominently
        - **InputPrim nodes** are shown as smaller green rounded rectangles with input count
        - **OutputPrim nodes** are shown as smaller orange rounded rectangles with output count
        - **ProjectionPrim nodes** are shown as small purple diamonds on edges between Groups
        - **ProjectionPrim connections** use solid purple lines
        - **InputPrim/OutputPrim connections** use dashed gray lines
        - Click any node to highlight its immediate predecessors and successors
        - Click empty space to clear highlights
        """
        from ._display import GraphDisplayer
        displayer = GraphDisplayer(self)
        return displayer.display(layout=layout, figsize=figsize, **kwargs)


class CompiledGraphIR(NamedTuple):
    """Structured result returned by :func:`neuroir.compile`.

    Attributes
    ----------
    graph : NeuroGraph
        Execution order for all components. Contains groups, projections,
        inputs, outputs, and unknowns accessible via properties.

    Notes
    -----
    Groups, projections, inputs, outputs, and unknowns can be accessed
    through the graph object:

    - `compiled.graph.groups` - All GroupPrim nodes
    - `compiled.graph.projections` - All ProjectionPrim nodes
    - `compiled.graph.inputs` - All InputPrim nodes
    - `compiled.graph.outputs` - All OutputPrim nodes
    - `compiled.graph.unknowns` - All UnknownPrim nodes
    """
    # graph IR data
    graph: NeuroGraph

    # others
    static_argnames: Sequence
    static_argnums: Sequence
    cache_key: Hashable
    out_treedef: Any
    jaxpr: ClosedJaxpr
    in_states: Sequence[State]
    out_states: Sequence[State]
    write_states: Sequence[State]
    invar_to_state: Dict[Var, State]
    outvar_to_state: Dict[Var, State]
    state_to_invars: Dict[State, Sequence[Var]]
    state_to_outvars: Dict[State, Sequence[Var]]

    def __call__(self, *args, **kwargs) -> Any:
        """Execute the compiled graph IR (default execution mode).

        This makes the CompiledGraphIR callable, using the compiled graph execution
        by default.

        Parameters
        ----------
        *args, **kwargs
            Runtime arguments forwarded to the original stateful function.

        Returns
        -------
        Any
            Result of executing the compiled graph IR.
        """
        return self.run_compiled(*args, **kwargs)

    def cache_fn(self, *args, **kwargs):
        return get_arg_cache_key(self.static_argnums, self.static_argnames, args, kwargs)

    def run_compiled(self, *args, **kwargs) -> Any:
        """Execute the compiled graph IR representation.

        This is the default execution mode that uses the structured graph IR
        for efficient computation.

        Parameters
        ----------
        *args, **kwargs
            Runtime arguments forwarded to the original stateful function.

        Returns
        -------
        Any
            Result of executing the compiled graph IR.
        """
        return self._run_impl(self._run_graph, *args, **kwargs)

    def run_original(self, *args, **kwargs) -> Any:
        """Execute the original ClosedJaxpr for comparison/debugging.

        This mode evaluates the raw Jaxpr without using the structured graph IR.
        Useful for validating compilation correctness.

        Parameters
        ----------
        *args, **kwargs
            Runtime arguments forwarded to the original stateful function.

        Returns
        -------
        Any
            Result of executing the original Jaxpr.
        """
        return self._run_impl(
            lambda *data: jax.core.eval_jaxpr(self.jaxpr.jaxpr, self.jaxpr.consts, *data),
            *args,
            **kwargs,
        )

    def debug_compare(self, *args, **kwargs) -> Tuple[Any, Any]:
        """Execute both original and compiled versions and return both results.

        This mode runs both execution paths without assigning state values,
        allowing comparison of outputs to validate compilation correctness.

        Parameters
        ----------
        *args, **kwargs
            Runtime arguments forwarded to the original stateful function.

        Returns
        -------
        tuple[Any, Any]
            A tuple ``(original_result, compiled_result)`` containing results
            from both execution modes.
        """
        original = self._run_impl(
            lambda *data: jax.core.eval_jaxpr(self.jaxpr.jaxpr, self.jaxpr.consts, *data),
            *args,
            assign_state_val=False,
            **kwargs,
        )
        compiled = self._run_impl(
            self._run_graph,
            *args,
            assign_state_val=False,
            **kwargs,
        )
        return original, compiled

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
        if len(new_state_vals) != len(self.write_states):
            raise ValueError(
                f'State length mismatch in output: expected '
                f'{len(self.write_states)} states, got {len(new_state_vals)}'
            )
        if assign_state_val:
            for st, val in zip(self.write_states, new_state_vals):
                if st is not None:
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
            if isinstance(component, InputPrim):
                self._execute_input(component, var_env)
            elif isinstance(component, GroupPrim):
                self._execute_group(component, var_env)
            elif isinstance(component, ProjectionPrim):
                self._execute_projection(component, var_env)
            elif isinstance(component, OutputPrim):
                self._execute_output(component, var_env)
            elif isinstance(component, UnknownPrim):
                self._execute_unknown(component, var_env)
            else:
                raise ValueError(f"UnknownPrim component type: {type(component)}")

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
        for var, val in zip(self.jaxpr.jaxpr.constvars, self.jaxpr.consts):
            var_env[var] = val

    def _execute_input(self, input_comp: InputPrim, var_env: Dict[Var, Any]) -> None:
        """Evaluate an :class:`InputPrim` component and store its outputs."""
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

    def _execute_group(self, group: GroupPrim, var_env: Dict[Var, Any]) -> None:
        """Evaluate a :class:`GroupPrim` subgraph using values from ``var_env``."""
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

    def _execute_projection(self, projection: ProjectionPrim, var_env: Dict[Var, Any]) -> None:
        """Evaluate a :class:`ProjectionPrim` component, including const fallbacks."""
        # Gather input values from environment
        input_vals = [var_env[var] for var in projection.jaxpr.jaxpr.invars]

        # Execute the projection jaxpr
        results = jax.core.eval_jaxpr(projection.jaxpr.jaxpr, projection.jaxpr.consts, *input_vals)

        # Handle single vs multiple outputs
        if not isinstance(results, (tuple, list)):
            results = (results,)

        # Store results in environment
        for var, val in zip(projection.jaxpr.jaxpr.outvars, results):
            var_env[var] = val

    def _execute_output(self, output: OutputPrim, var_env: Dict[Var, Any]) -> None:
        """Evaluate an :class:`OutputPrim` component using values in ``var_env``."""
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

    def _execute_unknown(self, unknown: UnknownPrim, var_env: Dict[Var, Any]) -> None:
        """Evaluate an :class:`UnknownPrim` component using values in ``var_env``."""
        # Gather input values from environment
        input_vals = [var_env[var] for var in unknown.jaxpr.jaxpr.invars]

        # Execute the unknown jaxpr
        results = jax.core.eval_jaxpr(unknown.jaxpr.jaxpr, unknown.jaxpr.consts, *input_vals)

        # Handle single vs multiple outputs
        if not isinstance(results, (tuple, list)):
            results = (results,)

        # Store results in environment
        for var, val in zip(unknown.jaxpr.jaxpr.outvars, results):
            var_env[var] = val

    def _collect_outputs(self, var_env: Dict[Var, Any]) -> Sequence[jax.typing.ArrayLike]:
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
                raise RuntimeError(f"OutputPrim variable {var} not found in environment")
            output_vals.append(var_env[var])
        return output_vals

    def __repr__(self) -> str:
        """Return a concise summary of the compiled graph IR.

        Returns
        -------
        str
            Human-readable representation showing compilation statistics.
        """
        n_groups = len(self.graph.groups)
        n_projections = len(self.graph.projections)
        n_inputs = len(self.graph.inputs)
        n_outputs = len(self.graph.outputs)
        n_graph_nodes = len(self.graph)
        n_in_states = len(self.in_states)
        n_out_states = len(self.out_states)

        # Get total equation count
        total_eqns = len(self.jaxpr.jaxpr.eqns)

        # Build group summary
        group_names = [g.name for g in self.graph.groups[:3]]
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
