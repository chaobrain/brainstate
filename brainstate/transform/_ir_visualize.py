# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
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


import importlib.util
import typing
from typing import Tuple, Union, List

import jax

from brainstate._compatible_import import (
    Var, ClosedJaxpr, Jaxpr, JaxprEqn, Literal, DropVar
)
from brainstate.transform._ir_utils import UnsupportedPrimitiveError

pydot_is_installed = importlib.util.find_spec("pydot") is not None

__all__ = [
    'draw', 'view_pydot', 'draw_dot_graph'
]

# Control-flow primitives that have dedicated handlers (``get_conditional``,
# ``get_scan``, ``get_while``) and therefore must NOT be treated as generic
# inlinable calls even though some of them carry sub-jaxpr parameters.
_DEDICATED_CONTROL_FLOW = frozenset({"cond", "scan", "while"})


def _is_dropped_var(var) -> bool:
    """Return ``True`` for JAX dropped / unused variables.

    JAX names outputs that are produced but never consumed with a trailing
    underscore (e.g. ``d_``). Such variables carry no data flow, so the
    visualisation skips them rather than drawing dangling edges. Centralising
    the rule here keeps the four call sites consistent and documented.
    """
    return str(var).endswith("_")


def _call_jaxpr_of(eqn) -> typing.Optional[Jaxpr]:
    """Return the inner :class:`Jaxpr` for an inlinable call primitive.

    Recognises ``jit``/``pjit`` (``jaxpr`` param holding a ``ClosedJaxpr``),
    ``remat2`` (``jaxpr`` param holding a bare ``Jaxpr``), ``closed_call`` and
    ``custom_*_call`` style primitives (``call_jaxpr`` param), and any other
    primitive that carries a ``jaxpr``/``call_jaxpr`` sub-computation.

    Dedicated control-flow primitives (``cond``/``scan``/``while``) are
    explicitly excluded so they keep routing to their own handlers. Returns
    ``None`` for ordinary leaf primitives.
    """
    if eqn.primitive.name in _DEDICATED_CONTROL_FLOW:
        return None
    for key in ("jaxpr", "call_jaxpr"):
        sub = eqn.params.get(key)
        if sub is None:
            continue
        return sub.jaxpr if isinstance(sub, ClosedJaxpr) else sub
    return None


def _eqn_name(eqn) -> str:
    """Best-effort display name for a call/leaf equation."""
    name = eqn.params.get("name")
    return name if name else eqn.primitive.name

if pydot_is_installed:
    import pydot

    sub_graph_return = Tuple[
        Union[pydot.Node, pydot.Subgraph],
        List[pydot.Edge],
        List[pydot.Node],
        List[pydot.Edge],
        int,
    ]

    GRAPH_STYLING = dict(
        fontname="Courier",
        fontsize="12",
        style="dotted",
        labeljust="l",
        nodesep=0.05,
    )
    IN_ARG_STYLING = dict(
        shape="box",
        color="green",
        fontname="Courier",
        fontsize="10",
    )
    LITERAL_STYLING = dict(
        shape="box",
        color="orange",
        fontname="Courier",
        fontsize="10",
    )
    CONST_ARG_STYLING = dict(
        shape="box",
        color="darkgreen",
        fontname="Courier",
        fontsize="10",
    )
    OUT_ARG_STYLING = dict(
        shape="box",
        color="red",
        fontname="Courier",
        fontsize="10",
    )
    VAR_STYLING = dict(
        shape="box",
        color="blue",
        fontname="Courier",
        fontsize="10",
    )
    PRIMITIVE_STYLING = dict(
        shape="box",
        fontname="Courier",
        fontsize="10",
        color="grey",
    )
    COND_NODE_STYLING = dict(
        shape="box",
        fontname="Courier",
        fontsize="10",
        color="grey",
    )
    FUNCTION_NODE_STYLING = dict(
        shape="rectangle",
        fontname="Courier",
        fontsize="10",
        style="dotted"
    )
    ARG_SUBGRAPH_STYLING = dict(label="", style="invis")


    def draw(f, collapse_primitives=True, show_avals=True) -> typing.Callable:
        """
        Visualise a JAX computation graph

        Wraps a JAX jit compiled function, which when called
        visualises the computation graph using
        pydot.

        Examples
        --------

        .. highlight:: python
        .. code-block:: python

           import jax
           import jpviz

           @jax.jit
           def foo(x):
               return 2 * x

           @jax.jit
           def bar(x):
               x = foo(x)
               return x - 1

           g = jpviz.draw(bar)(jax.numpy.arange(10))

        Parameters
        ----------
        f:
            JAX jit compiled function
        collapse_primitives: bool
            If `True` sub-functions that contain only JAX primitives
            will be collapsed into a single node in the generated
            graph
        show_avals: bool
            If `True` then type information will be
            included on node labels

        Returns
        -------
        Wrapped function that when called with concrete
        values generated the corresponding visualisation
        of the computation graph
        """

        def _inner_draw(*args, **kwargs) -> pydot.Graph:
            jaxpr = jax.make_jaxpr(f)(*args, **kwargs)
            return draw_dot_graph(jaxpr, collapse_primitives, show_avals)

        return _inner_draw


    def view_pydot(dot_graph: pydot.Dot) -> None:
        """
        Show a pydot graph in a jupyter notebook

        Parameters
        ----------
        dot_graph: Graph
            Pydot graph as generated by `draw`
        """
        from IPython.display import Image, display

        plt = Image(dot_graph.create_png())
        display(plt)


    def draw_dot_graph(
        fn: ClosedJaxpr, collapse_primitives: bool, show_avals: bool
    ) -> pydot.Graph:
        """
        Generate a pydot representation of an XLA graph

        Parameters
        ----------
        fn : ClosedJaxpr
        collapse_primitives: bool
            If `True` functions that are composed of only primitive
            values will be collapsed
        show_avals: bool
            If `True` type information will be included in the node label

        Returns
        -------
        Graph
            Pydot graph
        """

        g = pydot.Dot(graph_type="digraph")

        eqns = list(fn.eqns)

        # An empty top-level jaxpr (e.g. the identity function, or one that only
        # forwards/closes over constants) has no equations to expand. Render the
        # inputs and outputs so the graph is still meaningful instead of raising
        # ``IndexError`` on ``fn.eqns[0]``.
        if len(eqns) == 0:
            seen = set()
            for var in list(fn.invars):
                if _is_dropped_var(var):
                    continue
                node_id = f"_{var}"
                if node_id not in seen:
                    g.add_node(get_arg_node(node_id, var, show_avals,
                                            isinstance(var, Literal)))
                    seen.add(node_id)
            for var in list(fn.outvars):
                if isinstance(var, (Literal, DropVar)):
                    continue
                node_id = f"_{var}"
                if node_id not in seen:
                    g.add_node(get_out_node(node_id, var, show_avals))
                    seen.add(node_id)
            return g

        # Expand *every* top-level equation, not just the first one. The
        # original implementation indexed ``fn.eqns[0]`` and silently dropped
        # the rest whenever the top-level jaxpr held more than one equation.
        n = 0
        for eqn in eqns:
            sub_graph, in_edges, out_nodes, out_edges, n = get_sub_graph(
                eqn, "", n, collapse_primitives, show_avals
            )
            if isinstance(sub_graph, pydot.Subgraph):
                g.add_subgraph(sub_graph)
            else:
                g.add_node(sub_graph)
            for edge in in_edges:
                g.add_edge(edge)
            for node in out_nodes:
                g.add_node(node)
            for edge in out_edges:
                g.add_edge(edge)

        return g


    def get_conditional(
        conditional: Jaxpr,
        parent_id: str,
        n: int,
        collapse_primitives: bool,
        show_avals: bool,
    ) -> sub_graph_return:
        """
        Generate a subgraph representing a conditional function

        Parameters
        ----------
        conditional: Jaxpr
            Jaxpr of the conditional function
        parent_id: str
            ID of the parent subgraph of the conditional node
        collapse_primitives: bool
            If `True` any subgraph only consisting of primitive
            functions is collapsed into a single node
        show_avals: bool
            If `True` the type of the data is shown on
            argument/variable nodes on the generated graph
        n: int
            Integer used to generate unique ids for nodes, incremented
            as new nodes are added

        Returns
        -------
        (
            Union[pydot.Node, pydot.Subgraph],
            List[pydot.Edge],
            List[pydot.Node],
            List[pydot.Edge],
            int
        )
            Tuple containing:
                - Subgraph representing the conditional function and branches
                - List of edges that will connect a parent graph to the
                  arguments of the conditional function
                - List of nodes that should be added to a parent graph (i.e.
                  outputs of this graph)
                - List of edges connecting the outputs of this graph to
                  parent graph
                - Updated incremented integer used to get unique node ids
        """

        cond_graph_id = f"{parent_id}_cond_{n}"
        cond_graph = get_subgraph(f"cluster_{cond_graph_id}", "switch")
        n = n + 1

        cond_node_id = f"{cond_graph_id}_node"
        cond_arguments = pydot.Subgraph(f"{cond_graph_id}_inputs", rank="same")
        cond_arguments.add_node(
            pydot.Node(name=cond_node_id, label="idx", **COND_NODE_STYLING)
        )

        in_edges = list()
        new_nodes = list()
        out_edges = list()

        cond_var = conditional.invars[0]
        cond_var_id = f"{parent_id}_{cond_var}"
        if isinstance(cond_var, Literal):
            new_nodes.append(
                get_arg_node(cond_var_id, cond_var, show_avals, True)
            )
        in_edges.append(pydot.Edge(cond_var_id, cond_node_id))

        for arg in conditional.invars[1:]:
            arg_id = f"{cond_graph_id}_{arg}"
            is_literal = isinstance(arg, Literal)
            cond_arguments.add_node(
                get_arg_node(arg_id, arg, show_avals, is_literal)
            )
            in_edges.append(pydot.Edge(f"{parent_id}_{arg}", arg_id))

        cond_graph.add_subgraph(cond_arguments)

        for i, branch in enumerate(conditional.params["branches"]):
            if len(branch.eqns) == 0:
                branch_graph_id = f"{cond_node_id}_branch_{i}"
                label = f"branch {i}"

                if collapse_primitives:
                    cond_graph.add_node(
                        pydot.Node(
                            name=branch_graph_id,
                            label=label,
                            **FUNCTION_NODE_STYLING,
                        )
                    )

                    for (var, p_var) in zip(branch.jaxpr.invars, conditional.invars[1:]):
                        # Skip JAX dropped/unused vars (trailing-underscore names).
                        if _is_dropped_var(var):
                            continue
                        cond_graph.add_edge(
                            pydot.Edge(f"{cond_graph_id}_{p_var}", branch_graph_id)
                        )
                    for var in conditional.outvars:
                        cond_graph.add_edge(
                            pydot.Edge(branch_graph_id, f"{cond_graph_id}_{var}")
                        )
                else:
                    branch_graph = get_subgraph(
                        f"cluster_{branch_graph_id}", label
                    )
                    for (var, p_var) in zip(branch.jaxpr.invars, conditional.invars[1:]):
                        # Skip JAX dropped/unused vars (trailing-underscore names).
                        if _is_dropped_var(var):
                            continue
                        arg_id = f"{branch_graph_id}_{var}"
                        branch_graph.add_node(
                            get_var_node(arg_id, var, show_avals)
                        )
                        cond_graph.add_edge(pydot.Edge(f"{cond_graph_id}_{p_var}", arg_id))
                    for var, c_var in zip(branch.jaxpr.outvars, conditional.outvars):
                        arg_id = f"{branch_graph_id}_{var}"
                        cond_graph.add_edge(pydot.Edge(arg_id, f"{cond_graph_id}_{c_var}"))
                    cond_graph.add_subgraph(branch_graph)
            else:
                branch_graph_id = f"{cond_node_id}_branch_{i}"
                if len(branch.eqns) == 1:
                    eqn = branch.eqns[0]
                    branch_label = (
                        eqn.params["name"] if "name" in eqn.params else eqn.primitive.name
                    )
                    no_literal_inputs = any(
                        [isinstance(a, Literal) for a in branch.jaxpr.invars]
                    )
                    collapse_branch = no_literal_inputs or collapse_primitives
                    if collapse_branch:
                        branch_label = f"branch {i}: {branch_label}"
                    else:
                        branch_label = f"branch {i}"
                else:
                    branch_label = f"branch {i}"
                    collapse_branch = collapse_primitives

                if contains_non_primitives(branch.eqns) or not collapse_branch:
                    branch_graph = get_subgraph(
                        f"cluster_{branch_graph_id}", branch_label
                    )
                    branch_args, arg_edges = get_arguments(
                        branch_graph_id,
                        cond_graph_id,
                        branch.jaxpr.constvars,
                        branch.jaxpr.invars,
                        conditional.invars[1:],
                        show_avals,
                    )
                    for edge in arg_edges:
                        cond_graph.add_edge(edge)
                    branch_graph.add_subgraph(branch_args)

                    for eqn in branch.eqns:
                        (
                            eqn_graph,
                            eqn_in_edges,
                            eqn_out_nodes,
                            eqn_out_edges,
                            n,
                        ) = get_sub_graph(
                            eqn,
                            branch_graph_id,
                            n,
                            collapse_primitives,
                            show_avals,
                        )
                        if isinstance(eqn_graph, pydot.Subgraph):
                            branch_graph.add_subgraph(eqn_graph)
                        else:
                            branch_graph.add_node(eqn_graph)
                        for edge in eqn_in_edges:
                            branch_graph.add_edge(edge)
                        for node in eqn_out_nodes:
                            branch_graph.add_node(node)
                        for edge in eqn_out_edges:
                            branch_graph.add_edge(edge)

                    (
                        branch_out_graph,
                        branch_out_edges,
                        branch_out_nodes,
                        id_edges,
                    ) = get_outputs(
                        branch_graph_id,
                        cond_graph_id,
                        branch.jaxpr.invars,
                        branch.jaxpr.outvars,
                        conditional.outvars,
                        show_avals,
                    )
                    branch_graph.add_subgraph(branch_out_graph)
                    for edge in branch_out_edges:
                        cond_graph.add_edge(edge)
                    for node in branch_out_nodes:
                        cond_graph.add_node(node)
                    for edge in id_edges:
                        branch_graph.add_edge(edge)

                    cond_graph.add_subgraph(branch_graph)
                else:
                    cond_graph.add_node(
                        pydot.Node(
                            name=branch_graph_id,
                            label=branch_label,
                            **FUNCTION_NODE_STYLING,
                        )
                    )
                    for (var, p_var) in zip(branch.jaxpr.invars, conditional.invars[1:]):
                        # Skip JAX dropped/unused vars (trailing-underscore names).
                        if _is_dropped_var(var):
                            continue

                        # Compute literal-ness for *this* edge. The previous code
                        # referenced a stray ``is_literal`` left over from an
                        # unrelated earlier loop, which produced a stale value.
                        edge_is_literal = isinstance(var, Literal) or isinstance(p_var, Literal)
                        if not edge_is_literal:
                            cond_graph.add_edge(
                                pydot.Edge(f"{cond_graph_id}_{p_var}", branch_graph_id)
                            )

                    for var in conditional.outvars:
                        cond_graph.add_edge(
                            pydot.Edge(branch_graph_id, f"{cond_graph_id}_{var}")
                        )

        cond_out_graph, cond_out_edges, cond_out_nodes, _ = get_outputs(
            cond_graph_id,
            parent_id,
            conditional.invars,
            conditional.outvars,
            conditional.outvars,
            show_avals,
        )
        cond_graph.add_subgraph(cond_out_graph)
        out_edges.extend(cond_out_edges)
        new_nodes.extend(cond_out_nodes)

        return cond_graph, in_edges, new_nodes, out_edges, n


    def _get_node(
        eqn: JaxprEqn,
        graph_id: str,
        show_avals: bool,
        n: int,
        is_primitive: bool,
    ) -> sub_graph_return:
        """
        Generate a node representing a function and edges connecting it
        to a parent graph

        Parameters
        ----------
        eqn: JaxprEqn
            JaxprEqn of the function
        graph_id: str
            Id of the computation graph containing this node
        show_avals: bool
            If `True` the type of the data is shown on
            argument/variable nodes on the generated graph
        n: int
            Integer used to generate unique ids for nodes, incremented
            as new nodes are added
        is_primitive: bool
            Should be `True` if the function is a JAX primitive

        Returns
        -------
        (
            Union[pydot.Node, pydot.Subgraph],
            List[pydot.Edge],
            List[pydot.Node],
            List[pydot.Edge],
            int
        )
            Tuple containing:
                - Node representing the function
                - List of edges that will connect a parent graph to the
                  arguments of the function
                - List of nodes that should be added to a parent graph (i.e.
                  outputs of this graph)
                - List of edges connecting the outputs of this node to
                  parent graph
                - Updated incremented integer used to get unique node ids
        """

        name = str(eqn.primitive) if is_primitive else _eqn_name(eqn)
        node_id = f"{name}_{n}"
        n = n + 1

        style = PRIMITIVE_STYLING if is_primitive else FUNCTION_NODE_STYLING
        node = pydot.Node(name=node_id, label=name, **style)

        new_nodes = list()
        in_edges = list()
        out_edges = list()

        for var in eqn.invars:
            if isinstance(var, Literal):
                new_nodes.append(
                    get_arg_node(f"{graph_id}_{var}", var, show_avals, True)
                )
            in_edges.append(pydot.Edge(f"{graph_id}_{var}", node_id))

        for var in eqn.outvars:
            var_id = f"{graph_id}_{var}"
            new_nodes.append(get_var_node(var_id, var, show_avals))
            out_edges.append(pydot.Edge(node_id, var_id))

        return node, in_edges, new_nodes, out_edges, n


    def expand_non_primitive(
        eqn: JaxprEqn,
        parent_id: str,
        n: int,
        collapse_primitives: bool,
        show_avals: bool,
    ) -> sub_graph_return:
        """
        Expand a JaxprEqn into a computation graph/

        Parameters
        ----------
        eqn: JaxprEqn
            JaxprEqn of the function
        parent_id: str
            ID of the parent graph to this eqn
        n: int
            Integer used to generate unique ids for nodes, incremented
            as new nodes are added
        collapse_primitives: bool
            If `True` any functions that consist of only primitive
            elements will be collapsed to a single node
        show_avals: bool
            If `True` the type of the data is shown on
            argument/variable nodes on the generated graph

        Returns
        -------
        (
            Union[pydot.Node, pydot.Subgraph],
            List[pydot.Edge],
            List[pydot.Node],
            List[pydot.Edge],
            int
        )
            Tuple containing:
                - Node representing the function
                - List of edges that will connect a parent graph to the
                  arguments of the function
                - List of nodes that should be added to a parent graph (i.e.
                  outputs of this graph)
                - List of edges connecting the outputs of this node to
                  parent graph
                - Updated incremented integer used to get unique node ids
        """
        # Extract the inner jaxpr in a form-agnostic way: ``jit``/``pjit`` carry
        # a ``ClosedJaxpr`` under ``jaxpr``, ``remat2`` a bare ``Jaxpr``, and
        # ``closed_call`` a ``ClosedJaxpr`` under ``call_jaxpr``.
        inner_jaxpr = _call_jaxpr_of(eqn)
        if inner_jaxpr is None:
            raise UnsupportedPrimitiveError(
                f"Cannot expand primitive '{eqn.primitive.name}': no inlinable "
                f"jaxpr/call_jaxpr parameter found. Params: {sorted(eqn.params)}."
            )

        graph_name = _eqn_name(eqn)
        graph_id = f"{graph_name}_{n}"
        n = n + 1

        graph = pydot.Subgraph(
            f"cluster_{graph_id}",
            rank="same",
            label=graph_name,
            **GRAPH_STYLING,
        )

        argument_nodes, argument_edges = get_arguments(
            graph_id,
            parent_id,
            inner_jaxpr.constvars,
            inner_jaxpr.invars,
            eqn.invars,
            show_avals,
        )
        graph.add_subgraph(argument_nodes)

        for sub_eqn in inner_jaxpr.eqns:
            sub_graph, in_edges, out_nodes, out_edges, n = get_sub_graph(
                sub_eqn, graph_id, n, collapse_primitives, show_avals
            )
            if isinstance(sub_graph, pydot.Subgraph):
                graph.add_subgraph(sub_graph)
            else:
                graph.add_node(sub_graph)
            for edge in in_edges:
                graph.add_edge(edge)
            for node in out_nodes:
                graph.add_node(node)
            for edge in out_edges:
                graph.add_edge(edge)

        output_nodes, out_edges, out_nodes, id_edges = get_outputs(
            graph_id,
            parent_id,
            inner_jaxpr.invars,
            inner_jaxpr.outvars,
            eqn.outvars,
            show_avals,
        )

        graph.add_subgraph(output_nodes)
        for edge in id_edges:
            graph.add_edge(edge)

        return graph, argument_edges, out_nodes, out_edges, n


    def expand_pjit_primitive(
        jaxpr: Jaxpr,
        parent_id: str,
        n: int,
        collapse_primitives: bool,
        show_avals: bool,
        graph_name: str = ''
    ) -> sub_graph_return:
        """
        Expand a JaxprEqn into a computation graph/

        Parameters
        ----------
        jaxpr: Jaxpr
            JaxprEqn of the function
        parent_id: str
            ID of the parent graph to this eqn
        n: int
            Integer used to generate unique ids for nodes, incremented
            as new nodes are added
        collapse_primitives: bool
            If `True` any functions that consist of only primitive
            elements will be collapsed to a single node
        show_avals: bool
            If `True` the type of the data is shown on
            argument/variable nodes on the generated graph

        Returns
        -------
        (
            Union[pydot.Node, pydot.Subgraph],
            List[pydot.Edge],
            List[pydot.Node],
            List[pydot.Edge],
            int
        )
            Tuple containing:
                - Node representing the function
                - List of edges that will connect a parent graph to the
                  arguments of the function
                - List of nodes that should be added to a parent graph (i.e.
                  outputs of this graph)
                - List of edges connecting the outputs of this node to
                  parent graph
                - Updated incremented integer used to get unique node ids
        """
        graph_id = f"{graph_name}_{n}"
        n = n + 1

        graph = pydot.Subgraph(
            f"cluster_{graph_id}",
            rank="same",
            label=graph_name,
            **GRAPH_STYLING,
        )

        argument_nodes, argument_edges = get_arguments(
            graph_id,
            parent_id,
            jaxpr.constvars,
            jaxpr.invars,
            jaxpr.invars,
            show_avals,
        )
        graph.add_subgraph(argument_nodes)

        for sub_eqn in jaxpr.eqns:
            sub_graph, in_edges, out_nodes, out_edges, n = get_sub_graph(
                sub_eqn, graph_id, n, collapse_primitives, show_avals
            )
            if isinstance(sub_graph, pydot.Subgraph):
                graph.add_subgraph(sub_graph)
            else:
                graph.add_node(sub_graph)
            for edge in in_edges:
                graph.add_edge(edge)
            for node in out_nodes:
                graph.add_node(node)
            for edge in out_edges:
                graph.add_edge(edge)

        output_nodes, out_edges, out_nodes, id_edges = get_outputs(
            graph_id,
            parent_id,
            jaxpr.invars,
            jaxpr.outvars,
            jaxpr.outvars,
            show_avals,
        )

        graph.add_subgraph(output_nodes)
        for edge in id_edges:
            graph.add_edge(edge)

        return graph, argument_edges, out_nodes, out_edges, n


    def get_scan(
        eqn: JaxprEqn,
        parent_id: str,
        n: int,
        collapse_primitives: bool,
        show_avals: bool,
    ) -> sub_graph_return:
        graph_name = "scan"
        graph_id = f"{graph_name}_{n}"
        n = n + 1

        graph = pydot.Subgraph(
            f"cluster_{graph_id}",
            rank="same",
            label=graph_name,
            **GRAPH_STYLING,
        )

        argument_nodes, argument_edges = get_scan_arguments(
            graph_id,
            parent_id,
            eqn.params["jaxpr"].jaxpr.invars,
            eqn.invars,
            eqn.params["num_consts"],
            eqn.params["num_carry"],
            show_avals,
        )
        graph.add_subgraph(argument_nodes)

        out_var_keys = set(f"{graph_id}_{v}" for v in eqn.params["jaxpr"].jaxpr.outvars)
        eqns = eqn.params["jaxpr"].jaxpr.eqns

        body_graph_id = f"cluster_{graph_id}_body"

        if collapse_primitives and not contains_non_primitives(eqns):
            body_in = set()
            body_out = set()

            for sub_eqn in eqns:
                body_in.update([str(v) for v in sub_eqn.invars])
                body_out.update([str(v) for v in sub_eqn.outvars])

            parent_in = set(str(v) for v in eqn.params["jaxpr"].jaxpr.invars)
            parent_out = set(str(v) for v in eqn.params["jaxpr"].jaxpr.outvars)

            body_in = body_in.intersection(parent_in)
            body_out = body_out.intersection(parent_out)

            body_graph = pydot.Node(
                name=graph_id,
                label="body",
                **FUNCTION_NODE_STYLING,
            )
            graph.add_node(body_graph)

            for v in body_in:
                graph.add_edge(pydot.Edge(f"{graph_id}_{v}", graph_id))
            for v in body_out:
                graph.add_edge(pydot.Edge(graph_id, f"{graph_id}_{v}"))
        else:
            body_graph = pydot.Subgraph(
                body_graph_id,
                rank="same",
                label="body",
                **GRAPH_STYLING,
            )

            for sub_eqn in eqns:
                sub_graph, in_edges, out_nodes, out_edges, n = get_sub_graph(
                    sub_eqn, graph_id, n, collapse_primitives, show_avals
                )
                if isinstance(sub_graph, pydot.Subgraph):
                    body_graph.add_subgraph(sub_graph)
                else:
                    body_graph.add_node(sub_graph)
                for edge in in_edges:
                    graph.add_edge(edge)
                for node in out_nodes:
                    if not node.get_name() in out_var_keys:
                        body_graph.add_node(node)
                for edge in out_edges:
                    graph.add_edge(edge)

            graph.add_subgraph(body_graph)

        output_nodes, out_edges, out_nodes, id_edges = get_scan_outputs(
            graph_id,
            parent_id,
            eqn.params["jaxpr"].jaxpr.invars,
            eqn.params["jaxpr"].jaxpr.outvars,
            eqn.outvars,
            eqn.params["num_carry"],
            show_avals,
        )

        graph.add_subgraph(output_nodes)
        for edge in id_edges:
            graph.add_edge(edge)

        return graph, argument_edges, out_nodes, out_edges, n


    def get_while_branch(
        jaxpr: Jaxpr,
        parent_id: str,
        parent_args: List[Var],
        parent_outvars: List[Var],
        label: str,
        n: int,
        show_avals: bool,
        collapse_primitives: bool,
    ) -> Tuple[
        Union[pydot.Subgraph, pydot.Node],
        List[pydot.Edge],
        List[pydot.Edge],
        int,
    ]:
        """Build the ``cond``/``body`` branch of a ``while`` loop.

        Unlike :func:`get_sub_graph` (which returns the 5-tuple
        ``sub_graph_return``), this helper attaches its own output nodes to its
        own subgraph and therefore returns a 4-tuple:
        ``(node_or_subgraph, arg_edges, out_edges, n_counter)`` where the edge
        lists are the wiring that the *caller* must add to the enclosing
        ``while`` graph. The previous annotation declared only three elements,
        which did not match the value actually returned.
        """
        graph_id = f"cluster_{parent_id}_{label}"

        if collapse_primitives and not contains_non_primitives(jaxpr.eqns):
            graph = pydot.Node(
                name=graph_id,
                label=label,
                **FUNCTION_NODE_STYLING,
            )
            arg_edges = list()
            out_edges = list()

            for (var, p_var) in zip(jaxpr.invars, parent_args):
                # Skip JAX dropped/unused vars (trailing-underscore names).
                if _is_dropped_var(var):
                    continue
                is_literal = isinstance(var, Literal)
                if not is_literal:
                    arg_edges.append(pydot.Edge(f"{parent_id}_{p_var}", graph_id))

            for (var, p_var) in zip(jaxpr.outvars, parent_outvars):
                if isinstance(var, DropVar):
                    continue
                out_edges.append(pydot.Edge(graph_id, f"{parent_id}_{p_var}"))

            return graph, arg_edges, out_edges, n
        else:
            graph = get_subgraph(graph_id, label)
            arg_nodes, outer_arg_edges = get_arguments(
                graph_id,
                parent_id,
                [],
                jaxpr.invars,
                parent_args,
                show_avals,
            )
            graph.add_subgraph(arg_nodes)

            for eqn in jaxpr.eqns:
                (
                    sub_graph,
                    arg_edges,
                    out_nodes,
                    out_edges,
                    n,
                ) = get_sub_graph(eqn, graph_id, n, collapse_primitives, show_avals)
                if isinstance(sub_graph, pydot.Subgraph):
                    graph.add_subgraph(sub_graph)
                else:
                    graph.add_node(sub_graph)
                for edge in arg_edges:
                    graph.add_edge(edge)
                for node in out_nodes:
                    graph.add_node(node)
                for edge in out_edges:
                    graph.add_edge(edge)

            out_nodes, outer_out_edges, _, id_edges = get_outputs(
                graph_id,
                parent_id,
                jaxpr.invars,
                jaxpr.outvars,
                parent_outvars,
                show_avals,
            )
            graph.add_subgraph(out_nodes)
            for e in id_edges:
                graph.add_edge(e)

            return graph, outer_arg_edges, outer_out_edges, n


    def get_while(
        eqn: JaxprEqn,
        parent_id: str,
        n: int,
        collapse_primitives: bool,
        show_avals: bool,
    ) -> sub_graph_return:
        while_graph_id = f"{parent_id}_while_{n}"
        while_graph = get_subgraph(f"cluster_{while_graph_id}", "while")
        n = n + 1

        n_cond_const = eqn.params["cond_nconsts"]
        n_body_const = eqn.params["body_nconsts"]
        cond_consts = eqn.invars[:n_cond_const]
        body_consts = eqn.invars[n_cond_const: n_cond_const + n_body_const]
        init_carry = eqn.invars[n_cond_const + n_body_const:]

        arg_edges = list()
        out_edges = list()

        for var in eqn.invars:
            arg_id = f"{while_graph_id}_{var}"
            is_literal = isinstance(var, Literal)
            while_graph.add_node(
                get_arg_node(arg_id, var, show_avals, is_literal)
            )
            if not is_literal:
                arg_edges.append(pydot.Edge(f"{parent_id}_{var}", arg_id))

        cond_graph, cond_arg_edges, _, n = get_while_branch(
            eqn.params["cond_jaxpr"].jaxpr,
            while_graph_id,
            cond_consts + init_carry,
            eqn.outvars,
            "cond",
            n,
            show_avals,
            collapse_primitives,
        )
        for e in cond_arg_edges:
            while_graph.add_edge(e)

        body_graph, body_arg_edges, body_out_edges, n = get_while_branch(
            eqn.params["body_jaxpr"].jaxpr,
            while_graph_id,
            body_consts + init_carry,
            eqn.outvars,
            "body",
            n,
            show_avals,
            collapse_primitives,
        )
        for e in body_arg_edges:
            while_graph.add_edge(e)
        for e in body_out_edges:
            while_graph.add_edge(e)

        if isinstance(cond_graph, pydot.Subgraph):
            while_graph.add_subgraph(cond_graph)
        else:
            while_graph.add_node(cond_graph)
        if isinstance(body_graph, pydot.Subgraph):
            while_graph.add_subgraph(body_graph)
        else:
            while_graph.add_node(body_graph)

        for var in eqn.outvars:
            arg_id = f"{while_graph_id}_{var}"
            while_graph.add_node(get_out_node(arg_id, var, show_avals))
            if not isinstance(var, DropVar):
                out_edges.append(pydot.Edge(arg_id, f"{parent_id}_{var}"))

        return while_graph, arg_edges, [], out_edges, n


    def get_sub_graph(
        eqn: JaxprEqn,
        parent_id: str,
        n: int,
        collapse_primitives: bool,
        show_avals: bool,
    ) -> sub_graph_return:
        """
        Generate a node/subgraph representing a function

        The returned node/subgraph is conditional on the function
        type. This function recursively walks nodes on the graph
        of this function to generate sub-graphs of sub-functions

        Parameters
        ----------
        eqn: JaxprEqn
            JaxprEqn of the function
        parent_id: str
            ID of the
        n: int
            Integer used to generate unique ids for nodes, incremented
            as new nodes are added
        collapse_primitives: bool
            If `True` any subgraph only consisting of primitive
            functions is collapsed into a single node
        show_avals: bool
            If `True` the type of the data is shown on
            argument/variable nodes on the generated graph

        Returns
        -------
        (
            Union[pydot.Node, pydot.Subgraph],
            List[pydot.Edge],
            List[pydot.Node],
            List[pydot.Edge],
            int
        )
            Tuple containing:
                - Subgraph or node representing the function
                - List of edges that will connect a parent graph to the
                  arguments of the function
                - List of nodes that should be added to a parent graph (i.e.
                  outputs of this graph)
                - List of edges connecting the outputs of this node to
                  parent graph
                - Updated incremented integer used to get unique node ids
        """

        if is_not_primitive(eqn):
            inner_jaxpr = _call_jaxpr_of(eqn)
            if (
                contains_non_primitives(inner_jaxpr.eqns)
                or not collapse_primitives
            ):
                return expand_non_primitive(
                    eqn,
                    parent_id,
                    n,
                    collapse_primitives,
                    show_avals,
                )
            else:
                # Return a node representing a function
                return _get_node(
                    eqn,
                    parent_id,
                    show_avals,
                    n,
                    False,
                )
        else:
            if eqn.primitive.name == "cond":
                # Return a conditional subgraph
                return get_conditional(eqn, parent_id, n, collapse_primitives, show_avals)
            elif eqn.primitive.name == "scan":
                return get_scan(
                    eqn,
                    parent_id,
                    n,
                    collapse_primitives,
                    show_avals,
                )
            elif eqn.primitive.name == "while":
                return get_while(
                    eqn,
                    parent_id,
                    n,
                    collapse_primitives,
                    show_avals,
                )
            else:
                # Return a primitive node
                return _get_node(
                    eqn,
                    parent_id,
                    show_avals,
                    n,
                    True,
                )


    def get_arg_node(
        arg_id: str,
        var: Union[Var, Literal],
        show_avals: bool,
        is_literal: bool,
    ) -> pydot.Node:
        """
        Return a pydot node representing a function input/argument

        Parameters
        ----------
        arg_id: str
            Unique ID of the node
        var: jax._src.core.Var
            JAX variable or literal instance
        show_avals: bool
            If `True` show the type in the node
        is_literal: True
            Should be `True` if the node is a literal (and
            should be styled as such)

        Returns
        -------
        pydot.Node
        """
        style = LITERAL_STYLING if is_literal else IN_ARG_STYLING
        return pydot.Node(
            name=arg_id,
            label=get_node_label(var, show_avals),
            **style,
        )


    def get_const_node(
        arg_id: str,
        var: Union[Var, Literal],
        show_avals: bool,
    ) -> pydot.Node:
        """
        Return a pydot node representing a function const arg

        Parameters
        ----------
        arg_id: str
            Unique ID of the node
        var: jax._src.core.Var
            JAX variable
        show_avals: bool
            If `True` show the type in the node

        Returns
        -------
        pydot.Node
        """
        return pydot.Node(
            name=arg_id,
            label=get_node_label(var, show_avals),
            **CONST_ARG_STYLING,
        )


    def get_var_node(var_id: str, var: Var, show_avals: bool) -> pydot.Node:
        """
        Get a pydot node representing a variable internal to a function

        Parameters
        ----------
        var_id: str
            Unique ID of the node
        var: jax._src.core.Var
            JAX variable instance
        show_avals: bool
            If `True` show the type in the node

        Returns
        -------
        pydot.Node
        """
        return pydot.Node(
            name=var_id,
            label=get_node_label(var, show_avals),
            **VAR_STYLING,
        )


    def get_out_node(out_id: str, var: Var, show_avals: bool) -> pydot.Node:
        """
        Get a pydot node representing the outputs of a function

        Parameters
        ----------
        out_id: str
            Unique ID of the node
        var: jax._src.core.Var
            JAX variable instance
        show_avals: bool
            If `True` show the type in the node

        Returns
        -------
        pydot.Node
        """
        return pydot.Node(
            name=out_id,
            label=get_node_label(var, show_avals),
            **OUT_ARG_STYLING,
        )


    def get_subgraph(graph_id: str, label: str) -> pydot.Subgraph:
        """
        Get a pydot subgraph

        Parameters
        ----------
        graph_id: str
            Unique ID of the subgraph
        label: str
            Label of the subgraph

        Returns
        -------
        pydot.Subgraph
        """
        return pydot.Subgraph(
            graph_id,
            label=label,
            rank="same",
            **GRAPH_STYLING,
        )


    def get_arguments(
        graph_id: str,
        parent_id: str,
        graph_consts: List[Var],
        graph_invars: List[Var],
        parent_invars: List[Var],
        show_avals: bool,
    ) -> Tuple[pydot.Subgraph, List[pydot.Edge]]:
        """
        Generate a subgraph containing arguments, and edges connecting
        it to its parent graph

        Parameters
        ----------
        graph_id: str
            ID of the subgraph that owns the arguments
        parent_id: str
            ID of the parent of the subgraph
        graph_consts: List[jax._src.core.Var]
            List of graph const-vars
        graph_invars: List[jax._src.core.Var]
            List of input variables to the subgraph
        parent_invars: List[jax._src.core.Var]
            List of the corresponding input variables from the parent subgraph
        show_avals: bool
            If `True` show the type in the node

        Returns
        -------
        (pydot.Subgraph, List[pydot.Edge])
            Tuple containing the argument subgraph and a list of
            edges that connect variables in the parent graph to
            the inputs of this subgraph.
        """
        argument_nodes = pydot.Subgraph(
            f"{graph_id}_args", rank="same", **ARG_SUBGRAPH_STYLING
        )
        argument_edges = list()

        for var in graph_consts:
            arg_id = f"{graph_id}_{var}"
            argument_nodes.add_node(get_const_node(arg_id, var, show_avals))

        for var, p_var in zip(graph_invars, parent_invars):
            # Skip JAX dropped/unused vars (trailing-underscore names).
            if _is_dropped_var(var):
                continue
            arg_id = f"{graph_id}_{var}"
            is_literal = isinstance(var, Literal)
            argument_nodes.add_node(get_arg_node(arg_id, var, show_avals, is_literal))
            if not is_literal:
                argument_edges.append(pydot.Edge(f"{parent_id}_{p_var}", arg_id))

        return argument_nodes, argument_edges


    def get_scan_arguments(
        graph_id: str,
        parent_id: str,
        graph_invars: List[Var],
        parent_invars: List[Var],
        n_const: int,
        n_carry: int,
        show_avals: bool,
    ) -> Tuple[pydot.Subgraph, List[pydot.Edge]]:
        """
        Generate a subgraph containing arguments, and edges connecting
        it to its parent graph. Groups scan init/carry nodes.

        Parameters
        ----------
        graph_id: str
            ID of the subgraph that owns the arguments
        parent_id: str
            ID of the parent of the subgraph
        graph_invars: List[jax._src.core.Var]
            List of input variables to the subgraph
        parent_invars: List[jax._src.core.Var]
            List of the corresponding input variables from the parent subgraph
        n_const: int
            Number of scan constant arguments
        n_carry: int
            Number of scan carry arguments
        show_avals: bool
            If `True` show the type in the node

        Returns
        -------
        (pydot.Subgraph, List[pydot.Edge])
            Tuple containing the argument subgraph and a list of
            edges that connect variables in the parent graph to
            the inputs of this subgraph.
        """
        argument_nodes = pydot.Subgraph(
            f"cluster_{graph_id}_args", rank="min", **ARG_SUBGRAPH_STYLING
        )
        const_nodes = pydot.Subgraph(
            f"cluster_{graph_id}_const", rank="same", label="init", style="dotted"
        )
        carry_nodes = pydot.Subgraph(
            f"cluster_{graph_id}_init", rank="same", label="consts", style="dotted"
        )
        iterate_nodes = pydot.Subgraph(
            f"cluster_{graph_id}_iter", rank="same", label="iterate", style="dotted"
        )
        argument_edges = list()

        for i, (var, p_var) in enumerate(zip(graph_invars, parent_invars)):
            # Skip JAX dropped/unused vars (trailing-underscore names).
            if _is_dropped_var(var):
                continue
            arg_id = f"{graph_id}_{var}"

            var_is_literal = isinstance(var, Literal)
            parent_is_literal = isinstance(p_var, Literal)
            is_literal = var_is_literal or parent_is_literal

            if parent_is_literal:
                literal_id = f"{arg_id}_lit"
                argument_nodes.add_node(get_arg_node(literal_id, p_var, show_avals, True))
                argument_nodes.add_edge(pydot.Edge(literal_id, arg_id))

            if i < n_const:
                const_nodes.add_node(get_arg_node(arg_id, var, show_avals, var_is_literal))
            elif i < n_carry + n_const:
                carry_nodes.add_node(get_arg_node(arg_id, var, show_avals, var_is_literal))
            else:
                iterate_nodes.add_node(
                    get_arg_node(arg_id, var, show_avals, var_is_literal)
                )

            if not is_literal:
                argument_edges.append(pydot.Edge(f"{parent_id}_{p_var}", arg_id))

        argument_nodes.add_subgraph(const_nodes)
        argument_nodes.add_subgraph(carry_nodes)
        argument_nodes.add_subgraph(iterate_nodes)
        return argument_nodes, argument_edges


    def get_outputs(
        graph_id: str,
        parent_id: str,
        graph_invars: List[Var],
        graph_outvars: List[Var],
        parent_outvars: List[Var],
        show_avals: bool,
    ) -> Tuple[
        pydot.Subgraph,
        List[pydot.Edge],
        List[pydot.Node],
        List[pydot.Edge],
    ]:
        """
        Generate a subgraph containing function output nodes, and
        edges and nodes that connect outputs to the parent graph

        Parameters
        ----------
        graph_id: str
            ID of the subgraph
        parent_id: str
            ID of the parent graph
        graph_invars: List[jax._src.core.Var]
            List of funtion input variables
        graph_outvars: List[jax._src.core.Var]
            List of output function variables
        parent_outvars: List[jax._src.core.Var]
            Corresponding list of variable from the parent
            graph that are outputs from this graph
        show_avals: bool
            If `True` show the type in the node

        Returns
        -------
        (
            pydot.Subgraph,
            List[pydot.Edge],
            List[pydot.Node],
            List[pydot.Edge]
        )
            Tuple containing:
                - The subgraph wrapping the output nodes
                - A list of edges connecting to the parent graph
                - A list of variable nodes that should be added to the
                  parent graph (as outputs from this graph)
                - A list of edges that connect inputs directly to outputs
                  in the case an argument is returned by the function
        """
        out_graph = pydot.Subgraph(
            f"{graph_id}_outs", rank="same", **ARG_SUBGRAPH_STYLING
        )
        out_edges = list()
        out_nodes = list()
        id_edges = list()
        in_var_set = set([str(x) for x in graph_invars])

        for var, p_var in zip(graph_outvars, parent_outvars):
            if str(var) in in_var_set:
                arg_id = f"{graph_id}_{var}_out"
                id_edges.append(pydot.Edge(f"{graph_id}_{var}", arg_id))
            else:
                arg_id = f"{graph_id}_{var}"
            out_graph.add_node(get_out_node(arg_id, var, show_avals))
            out_edges.append(pydot.Edge(arg_id, f"{parent_id}_{p_var}"))
            out_nodes.append(get_var_node(f"{parent_id}_{p_var}", p_var, show_avals))

        return out_graph, out_edges, out_nodes, id_edges


    def get_scan_outputs(
        graph_id: str,
        parent_id: str,
        graph_invars: List[Var],
        graph_outvars: List[Var],
        parent_outvars: List[Var],
        n_carry: int,
        show_avals: bool,
    ) -> Tuple[
        pydot.Subgraph,
        List[pydot.Edge],
        List[pydot.Node],
        List[pydot.Edge],
    ]:
        """
        Generate a subgraph containing function output nodes, and
        edges and nodes that connect outputs to the parent graph.
        Groups carry nodes.

        Parameters
        ----------
        graph_id: str
            ID of the subgraph
        parent_id: str
            ID of the parent graph
        graph_invars: List[jax._src.core.Var]
            List of funtion input variables
        graph_outvars: List[jax._src.core.Var]
            List of output function variables
        parent_outvars: List[jax._src.core.Var]
            Corresponding list of variable from the parent
            graph that are outputs from this graph
        n_carry: int
            Number of scan carry arguments
        show_avals: bool
            If `True` show the type in the node

        Returns
        -------
        (
            pydot.Subgraph,
            List[pydot.Edge],
            List[pydot.Node],
            List[pydot.Edge]
        )
            Tuple containing:
                - The subgraph wrapping the output nodes
                - A list of edges connecting to the parent graph
                - A list of variable nodes that should be added to the
                  parent graph (as outputs from this graph)
                - A list of edges that connect inputs directly to outputs
                  in the case an argument is returned by the function
        """
        out_graph = pydot.Subgraph(
            f"cluster_{graph_id}_outs", rank="same", **ARG_SUBGRAPH_STYLING
        )
        carry_nodes = pydot.Subgraph(
            f"cluster_{graph_id}_carry", rank="same", label="carry", style="dotted"
        )
        accumulate_nodes = pydot.Subgraph(
            f"cluster_{graph_id}_acc", rank="same", label="accumulate", style="dotted"
        )
        out_edges = list()
        out_nodes = list()
        id_edges = list()
        in_var_set = set([str(x) for x in graph_invars])

        for i, (var, p_var) in enumerate(zip(graph_outvars, parent_outvars)):
            if str(var) in in_var_set:
                arg_id = f"{graph_id}_{var}_out"
                id_edges.append(pydot.Edge(f"{graph_id}_{var}", arg_id))
            else:
                arg_id = f"{graph_id}_{var}"
            if i < n_carry:
                carry_nodes.add_node(get_out_node(arg_id, var, show_avals))
            else:
                accumulate_nodes.add_node(get_out_node(arg_id, var, show_avals))
            out_edges.append(pydot.Edge(arg_id, f"{parent_id}_{p_var}"))
            out_nodes.append(get_var_node(f"{parent_id}_{p_var}", p_var, show_avals))

        out_graph.add_subgraph(carry_nodes)
        out_graph.add_subgraph(accumulate_nodes)
        return out_graph, out_edges, out_nodes, id_edges


    def get_node_label(
        v: Union[Var, Literal],
        show_avals: bool
    ) -> str:
        """
        Concatenate a variable name and its type.

        Parameters
        ----------
        v: Var
            Jax variable
        show_avals: bool
            If `True` then the type will be included in the
            node label

        Returns
        -------
        str
        """
        if show_avals:
            return f"{v}: {v.aval.str_short()}"
        else:
            return str(v)


    def is_not_primitive(x: JaxprEqn) -> bool:
        """
        Test whether a JaxprEqn is an inlinable call (i.e. *not* a leaf
        primitive) that should be expanded into a sub-graph.

        Recognises ``jit``/``pjit``, ``remat2``, ``closed_call`` and any other
        primitive carrying a ``jaxpr``/``call_jaxpr`` sub-computation. The
        original implementation only matched the literal name ``"pjit"``, which
        no longer matches modern JAX (the jit primitive is named ``"jit"``) and
        missed ``remat2``/``closed_call`` entirely.

        Parameters
        ----------
        x: JaxprEqn

        Returns
        -------
        bool
            'True' if this equation wraps an inlinable sub-jaxpr.
        """
        return _call_jaxpr_of(x) is not None


    def contains_non_primitives(eqns: List[JaxprEqn]) -> bool:
        """
        Check if the sub-functions of a JaxPR contain anything other than
        leaf JAX primitives (i.e. nested calls or control flow).

        Parameters
        ----------
        eqns: List[jax._src.core.JaxprEqn]
            List of JaxprEqns

        Returns
        -------
        bool:
            `True` if any of the sub-eqns are non-primitive
        """
        return any(
            [
                (
                    "jaxpr" in e.params
                    or "call_jaxpr" in e.params
                    or e.primitive.name in {"cond", "scan", "while"}
                )
                for e in eqns
            ]
        )

else:
    def draw(*args, **kwargs):
        raise NotImplementedError('pydot is not installed. ')


    def view_pydot(*args, **kwargs):
        raise NotImplementedError('pydot is not installed. ')


    def draw_dot_graph(*args, **kwargs):
        raise NotImplementedError('pydot is not installed. ')
