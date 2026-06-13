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

import importlib.util
import sys
import types
import unittest

import jax
import jax.numpy as jnp

PYDOT = importlib.util.find_spec("pydot") is not None


class TestVisualizeImportSafety(unittest.TestCase):
    def test_module_imports_without_pydot(self):
        # Importing the module must never fail, even if pydot is missing.
        import brainstate.transform._ir_visualize as viz
        self.assertTrue(hasattr(viz, 'draw'))
        self.assertTrue(hasattr(viz, 'view_pydot'))
        self.assertTrue(hasattr(viz, 'draw_dot_graph'))

    @unittest.skipIf(PYDOT, "pydot is installed; stub behavior not exercised")
    def test_stubs_raise_without_pydot(self):
        import brainstate.transform._ir_visualize as viz
        with self.assertRaises(NotImplementedError):
            viz.draw(lambda x: x)


@unittest.skipUnless(PYDOT, "pydot not installed")
class TestVisualizeBasic(unittest.TestCase):
    def test_simple_arithmetic_builds(self):
        from brainstate.transform import draw
        @jax.jit
        def f(x):
            return 2.0 * x + 1.0
        g = draw(f)(jnp.float32([1., 2., 3.]))
        self.assertIsNotNone(g)

    def test_show_avals_false_builds(self):
        from brainstate.transform import draw
        @jax.jit
        def f(x):
            return x * x
        g = draw(f, show_avals=False)(jnp.float32([1., 2.]))
        self.assertIsNotNone(g)


@unittest.skipUnless(PYDOT, "pydot not installed")
class TestVisualizeCond(unittest.TestCase):
    def test_cond_with_function_branch_builds(self):
        from brainstate.transform import draw
        @jax.jit
        def f(x):
            # Branches non-trivial enough to render as function nodes.
            return jax.lax.cond(
                x[0] > 0,
                lambda v: jnp.sum(v * v),
                lambda v: jnp.sum(v) - 1.0,
                x,
            )
        # Must not raise UnboundLocalError / NameError.
        g = draw(f)(jnp.float32([1., 2., 3.]))
        self.assertIsNotNone(g)

    def test_cond_collapsed_builds(self):
        from brainstate.transform import draw
        @jax.jit
        def f(x):
            return jax.lax.cond(x > 0, lambda v: v + 1.0, lambda v: v - 1.0, x)
        g = draw(f, collapse_primitives=True)(jnp.float32(1.0))
        self.assertIsNotNone(g)


def _count_nodes(graph):
    """Recursively count all nodes in a pydot graph/subgraph."""
    total = len(graph.get_nodes())
    for sub in graph.get_subgraphs():
        total += _count_nodes(sub)
    return total


def _all_nodes(graph):
    """Recursively collect every pydot.Node in a graph/subgraph."""
    nodes = list(graph.get_nodes())
    for sub in graph.get_subgraphs():
        nodes += _all_nodes(sub)
    return nodes


def _all_edges(graph):
    """Recursively collect every pydot.Edge in a graph/subgraph."""
    edges = list(graph.get_edges())
    for sub in graph.get_subgraphs():
        edges += _all_edges(sub)
    return edges


def _all_subgraphs(graph):
    """Recursively collect every pydot.Subgraph below ``graph``."""
    out = []
    for sub in graph.get_subgraphs():
        out.append(sub)
        out += _all_subgraphs(sub)
    return out


def _base(name):
    """Port-stripped node name (pydot treats ':' as a port separator)."""
    return str(name).split(":")[0]


def _declared_node_bases(graph):
    """Set of port-stripped names of all declared nodes."""
    return {_base(nd.get_name()) for nd in _all_nodes(graph)}


def _undeclared_edge_sources(graph):
    """Edge-source base ids that are not declared as nodes anywhere.

    A non-empty result means an edge references a phantom node that graphviz
    would fabricate with default styling -- the signature of the literal /
    boundary-node bugs.
    """
    declared = _declared_node_bases(graph)
    return sorted(
        {_base(e.get_source()) for e in _all_edges(graph)} - declared
    )


@unittest.skipUnless(PYDOT, "pydot not installed")
class TestVisualizeTopLevel(unittest.TestCase):
    def test_empty_jaxpr_does_not_crash(self):
        # A *non-jitted* identity traces to a top-level jaxpr with zero eqns,
        # which used to raise ``IndexError`` on ``fn.eqns[0]``.
        from brainstate.transform import draw
        g = draw(lambda x: x)(jnp.float32(1.0))
        self.assertIsNotNone(g)
        self.assertGreaterEqual(_count_nodes(g), 1)

    def test_multi_equation_top_level(self):
        # A non-jitted multi-equation function exposes all top-level eqns; the
        # old code rendered only ``fn.eqns[0]``.
        from brainstate.transform import draw

        def f(x):
            a = x + 1.0
            b = x * 2.0
            return a + b   # several top-level eqns, not just eqns[0]

        g = draw(f)(jnp.float32(1.0))
        self.assertIsNotNone(g)
        # 3 primitive eqns (add, mul, add) each contribute at least one node;
        # the buggy single-eqn version produced far fewer.
        self.assertGreaterEqual(_count_nodes(g), 3)

    def test_jitted_identity_builds(self):
        # The jitted identity has exactly one top-level (jit) eqn; ensure that
        # path still builds.
        from brainstate.transform import draw

        @jax.jit
        def f(x):
            return x

        g = draw(f)(jnp.float32(1.0))
        self.assertIsNotNone(g)


@unittest.skipUnless(PYDOT, "pydot not installed")
class TestVisualizeLoops(unittest.TestCase):
    def test_while_builds(self):
        from brainstate.transform import draw
        @jax.jit
        def f(x):
            return jax.lax.while_loop(lambda v: v < 5.0, lambda v: v + 1.0, x)
        g = draw(f)(jnp.float32(0.0))
        self.assertIsNotNone(g)

    def test_while_expanded_builds(self):
        from brainstate.transform import draw
        @jax.jit
        def f(x):
            return jax.lax.while_loop(lambda v: v < 5.0, lambda v: v + 1.0, x)
        g = draw(f, collapse_primitives=False)(jnp.float32(0.0))
        self.assertIsNotNone(g)

    def test_scan_builds(self):
        from brainstate.transform import draw
        @jax.jit
        def f(x):
            def body(c, _):
                return c + 1.0, c
            final, ys = jax.lax.scan(body, x, None, length=3)
            return final
        g = draw(f)(jnp.float32(0.0))
        self.assertIsNotNone(g)


@unittest.skipUnless(PYDOT, "pydot not installed")
class TestVisualizeUnsupported(unittest.TestCase):
    def test_pjit_nested_build(self):
        from brainstate.transform import draw
        @jax.jit
        def inner(x):
            return x * x
        @jax.jit
        def f(x):
            return inner(x) + 1.0
        g = draw(f)(jnp.float32(2.0))
        self.assertIsNotNone(g)

    def test_remat_builds(self):
        # remat2 carries a bare Jaxpr (no `name` param); it must be recognised
        # as an inlinable call and expanded without crashing.
        from brainstate.transform import draw

        def g(x):
            return jnp.sin(x) * 2.0

        def f(x):
            return jax.checkpoint(g)(x) + 1.0

        g_graph = draw(f, collapse_primitives=False)(jnp.float32(1.0))
        self.assertIsNotNone(g_graph)

    def test_dropped_var_helper(self):
        import brainstate.transform._ir_visualize as viz
        self.assertTrue(viz._is_dropped_var("d_"))
        self.assertFalse(viz._is_dropped_var("d"))

    def test_call_jaxpr_of_leaf_returns_none(self):
        import brainstate.transform._ir_visualize as viz

        def f(x):
            return x + 1.0

        jpr = jax.make_jaxpr(f)(jnp.float32(1.0))
        # The single top-level eqn is a leaf primitive (add) -> not a call.
        self.assertIsNone(viz._call_jaxpr_of(jpr.eqns[0]))


@unittest.skipUnless(PYDOT, "pydot not installed")
class TestVisualizeHelpers(unittest.TestCase):
    """Direct unit tests for the small helper functions."""

    def test_is_dropped_var_trailing_underscore(self):
        """_is_dropped_var keys off a trailing underscore in the name."""
        import brainstate.transform._ir_visualize as viz
        self.assertTrue(viz._is_dropped_var("foo_"))
        self.assertFalse(viz._is_dropped_var("foo"))

    def test_call_jaxpr_of_skips_control_flow(self):
        """Dedicated control-flow primitives are never treated as inlinable calls."""
        import brainstate.transform._ir_visualize as viz

        def f(x):
            return jax.lax.cond(x[0] > 0, lambda v: v + 1.0, lambda v: v - 1.0, x)

        jpr = jax.make_jaxpr(f)(jnp.float32([1., 2.]))
        cond_eqn = [e for e in jpr.eqns if e.primitive.name == "cond"][0]
        self.assertIsNone(viz._call_jaxpr_of(cond_eqn))

    def test_call_jaxpr_of_recognises_jit(self):
        """A jit equation exposes its inner jaxpr via _call_jaxpr_of."""
        import brainstate.transform._ir_visualize as viz

        @jax.jit
        def inner(x):
            return x * x

        def f(x):
            return inner(x) + 1.0

        jpr = jax.make_jaxpr(f)(jnp.float32(2.0))
        jit_eqn = [e for e in jpr.eqns if viz.is_not_primitive(e)][0]
        self.assertIsNotNone(viz._call_jaxpr_of(jit_eqn))

    def test_eqn_name_prefers_name_param(self):
        """_eqn_name returns the 'name' param when present, else the primitive name."""
        import brainstate.transform._ir_visualize as viz

        def f(x):
            return x + 1.0

        jpr = jax.make_jaxpr(f)(jnp.float32(1.0))
        add_eqn = jpr.eqns[0]
        # The add primitive carries no 'name' param.
        self.assertEqual(viz._eqn_name(add_eqn), "add")

    def test_get_node_label_with_and_without_avals(self):
        """get_node_label includes the aval type only when show_avals is True."""
        import brainstate.transform._ir_visualize as viz

        def f(x):
            return x + 1.0

        jpr = jax.make_jaxpr(f)(jnp.float32(1.0))
        var = jpr.jaxpr.invars[0]
        label_with = viz.get_node_label(var, True)
        label_without = viz.get_node_label(var, False)
        # With avals, the label appends the short aval type after the var.
        self.assertEqual(label_with, f"{var}: {var.aval.str_short()}")
        self.assertEqual(label_without, str(var))

    def test_contains_non_primitives(self):
        """contains_non_primitives detects nested calls / control flow."""
        import brainstate.transform._ir_visualize as viz

        def leaf(x):
            return x + 1.0

        leaf_eqns = jax.make_jaxpr(leaf)(jnp.float32(1.0)).eqns
        self.assertFalse(viz.contains_non_primitives(leaf_eqns))

        def with_cond(x):
            return jax.lax.cond(x[0] > 0, lambda v: v + 1.0, lambda v: v, x)

        cond_eqns = jax.make_jaxpr(with_cond)(jnp.float32([1., 2.])).eqns
        self.assertTrue(viz.contains_non_primitives(cond_eqns))

    def test_get_const_node_builds(self):
        """get_const_node produces a styled pydot node for a const var."""
        import brainstate.transform._ir_visualize as viz

        def f(x):
            return x + 1.0

        var = jax.make_jaxpr(f)(jnp.float32(1.0)).jaxpr.invars[0]
        node = viz.get_const_node("cid", var, True)
        self.assertIsNotNone(node)


@unittest.skipUnless(PYDOT, "pydot not installed")
class TestExpandPjitPrimitive(unittest.TestCase):
    """Cover expand_pjit_primitive (which is not reached via the public draw path)."""

    def test_expand_pjit_primitive_with_constvars(self):
        """A jaxpr that closes over a constant exercises the const-node argument path."""
        import brainstate.transform._ir_visualize as viz
        import pydot

        c = jnp.arange(3, dtype=jnp.float32)

        def f(x):
            return x + c  # closes over c -> constvar in the jaxpr

        jpr = jax.make_jaxpr(f)(jnp.zeros(3, jnp.float32))
        self.assertGreater(len(jpr.jaxpr.constvars), 0)
        graph, arg_edges, out_nodes, out_edges, n = viz.expand_pjit_primitive(
            jpr.jaxpr, "", 0, True, True, "mygraph"
        )
        self.assertIsInstance(graph, pydot.Subgraph)
        self.assertGreaterEqual(n, 1)

    def test_expand_pjit_primitive_expanded(self):
        """collapse_primitives=False still renders the inner equations."""
        import brainstate.transform._ir_visualize as viz

        def f(x):
            return jnp.sin(x) * 2.0

        jpr = jax.make_jaxpr(f)(jnp.float32(1.0))
        graph, *_ = viz.expand_pjit_primitive(jpr.jaxpr, "", 0, False, True, "g")
        self.assertIsNotNone(graph)

    def test_expand_pjit_primitive_no_phantom_parent_edges(self):
        """The standalone pjit expansion must not return parent-linking edges or
        nodes keyed off its own vars: it has no enclosing equation, so such edges
        point at non-existent ``{parent_id}_{var}`` nodes (audit item 4)."""
        import brainstate.transform._ir_visualize as viz

        def f(x):
            return jnp.sin(x) * 2.0

        jpr = jax.make_jaxpr(f)(jnp.float32(1.0))
        graph, arg_edges, out_nodes, out_edges, n = viz.expand_pjit_primitive(
            jpr.jaxpr, "parent", 0, False, True, "g"
        )
        self.assertIsNotNone(graph)
        # No phantom parent linkage.
        self.assertEqual(list(arg_edges), [])
        self.assertEqual(list(out_nodes), [])
        self.assertEqual(list(out_edges), [])


@unittest.skipUnless(PYDOT, "pydot not installed")
class TestExpandNonPrimitiveErrors(unittest.TestCase):
    """expand_non_primitive raises for non-inlinable leaf primitives."""

    def test_unsupported_primitive_raises(self):
        import brainstate.transform._ir_visualize as viz
        from brainstate.transform._ir_utils import UnsupportedPrimitiveError

        def f(x):
            return x + 1.0

        add_eqn = jax.make_jaxpr(f)(jnp.float32(1.0)).eqns[0]
        with self.assertRaises(UnsupportedPrimitiveError):
            viz.expand_non_primitive(add_eqn, "", 0, True, True)


@unittest.skipUnless(PYDOT, "pydot not installed")
class TestVisualizeCondEmptyBranches(unittest.TestCase):
    """cond with empty (selection-only) branches: the len(branch.eqns)==0 path."""

    def _select_cond(self):
        # Branches that merely select one of their inputs trace to empty
        # branch jaxprs (zero eqns) whose outvar is one of the invars.
        def f(pred, x, y):
            return jax.lax.cond(pred, lambda a, b: a, lambda a, b: b, x, y)
        return f

    def test_empty_branches_collapsed(self):
        """collapse_primitives=True renders empty branches as single nodes."""
        from brainstate.transform import draw
        g = draw(self._select_cond(), collapse_primitives=True)(
            True, jnp.float32([1., 2.]), jnp.float32([3., 4.])
        )
        self.assertIsNotNone(g)

    def test_empty_branches_expanded(self):
        """collapse_primitives=False renders empty branches as subgraphs."""
        from brainstate.transform import draw
        g = draw(self._select_cond(), collapse_primitives=False)(
            True, jnp.float32([1., 2.]), jnp.float32([3., 4.])
        )
        self.assertIsNotNone(g)


@unittest.skipUnless(PYDOT, "pydot not installed")
class TestVisualizeCondVariants(unittest.TestCase):
    """Exercise the single-eqn and multi-eqn cond-branch rendering paths."""

    def test_single_eqn_branches_expanded(self):
        """Single-primitive branches with collapse_primitives=False."""
        from brainstate.transform import draw

        def f(pred, x):
            return jax.lax.cond(pred, lambda v: v + 1.0, lambda v: v * 2.0, x)

        g = draw(f, collapse_primitives=False)(True, jnp.float32([1., 2.]))
        self.assertIsNotNone(g)

    def test_single_eqn_branches_collapsed(self):
        """Single-primitive branches with collapse_primitives=True."""
        from brainstate.transform import draw

        def f(pred, x):
            return jax.lax.cond(pred, lambda v: v + 1.0, lambda v: v * 2.0, x)

        g = draw(f, collapse_primitives=True)(True, jnp.float32([1., 2.]))
        self.assertIsNotNone(g)

    def test_multi_eqn_branches_collapsed(self):
        """Multi-primitive branches with collapse_primitives=True (collapsed node)."""
        from brainstate.transform import draw

        def f(pred, x):
            return jax.lax.cond(
                pred,
                lambda v: jnp.sum(v * v) + 1.0,
                lambda v: jnp.sum(v) - 1.0,
                x,
            )

        g = draw(f, collapse_primitives=True)(True, jnp.float32([1., 2., 3.]))
        self.assertIsNotNone(g)

    def test_branches_expanded_subgraph(self):
        """Multi-primitive branches with collapse_primitives=False (full subgraphs)."""
        from brainstate.transform import draw

        def f(pred, x):
            return jax.lax.cond(
                pred,
                lambda v: jnp.sum(v * v) + 1.0,
                lambda v: jnp.sum(v) - 1.0,
                x,
            )

        g = draw(f, collapse_primitives=False)(True, jnp.float32([1., 2., 3.]))
        self.assertIsNotNone(g)


@unittest.skipUnless(PYDOT, "pydot not installed")
class TestVisualizeScanExpanded(unittest.TestCase):
    """Scan body rendered as a subgraph (collapse_primitives=False)."""

    def test_scan_expanded_builds(self):
        """A scan rendered with collapse_primitives=False expands its body subgraph."""
        from brainstate.transform import draw

        def f(x):
            def body(c, inp):
                return c + inp, c * 2.0
            final, ys = jax.lax.scan(body, x, jnp.float32([1., 2., 3.]))
            return final, ys

        g = draw(f, collapse_primitives=False)(jnp.float32(0.0))
        self.assertIsNotNone(g)

    def test_scan_with_const_argument_builds(self):
        """A scan with a loop-invariant const exercises the n_const grouping path."""
        from brainstate.transform import draw

        def f(x, w):
            def body(c, inp):
                return c + w * inp, c  # w is a scan const
            final, ys = jax.lax.scan(body, x, jnp.float32([1., 2., 3.]))
            return final, ys

        g = draw(f, collapse_primitives=False)(jnp.float32(0.0), jnp.float32(2.0))
        self.assertIsNotNone(g)

    def test_scan_with_nonprimitive_body_builds(self):
        """A scan whose body contains a jit is never collapsed."""
        from brainstate.transform import draw

        @jax.jit
        def step(c):
            return c + 1.0

        def f(x):
            def body(c, _):
                return step(c), c
            final, ys = jax.lax.scan(body, x, None, length=3)
            return final, ys

        g = draw(f, collapse_primitives=True)(jnp.float32(0.0))
        self.assertIsNotNone(g)


@unittest.skipUnless(PYDOT, "pydot not installed")
class TestVisualizeWhileExpanded(unittest.TestCase):
    """While cond/body rendered as subgraphs (collapse_primitives=False)."""

    def test_while_nonprimitive_body_builds(self):
        """A while loop whose body contains a jit forces subgraph expansion."""
        from brainstate.transform import draw

        @jax.jit
        def incr(v):
            return v + 1.0

        def f(x):
            return jax.lax.while_loop(lambda v: v < 5.0, lambda v: incr(v), x)

        g = draw(f, collapse_primitives=True)(jnp.float32(0.0))
        self.assertIsNotNone(g)

    def test_while_expanded_with_multi_carry(self):
        """While with a multi-element carry, fully expanded."""
        from brainstate.transform import draw

        def f(x):
            def cond(carry):
                i, acc = carry
                return i < 3.0

            def body(carry):
                i, acc = carry
                return i + 1.0, acc + i

            return jax.lax.while_loop(cond, body, (x, x))

        g = draw(f, collapse_primitives=False)(jnp.float32(0.0))
        self.assertIsNotNone(g)

    def test_while_expanded_jit_body_with_passthrough(self):
        """Expanded while whose body holds a jit (subgraph) and forwards a carry (id-edge)."""
        from brainstate.transform import draw

        @jax.jit
        def incr(v):
            return v + 1.0

        def f(x, y):
            def cond(c):
                i, acc = c
                return i < 3.0

            def body(c):
                i, acc = c
                # incr is a jit -> renders as a subgraph; acc is forwarded.
                return incr(i), acc

            return jax.lax.while_loop(cond, body, (x, y))

        g = draw(f, collapse_primitives=False)(jnp.float32(0.0), jnp.float32(5.0))
        self.assertIsNotNone(g)


@unittest.skipUnless(PYDOT, "pydot not installed")
class TestVisualizeReturnsArg(unittest.TestCase):
    """A function that returns one of its inputs exercises the id-edge path."""

    def test_identity_through_jit_builds(self):
        """Returning an argument through a jit hits get_outputs' id_edges branch."""
        from brainstate.transform import draw

        @jax.jit
        def inner(x, y):
            # Return the first argument unchanged; y is consumed so it survives.
            return x, y + 1.0

        def f(x, y):
            a, b = inner(x, y)
            return a + b

        g = draw(f, collapse_primitives=False)(jnp.float32(1.0), jnp.float32(2.0))
        self.assertIsNotNone(g)


@unittest.skipUnless(PYDOT, "pydot not installed")
class TestViewPydot(unittest.TestCase):
    """view_pydot displays a PNG via IPython (mocked to avoid graphviz)."""

    def test_view_pydot_calls_display(self):
        """view_pydot calls create_png and hands the image to IPython.display."""
        from brainstate.transform import draw, view_pydot

        @jax.jit
        def f(x):
            return x + 1.0

        g = draw(f)(jnp.float32(1.0))

        calls = {}

        # Stub create_png so we never invoke the (absent) graphviz binary.
        g.create_png = lambda: b"\x89PNG-fake"

        fake_display = types.ModuleType("IPython.display")

        class _Image:
            def __init__(self, data):
                calls["image_data"] = data

        def _display(obj):
            calls["displayed"] = obj

        fake_display.Image = _Image
        fake_display.display = _display
        fake_ipython = types.ModuleType("IPython")
        fake_ipython.display = fake_display

        saved_ip = sys.modules.get("IPython")
        saved_disp = sys.modules.get("IPython.display")
        sys.modules["IPython"] = fake_ipython
        sys.modules["IPython.display"] = fake_display
        try:
            view_pydot(g)
        finally:
            if saved_ip is not None:
                sys.modules["IPython"] = saved_ip
            else:
                sys.modules.pop("IPython", None)
            if saved_disp is not None:
                sys.modules["IPython.display"] = saved_disp
            else:
                sys.modules.pop("IPython.display", None)

        self.assertEqual(calls["image_data"], b"\x89PNG-fake")
        self.assertIn("displayed", calls)


@unittest.skipUnless(PYDOT, "pydot not installed")
class TestDrawDotGraphEmpty(unittest.TestCase):
    """draw_dot_graph handles an empty top-level jaxpr (no eqns)."""

    def test_empty_jaxpr_with_literal_and_dropped_outputs(self):
        """An empty jaxpr renders only its in/out nodes without crashing."""
        from brainstate.transform import draw_dot_graph

        # Returning a constant alongside the input yields a top-level jaxpr
        # whose only structure is forwarded in/out vars (zero eqns is possible
        # for pure pass-throughs).
        jpr = jax.make_jaxpr(lambda x: x)(jnp.float32(1.0))
        g = draw_dot_graph(jpr, True, True)
        self.assertIsNotNone(g)


@unittest.skipUnless(PYDOT, "pydot not installed")
class TestLiteralCallArgument(unittest.TestCase):
    """Regression for M37: a literal passed into an inlinable call.

    ``get_arguments`` used to test ``isinstance(var, Literal)`` on the callee's
    *formal* parameter (never a literal) and unconditionally draw an edge from
    ``{parent}_{Literal(...)}`` -- a node that is never created -- dropping the
    literal value and leaving a dangling edge to a phantom node.
    """

    def test_literal_call_argument_node_and_no_dangling_edge(self):
        from brainstate.transform import draw

        @jax.jit
        def inner(a, b):
            return a + b

        def f(x):
            return inner(x, 3.0)  # 3.0 is a Literal passed positionally

        g = draw(f, collapse_primitives=False)(jnp.float32(1.0))

        # The literal value must be rendered as a declared node (orange literal
        # box), not lost. Pre-fix no node anywhere carried the literal.
        literal_nodes = [
            nd for nd in _all_nodes(g)
            if nd.get_label() and "Literal(3.0)" in str(nd.get_label())
        ]
        self.assertEqual(
            len(literal_nodes), 1,
            f"expected exactly one declared Literal(3.0) node, got "
            f"{[nd.get_name() for nd in literal_nodes]}",
        )

        # No edge may reference an undeclared source. Pre-fix the edge
        # ``_Literal(3.0) -> inner_0_<var>`` had a phantom source.
        self.assertEqual(_undeclared_edge_sources(g), [])

        # The literal node feeds the formal-parameter node via a *local* edge
        # (inside the argument cluster); its declared name keeps the ``_lit``
        # marker (pre-fix the ``:aval`` port stripped it, colliding with the
        # parameter node).
        lit_name = literal_nodes[0].get_name()
        self.assertTrue(
            lit_name.endswith("_lit"),
            f"literal node name should end with '_lit', got {lit_name!r}",
        )


@unittest.skipUnless(PYDOT, "pydot not installed")
class TestTopLevelBoundaryNodes(unittest.TestCase):
    """Regression for L19: top-level input/output boundary nodes.

    On the non-empty-jaxpr path ``draw_dot_graph`` used to create no node for
    the function's input arguments (they were only referenced as edge sources)
    and rendered the final outputs with blue VAR styling rather than red OUT
    styling.
    """

    def test_input_node_declared_and_styled(self):
        from brainstate.transform import draw, _ir_visualize as viz

        def f(x):
            return x + 1.0  # one top-level eqn -> non-empty path

        g = draw(f)(jnp.float32(1.0))

        # Every edge source must be a declared node. Pre-fix the top-level
        # input ``_Var(x)`` was referenced by the add's in-edge but never
        # declared, so graphviz fabricated a phantom node.
        self.assertEqual(_undeclared_edge_sources(g), [])

        # The input argument is declared with green IN_ARG styling.
        green = viz.IN_ARG_STYLING["color"]
        red = viz.OUT_ARG_STYLING["color"]
        colors = {nd.get_color() for nd in _all_nodes(g)}
        self.assertIn(green, colors, "expected a green top-level input node")
        # The final output is styled as an output (red), not a plain VAR.
        self.assertIn(red, colors, "expected a red top-level output node")

    def test_multi_equation_no_undeclared_sources(self):
        from brainstate.transform import draw

        def f(x):
            a = x + 1.0
            b = x * 2.0
            return a + b

        g = draw(f)(jnp.float32(1.0))
        self.assertEqual(_undeclared_edge_sources(g), [])

    def test_input_returned_verbatim_recoloured(self):
        """An input that is also an output is recoloured red (no duplicate)."""
        from brainstate.transform import draw, _ir_visualize as viz

        def f(x, y):
            # x + 1.0 is an eqn (non-empty path); y is returned verbatim and is
            # therefore both an input and an output with no producing eqn.
            return x + 1.0, y

        g = draw(f)(jnp.float32(1.0), jnp.float32(2.0))

        # No phantom sources and no duplicate declaration for the verbatim var.
        self.assertEqual(_undeclared_edge_sources(g), [])
        from collections import Counter
        name_counts = Counter(nd.get_name() for nd in _all_nodes(g))
        dups = {k: c for k, c in name_counts.items() if c > 1}
        self.assertEqual(dups, {}, f"unexpected duplicate nodes: {dups}")
        # The verbatim-returned input ends up red (OUT styling).
        red = viz.OUT_ARG_STYLING["color"]
        self.assertIn(red, {nd.get_color() for nd in _all_nodes(g)})


@unittest.skipUnless(PYDOT, "pydot not installed")
class TestScanBodyOutputDedup(unittest.TestCase):
    """Regression for L20: scan body output-dedup + duplicate scan nodes.

    The dedup filter compared an unstripped key set (``...:float32[]``) against
    pydot's port-stripped node names, so it never matched and scan output nodes
    were declared twice.
    """

    def test_no_duplicate_scan_nodes(self):
        from brainstate.transform import draw
        from collections import Counter

        def f(x):
            def body(c, _):
                return c + 1.0, c  # carry returned verbatim -> exercises _out path
            final, ys = jax.lax.scan(body, x, None, length=3)
            return final

        g = draw(f, collapse_primitives=False)(jnp.float32(0.0))

        scan_names = [
            nd.get_name() for nd in _all_nodes(g)
            if nd.get_name().startswith("scan_")
        ]
        dups = {k: c for k, c in Counter(scan_names).items() if c > 1}
        self.assertEqual(
            dups, {},
            f"scan_* node names must be unique, got duplicates: {dups}",
        )

    def test_no_self_loop_edges(self):
        """The ``_out`` id-edge must connect distinct endpoints, not self-loop."""
        from brainstate.transform import draw

        def f(x):
            def body(c, _):
                return c + 1.0, c
            final, ys = jax.lax.scan(body, x, None, length=3)
            return final

        g = draw(f, collapse_primitives=False)(jnp.float32(0.0))
        self_loops = [
            (e.get_source(), e.get_destination())
            for e in _all_edges(g)
            if _base(e.get_source()) == _base(e.get_destination())
        ]
        self.assertEqual(self_loops, [], f"unexpected self-loop edges: {self_loops}")


@unittest.skipUnless(PYDOT, "pydot not installed")
class TestScanArgumentClusterLabels(unittest.TestCase):
    """Regression for L21: scan argument subgraphs mislabel consts/carry.

    The const-holding cluster was labeled 'init' and the carry-holding cluster
    'consts' -- swapped relative to JAX's [consts, carry, xs] invar order.
    """

    def test_const_and_carry_clusters_labelled_correctly(self):
        from brainstate.transform import draw

        def f(x, w):
            def body(c, inp):
                return c + w * inp, c  # w is a scan const, c is the carry
            final, ys = jax.lax.scan(body, x, jnp.float32([1., 2., 3.]))
            return final, ys

        g = draw(f, collapse_primitives=False)(jnp.float32(0.0), jnp.float32(2.0))

        # The argument-side const cluster id ends in '_const'; the carry cluster
        # keeps the historical '_init' id (the '_carry' id is taken by the scan
        # output cluster). Distinguish them by id, then assert their labels.
        const_label = None
        carry_label = None
        for s in _all_subgraphs(g):
            name = s.get_name() or ""
            if name == "cluster_scan_0_const":
                const_label = s.get_label()
            elif name == "cluster_scan_0_init":
                carry_label = s.get_label()

        self.assertEqual(
            const_label, "consts",
            "the cluster holding the scan consts must be labeled 'consts'",
        )
        self.assertEqual(
            carry_label, "carry",
            "the cluster holding the scan carry must be labeled 'carry'",
        )


@unittest.skipUnless(PYDOT, "pydot not installed")
class TestCondLiteralOperand(unittest.TestCase):
    """Regression for L22: literal non-predicate operand to ``cond``.

    ``get_conditional`` unconditionally drew a parent edge for every operand,
    so a literal operand produced an edge from ``{parent}_{Literal(...)}`` -- a
    node that is never declared -- fabricating a phantom node.
    """

    def _cond_fn(self):
        def f(pred, x):
            return jax.lax.cond(
                pred, lambda a, b: a + b, lambda a, b: a - b, x, 5.0
            )
        return f

    def test_no_phantom_literal_source_collapsed(self):
        from brainstate.transform import draw
        g = draw(self._cond_fn(), collapse_primitives=True)(True, jnp.float32(1.0))
        self._check(g)

    def test_no_phantom_literal_source_expanded(self):
        from brainstate.transform import draw
        g = draw(self._cond_fn(), collapse_primitives=False)(True, jnp.float32(1.0))
        self._check(g)

    def _check(self, g):
        # No undeclared literal edge source (the phantom). Pre-fix there was an
        # edge ``_Literal(5.0) -> _cond_..._Literal(5.0)`` with no source node.
        undeclared = _undeclared_edge_sources(g)
        literal_phantoms = [s for s in undeclared if "Literal" in s]
        self.assertEqual(
            literal_phantoms, [],
            f"literal operand left a phantom edge source: {literal_phantoms}",
        )
        # The literal destination node IS declared (orange literal box).
        declared_literal = [
            nd for nd in _all_nodes(g)
            if "Literal(5.0)" in nd.get_name()
        ]
        self.assertTrue(
            declared_literal,
            "expected a declared cond literal destination node",
        )


if __name__ == '__main__':
    unittest.main()
