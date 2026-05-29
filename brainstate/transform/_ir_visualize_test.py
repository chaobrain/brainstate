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


if __name__ == '__main__':
    unittest.main()
