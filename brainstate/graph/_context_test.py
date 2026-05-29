# Copyright 2024 BDP Ecosystem Limited. All Rights Reserved.
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

from __future__ import annotations

import jax.numpy as jnp

import brainstate
from absl.testing import absltest


class TestGraphUtils(absltest.TestCase):
    def test_split_merge_context(self):
        m = brainstate.nn.Linear(2, 3, )
        with brainstate.graph.split_context() as (ctx, index_ref):
            graphdef1, state1 = ctx.treefy_split(m)
            graphdef2, state2 = ctx.treefy_split(m)
            pass

        self.assertFalse(hasattr(ctx, 'ref_index'))
        # graphdef1 is a full definition; graphdef2 is a pure back-reference to
        # the same node (shared across the split_context), so it carries no
        # node specs of its own.
        self.assertIsInstance(graphdef1.root, brainstate.graph.NodeEdge)
        self.assertGreaterEqual(len(graphdef1.node_specs), 1)
        self.assertEqual(graphdef1.type, brainstate.nn.Linear)
        self.assertIsInstance(graphdef2.root, brainstate.graph.NodeEdge)
        self.assertEqual(len(graphdef2.node_specs), 0)
        self.assertLen(state1.to_flat(), 1)
        self.assertLen(state2.to_flat(), 0)

        with brainstate.graph.merge_context() as (ctx, index_ref):
            m1 = ctx.treefy_merge(graphdef1, state1)
            m2 = ctx.treefy_merge(graphdef2, state2)

        self.assertIs(m1, m2)
        self.assertFalse(hasattr(ctx, 'index_ref'))

    def test_split_merge_context_nested(self):
        m2 = brainstate.nn.Linear(2, 3)
        m1 = brainstate.nn.Sequential(m2)
        with brainstate.graph.split_context() as (ctx, index_ref):
            graphdef1, state1 = ctx.treefy_split(m1)
            graphdef2, state2 = ctx.treefy_split(m2)

        self.assertIsInstance(graphdef1.root, brainstate.graph.NodeEdge)
        self.assertGreaterEqual(len(graphdef1.node_specs), 1)
        self.assertIsInstance(graphdef2.root, brainstate.graph.NodeEdge)
        self.assertEqual(len(graphdef2.node_specs), 0)   # back-reference
        self.assertLen(state1.to_flat(), 1)
        self.assertLen(state2.to_flat(), 0)

        with brainstate.graph.merge_context() as (ctx, index_ref):
            m1 = ctx.treefy_merge(graphdef1, state1)
            m2 = ctx.treefy_merge(graphdef2, state2)

        self.assertIs(m2, m1.layers[0])
        self.assertFalse(hasattr(ctx, 'index_ref'))


class TestSplitMergeContextIdentity(absltest.TestCase):
    """``treefy_split`` / ``treefy_merge`` preserve shared-node identity."""

    def test_merge_restores_shared_state_identity(self):
        """A state shared across slots remains aliased through split + merge."""
        shared = brainstate.ParamState(jnp.ones((2,)))
        g = {"a": shared, "b": shared}
        graphdef, refs = brainstate.graph.treefy_split(g)
        rebuilt = brainstate.graph.treefy_merge(graphdef, refs)
        self.assertIs(rebuilt["a"], rebuilt["b"])

    def test_split_then_merge_value_roundtrip(self):
        """``treefy_split`` / ``treefy_merge`` round-trips values and statics."""
        g = [brainstate.ParamState(jnp.arange(3.0)), 7, brainstate.ParamState(jnp.zeros((2,)))]
        graphdef, refs = brainstate.graph.treefy_split(g)
        rebuilt = brainstate.graph.treefy_merge(graphdef, refs)
        self.assertTrue(bool(jnp.allclose(rebuilt[0].value, jnp.arange(3.0))))
        self.assertEqual(rebuilt[1], 7)
        self.assertTrue(bool(jnp.allclose(rebuilt[2].value, jnp.zeros((2,)))))

    def test_context_managers_clean_up_state(self):
        """The context managers tear down their thread-local reference tables."""
        from brainstate.graph._context import split_context, merge_context

        with split_context() as (ctx, index_ref):
            self.assertIsNotNone(ctx)
        self.assertFalse(hasattr(ctx, 'ref_index'))

        with merge_context() as (mctx, idx):
            self.assertIsNotNone(mctx)
        self.assertFalse(hasattr(mctx, 'index_ref'))


if __name__ == '__main__':
    absltest.main()
