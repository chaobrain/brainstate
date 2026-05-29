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


if __name__ == '__main__':
    absltest.main()
