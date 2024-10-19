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

from absl.testing import absltest

import brainstate as bst


class TestGraphUtils(absltest.TestCase):
  def test_split_merge_context(self):
    m = bst.nn.Linear(2, 3, )
    with bst.graph.split_context() as ctx:
      graphdef1, state1 = ctx.split(m)
      graphdef2, state2 = ctx.split(m)
      pass

    self.assertFalse(hasattr(ctx, 'ref_index'))
    self.assertIsInstance(graphdef1, bst.graph.NodeDef)
    self.assertIsInstance(graphdef2, bst.graph.NodeRef)
    self.assertLen(state1.to_flat(), 1)
    self.assertLen(state2.to_flat(), 0)

    with bst.graph.merge_context() as ctx:
      m1 = ctx.merge(graphdef1, state1)
      m2 = ctx.merge(graphdef2, state2)

    self.assertIs(m1, m2)
    self.assertFalse(hasattr(ctx, 'index_ref'))

  def test_split_merge_context_nested(self):
    m2 = bst.nn.Linear(2, 3, )
    m1 = bst.nn.Sequential(m2)
    with bst.graph.split_context() as ctx:
      graphdef1, state1 = ctx.split(m1)
      graphdef2, state2 = ctx.split(m2)

    self.assertIsInstance(graphdef1, bst.graph.NodeDef)
    self.assertIsInstance(graphdef2, bst.graph.NodeRef)
    self.assertLen(state1.to_flat(), 1)
    self.assertLen(state2.to_flat(), 0)

    with bst.graph.merge_context() as ctx:
      m1 = ctx.merge(graphdef1, state1)
      m2 = ctx.merge(graphdef2, state2)

    self.assertIs(m2, m1.layers[0])
    self.assertFalse(hasattr(ctx, 'index_ref'))


if __name__ == '__main__':
  absltest.main()
