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
import unittest

import brainstate as bst


class TestSequential(unittest.TestCase):
  def test1(self):
    s = bst.graph.Sequential(bst.nn.Linear(1, 2),
                             bst.nn.Linear(2, 3))
    graphdef, states = bst.graph.treefy_split(s)
    print(states)
    self.assertTrue(len(states.to_flat()) == 2)


class TestStateRetrieve(unittest.TestCase):
  def test_list_of_states_1(self):
    class Model(bst.graph.Node):
      def __init__(self):
        self.a = [1, 2, 3]
        self.b = [bst.State(1), bst.State(2), bst.State(3)]

    m = Model()
    graphdef, states = bst.graph.treefy_split(m)
    print(states.to_flat())
    self.assertTrue(len(states.to_flat()) == 3)

  def test_list_of_states_2(self):
    class Model(bst.graph.Node):
      def __init__(self):
        self.a = [1, 2, 3]
        self.b = [bst.State(1), [bst.State(2), bst.State(3)]]

    m = Model()
    graphdef, states = bst.graph.treefy_split(m)
    print(states.to_flat())
    self.assertTrue(len(states.to_flat()) == 3)

  def test_list_of_node_1(self):
    class Model(bst.graph.Node):
      def __init__(self):
        self.a = [1, 2, 3]
        self.b = [bst.nn.Linear(1, 2), bst.nn.Linear(2, 3)]

    m = Model()
    graphdef, states = bst.graph.treefy_split(m)
    print(states.to_flat())
    self.assertTrue(len(states.to_flat()) == 2)

  def test_list_of_node_2(self):
    class Model(bst.graph.Node):
      def __init__(self):
        self.a = [1, 2, 3]
        self.b = [bst.nn.Linear(1, 2), [bst.nn.Linear(2, 3)], (bst.nn.Linear(3, 4), bst.nn.Linear(4, 5))]

    m = Model()
    graphdef, states = bst.graph.treefy_split(m)
    print(states.to_flat())
    self.assertTrue(len(states.to_flat()) == 4)
