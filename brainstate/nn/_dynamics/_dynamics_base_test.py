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

# -*- coding: utf-8 -*-


import unittest

import numpy as np

import brainstate as bst


class TestModuleGroup(unittest.TestCase):
  def test_initialization(self):
    group = bst.nn.ModuleGroup()
    self.assertIsInstance(group, bst.nn.ModuleGroup)


class TestProjection(unittest.TestCase):
  def test_initialization(self):
    proj = bst.nn.Projection()
    self.assertIsInstance(proj, bst.nn.Projection)

  def test_update_not_implemented(self):
    proj = bst.nn.Projection()
    with self.assertRaises(ValueError):
      proj.update()


class TestDynamics(unittest.TestCase):
  def test_initialization(self):
    dyn = bst.nn.Dynamics(size=10)
    self.assertIsInstance(dyn, bst.nn.Dynamics)
    self.assertEqual(dyn.size, (10,))
    self.assertEqual(dyn.num, 10)

  def test_size_validation(self):
    with self.assertRaises(ValueError):
      bst.nn.Dynamics(size=[])
    with self.assertRaises(ValueError):
      bst.nn.Dynamics(size="invalid")

  def test_input_handling(self):
    dyn = bst.nn.Dynamics(size=10)
    dyn.add_current_input("test_current", lambda: np.random.rand(10))
    dyn.add_delta_input("test_delta", lambda: np.random.rand(10))

    self.assertIn("test_current", dyn.current_inputs)
    self.assertIn("test_delta", dyn.delta_inputs)

  def test_duplicate_input_key(self):
    dyn = bst.nn.Dynamics(size=10)
    dyn.add_current_input("test", lambda: np.random.rand(10))
    with self.assertRaises(ValueError):
      dyn.add_current_input("test", lambda: np.random.rand(10))

  def test_varshape(self):
    dyn = bst.nn.Dynamics(size=(2, 3), keep_size=True)
    self.assertEqual(dyn.varshape, (2, 3))
    dyn = bst.nn.Dynamics(size=(2, 3))
    self.assertEqual(dyn.varshape, (6,))

  def test_keep_size(self):
    dyn_keep = bst.nn.Dynamics(size=(2, 3), keep_size=True)
    dyn_flatten = bst.nn.Dynamics(size=(2, 3), keep_size=False)

    self.assertEqual(dyn_keep.in_size, (2, 3))
    self.assertEqual(dyn_keep.out_size, (2, 3))
    self.assertEqual(dyn_flatten.in_size, (6,))
    self.assertEqual(dyn_flatten.out_size, (6,))


if __name__ == '__main__':
  unittest.main()