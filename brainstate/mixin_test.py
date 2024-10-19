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

import unittest

import brainstate as bc


class TestMixin(unittest.TestCase):
  def test_mixin(self):
    self.assertTrue(bc.mixin.Mixin)
    self.assertTrue(bc.mixin.DelayedInit)
    self.assertTrue(bc.mixin.DelayedInitializer)
    self.assertTrue(bc.mixin.JointTypes)
    self.assertTrue(bc.mixin.OneOfTypes)
    self.assertTrue(bc.mixin.Mode)
    self.assertTrue(bc.mixin.Batching)
    self.assertTrue(bc.mixin.Training)


class TestMode(unittest.TestCase):
  def test_JointMode(self):
    a = bc.mixin.JointMode(bc.mixin.Batching(), bc.mixin.Training())
    self.assertTrue(a.is_a(bc.mixin.JointTypes[bc.mixin.Batching, bc.mixin.Training]))
    self.assertTrue(a.has(bc.mixin.Batching))
    self.assertTrue(a.has(bc.mixin.Training))
    b = bc.mixin.JointMode(bc.mixin.Batching())
    self.assertTrue(b.is_a(bc.mixin.JointTypes[bc.mixin.Batching]))
    self.assertTrue(b.is_a(bc.mixin.Batching))
    self.assertTrue(b.has(bc.mixin.Batching))

  def test_Training(self):
    a = bc.mixin.Training()
    self.assertTrue(a.is_a(bc.mixin.Training))
    self.assertTrue(a.is_a(bc.mixin.JointTypes[bc.mixin.Training]))
    self.assertTrue(a.has(bc.mixin.Training))
    self.assertTrue(a.has(bc.mixin.JointTypes[bc.mixin.Training]))
    self.assertFalse(a.is_a(bc.mixin.Batching))
    self.assertFalse(a.has(bc.mixin.Batching))

  def test_Batching(self):
    a = bc.mixin.Batching()
    self.assertTrue(a.is_a(bc.mixin.Batching))
    self.assertTrue(a.is_a(bc.mixin.JointTypes[bc.mixin.Batching]))
    self.assertTrue(a.has(bc.mixin.Batching))
    self.assertTrue(a.has(bc.mixin.JointTypes[bc.mixin.Batching]))

    self.assertFalse(a.is_a(bc.mixin.Training))
    self.assertFalse(a.has(bc.mixin.Training))

  def test_Mode(self):
    a = bc.mixin.Mode()
    self.assertTrue(a.is_a(bc.mixin.Mode))
    self.assertTrue(a.is_a(bc.mixin.JointTypes[bc.mixin.Mode]))
    self.assertTrue(a.has(bc.mixin.Mode))
    self.assertTrue(a.has(bc.mixin.JointTypes[bc.mixin.Mode]))

    self.assertFalse(a.is_a(bc.mixin.Training))
    self.assertFalse(a.has(bc.mixin.Training))
    self.assertFalse(a.is_a(bc.mixin.Batching))
    self.assertFalse(a.has(bc.mixin.Batching))
