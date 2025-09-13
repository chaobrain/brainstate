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
import jax
import jax.numpy as jnp
import numpy as np

import brainstate


class TestMixin(unittest.TestCase):
    def test_mixin(self):
        self.assertTrue(brainstate.mixin.Mixin)
        self.assertTrue(brainstate.mixin.ParamDesc)
        self.assertTrue(brainstate.mixin.ParamDescriber)
        self.assertTrue(brainstate.mixin.JointTypes)
        self.assertTrue(brainstate.mixin.OneOfTypes)
        self.assertTrue(brainstate.mixin.Mode)
        self.assertTrue(brainstate.mixin.Batching)
        self.assertTrue(brainstate.mixin.Training)


class TestMode(unittest.TestCase):
    def test_JointMode(self):
        a = brainstate.mixin.JointMode(brainstate.mixin.Batching(), brainstate.mixin.Training())
        self.assertTrue(a.is_a(brainstate.mixin.JointTypes[brainstate.mixin.Batching, brainstate.mixin.Training]))
        self.assertTrue(a.has(brainstate.mixin.Batching))
        self.assertTrue(a.has(brainstate.mixin.Training))
        b = brainstate.mixin.JointMode(brainstate.mixin.Batching())
        self.assertTrue(b.is_a(brainstate.mixin.JointTypes[brainstate.mixin.Batching]))
        self.assertTrue(b.is_a(brainstate.mixin.Batching))
        self.assertTrue(b.has(brainstate.mixin.Batching))

    def test_Training(self):
        a = brainstate.mixin.Training()
        self.assertTrue(a.is_a(brainstate.mixin.Training))
        self.assertTrue(a.is_a(brainstate.mixin.JointTypes[brainstate.mixin.Training]))
        self.assertTrue(a.has(brainstate.mixin.Training))
        self.assertTrue(a.has(brainstate.mixin.JointTypes[brainstate.mixin.Training]))
        self.assertFalse(a.is_a(brainstate.mixin.Batching))
        self.assertFalse(a.has(brainstate.mixin.Batching))

    def test_Batching(self):
        a = brainstate.mixin.Batching()
        self.assertTrue(a.is_a(brainstate.mixin.Batching))
        self.assertTrue(a.is_a(brainstate.mixin.JointTypes[brainstate.mixin.Batching]))
        self.assertTrue(a.has(brainstate.mixin.Batching))
        self.assertTrue(a.has(brainstate.mixin.JointTypes[brainstate.mixin.Batching]))

        self.assertFalse(a.is_a(brainstate.mixin.Training))
        self.assertFalse(a.has(brainstate.mixin.Training))

    def test_Mode(self):
        a = brainstate.mixin.Mode()
        self.assertTrue(a.is_a(brainstate.mixin.Mode))
        self.assertTrue(a.is_a(brainstate.mixin.JointTypes[brainstate.mixin.Mode]))
        self.assertTrue(a.has(brainstate.mixin.Mode))
        self.assertTrue(a.has(brainstate.mixin.JointTypes[brainstate.mixin.Mode]))

        self.assertFalse(a.is_a(brainstate.mixin.Training))
        self.assertFalse(a.has(brainstate.mixin.Training))
        self.assertFalse(a.is_a(brainstate.mixin.Batching))
        self.assertFalse(a.has(brainstate.mixin.Batching))

