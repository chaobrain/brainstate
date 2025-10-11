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

# -*- coding: utf-8 -*-


import unittest

import numpy as np

import brainstate


class TestModuleGroup(unittest.TestCase):
    def test_initialization(self):
        group = brainstate.nn.DynamicsGroup()
        self.assertIsInstance(group, brainstate.nn.DynamicsGroup)


class TestDynamics(unittest.TestCase):
    def test_initialization(self):
        dyn = brainstate.nn.Dynamics(in_size=10)
        self.assertIsInstance(dyn, brainstate.nn.Dynamics)
        self.assertEqual(dyn.in_size, (10,))
        self.assertEqual(dyn.out_size, (10,))

    def test_size_validation(self):
        with self.assertRaises(ValueError):
            brainstate.nn.Dynamics(in_size=[])
        with self.assertRaises(ValueError):
            brainstate.nn.Dynamics(in_size="invalid")

    def test_varshape(self):
        dyn = brainstate.nn.Dynamics(in_size=(2, 3))
        self.assertEqual(dyn.varshape, (2, 3))
        dyn = brainstate.nn.Dynamics(in_size=(2, 3))
        self.assertEqual(dyn.varshape, (2, 3))


if __name__ == '__main__':
    unittest.main()
