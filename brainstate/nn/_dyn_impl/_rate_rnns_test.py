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
import unittest

import brainstate as bst


class TestRateRNNModels(unittest.TestCase):
    def setUp(self):
        self.num_in = 3
        self.num_out = 3
        self.batch_size = 4
        self.x = jnp.ones((self.batch_size, self.num_in))

    def test_ValinaRNNCell(self):
        model = bst.nn.ValinaRNNCell(num_in=self.num_in, num_out=self.num_out)
        model.init_state(batch_size=self.batch_size)
        output = model.update(self.x)
        self.assertEqual(output.shape, (self.batch_size, self.num_out))

    def test_GRUCell(self):
        model = bst.nn.GRUCell(num_in=self.num_in, num_out=self.num_out)
        model.init_state(batch_size=self.batch_size)
        output = model.update(self.x)
        self.assertEqual(output.shape, (self.batch_size, self.num_out))

    def test_MGUCell(self):
        model = bst.nn.MGUCell(num_in=self.num_in, num_out=self.num_out)
        model.init_state(batch_size=self.batch_size)
        output = model.update(self.x)
        self.assertEqual(output.shape, (self.batch_size, self.num_out))

    def test_LSTMCell(self):
        model = bst.nn.LSTMCell(num_in=self.num_in, num_out=self.num_out)
        model.init_state(batch_size=self.batch_size)
        output = model.update(self.x)
        self.assertEqual(output.shape, (self.batch_size, self.num_out))

    def test_URLSTMCell(self):
        model = bst.nn.URLSTMCell(num_in=self.num_in, num_out=self.num_out)
        model.init_state(batch_size=self.batch_size)
        output = model.update(self.x)
        self.assertEqual(output.shape, (self.batch_size, self.num_out))


if __name__ == '__main__':
    unittest.main()
