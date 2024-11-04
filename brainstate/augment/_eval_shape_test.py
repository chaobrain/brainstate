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


class TestEvalShape(unittest.TestCase):
    def test1(self):
        class MLP(bst.nn.Module):
            def __init__(self, n_in, n_mid, n_out):
                super().__init__()
                self.dense1 = bst.nn.Linear(n_in, n_mid)
                self.dense2 = bst.nn.Linear(n_mid, n_out)

            def __call__(self, x):
                x = self.dense1(x)
                x = bst.functional.relu(x)
                x = self.dense2(x)
                return x

        r = bst.augment.eval_shape(lambda: MLP(1, 2, 3))
        print(r)
        print(bst.random.DEFAULT)