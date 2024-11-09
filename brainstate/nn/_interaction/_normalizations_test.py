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

from absl.testing import absltest
from absl.testing import parameterized

import brainstate as bst


class Test_Normalization(parameterized.TestCase):
    @parameterized.product(
        fit=[True, False],
    )
    def test_BatchNorm1d(self, fit):
        net = bst.nn.BatchNorm1d((3, 10))
        bst.environ.set(fit=fit)
        input = bst.random.randn(1, 3, 10)
        output = net(input)

    @parameterized.product(
        fit=[True, False]
    )
    def test_BatchNorm2d(self, fit):
        net = bst.nn.BatchNorm2d([3, 4, 10])
        bst.environ.set(fit=fit)
        input = bst.random.randn(1, 3, 4, 10)
        output = net(input)

    @parameterized.product(
        fit=[True, False]
    )
    def test_BatchNorm3d(self, fit):
        net = bst.nn.BatchNorm3d([3, 4, 5, 10])
        bst.environ.set(fit=fit)
        input = bst.random.randn(1, 3, 4, 5, 10)
        output = net(input)

    # @parameterized.product(
    #   normalized_shape=(10, [5, 10])
    # )
    # def test_LayerNorm(self, normalized_shape):
    #   net = bst.nn.LayerNorm(normalized_shape, )
    #   input = bst.random.randn(20, 5, 10)
    #   output = net(input)
    #
    # @parameterized.product(
    #   num_groups=[1, 2, 3, 6]
    # )
    # def test_GroupNorm(self, num_groups):
    #   input = bst.random.randn(20, 10, 10, 6)
    #   net = bst.nn.GroupNorm(num_groups=num_groups, num_channels=6, )
    #   output = net(input)
    #
    # def test_InstanceNorm(self):
    #   input = bst.random.randn(20, 10, 10, 6)
    #   net = bst.nn.InstanceNorm(num_channels=6, )
    #   output = net(input)


if __name__ == '__main__':
    absltest.main()
