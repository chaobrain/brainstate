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


class Test_Activation(parameterized.TestCase):

    def test_Threshold(self):
        threshold_layer = bst.nn.Threshold(5, 20)
        input = bst.random.randn(2)
        output = threshold_layer(input)

    def test_ReLU(self):
        ReLU_layer = bst.nn.ReLU()
        input = bst.random.randn(2)
        output = ReLU_layer(input)

    def test_RReLU(self):
        RReLU_layer = bst.nn.RReLU(lower=0, upper=1)
        input = bst.random.randn(2)
        output = RReLU_layer(input)

    def test_Hardtanh(self):
        Hardtanh_layer = bst.nn.Hardtanh(min_val=0, max_val=1, )
        input = bst.random.randn(2)
        output = Hardtanh_layer(input)

    def test_ReLU6(self):
        ReLU6_layer = bst.nn.ReLU6()
        input = bst.random.randn(2)
        output = ReLU6_layer(input)

    def test_Sigmoid(self):
        Sigmoid_layer = bst.nn.Sigmoid()
        input = bst.random.randn(2)
        output = Sigmoid_layer(input)

    def test_Hardsigmoid(self):
        Hardsigmoid_layer = bst.nn.Hardsigmoid()
        input = bst.random.randn(2)
        output = Hardsigmoid_layer(input)

    def test_Tanh(self):
        Tanh_layer = bst.nn.Tanh()
        input = bst.random.randn(2)
        output = Tanh_layer(input)

    def test_SiLU(self):
        SiLU_layer = bst.nn.SiLU()
        input = bst.random.randn(2)
        output = SiLU_layer(input)

    def test_Mish(self):
        Mish_layer = bst.nn.Mish()
        input = bst.random.randn(2)
        output = Mish_layer(input)

    def test_Hardswish(self):
        Hardswish_layer = bst.nn.Hardswish()
        input = bst.random.randn(2)
        output = Hardswish_layer(input)

    def test_ELU(self):
        ELU_layer = bst.nn.ELU(alpha=0.5, )
        input = bst.random.randn(2)
        output = ELU_layer(input)

    def test_CELU(self):
        CELU_layer = bst.nn.CELU(alpha=0.5, )
        input = bst.random.randn(2)
        output = CELU_layer(input)

    def test_SELU(self):
        SELU_layer = bst.nn.SELU()
        input = bst.random.randn(2)
        output = SELU_layer(input)

    def test_GLU(self):
        GLU_layer = bst.nn.GLU()
        input = bst.random.randn(4, 2)
        output = GLU_layer(input)

    @parameterized.product(
        approximate=['tanh', 'none']
    )
    def test_GELU(self, approximate):
        GELU_layer = bst.nn.GELU()
        input = bst.random.randn(2)
        output = GELU_layer(input)

    def test_Hardshrink(self):
        Hardshrink_layer = bst.nn.Hardshrink(lambd=1)
        input = bst.random.randn(2)
        output = Hardshrink_layer(input)

    def test_LeakyReLU(self):
        LeakyReLU_layer = bst.nn.LeakyReLU()
        input = bst.random.randn(2)
        output = LeakyReLU_layer(input)

    def test_LogSigmoid(self):
        LogSigmoid_layer = bst.nn.LogSigmoid()
        input = bst.random.randn(2)
        output = LogSigmoid_layer(input)

    def test_Softplus(self):
        Softplus_layer = bst.nn.Softplus()
        input = bst.random.randn(2)
        output = Softplus_layer(input)

    def test_Softshrink(self):
        Softshrink_layer = bst.nn.Softshrink(lambd=1)
        input = bst.random.randn(2)
        output = Softshrink_layer(input)

    def test_PReLU(self):
        PReLU_layer = bst.nn.PReLU(num_parameters=2, init=0.5)
        input = bst.random.randn(2)
        output = PReLU_layer(input)

    def test_Softsign(self):
        Softsign_layer = bst.nn.Softsign()
        input = bst.random.randn(2)
        output = Softsign_layer(input)

    def test_Tanhshrink(self):
        Tanhshrink_layer = bst.nn.Tanhshrink()
        input = bst.random.randn(2)
        output = Tanhshrink_layer(input)

    def test_Softmin(self):
        Softmin_layer = bst.nn.Softmin(dim=2)
        input = bst.random.randn(2, 3, 4)
        output = Softmin_layer(input)

    def test_Softmax(self):
        Softmax_layer = bst.nn.Softmax(dim=2)
        input = bst.random.randn(2, 3, 4)
        output = Softmax_layer(input)

    def test_Softmax2d(self):
        Softmax2d_layer = bst.nn.Softmax2d()
        input = bst.random.randn(2, 3, 12, 13)
        output = Softmax2d_layer(input)

    def test_LogSoftmax(self):
        LogSoftmax_layer = bst.nn.LogSoftmax(dim=2)
        input = bst.random.randn(2, 3, 4)
        output = LogSoftmax_layer(input)


if __name__ == '__main__':
    absltest.main()
