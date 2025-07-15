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

import numpy as np

import brainstate


class TestDropout(unittest.TestCase):

    def test_dropout(self):
        # Create a Dropout layer with a dropout rate of 0.5
        dropout_layer = brainstate.nn.Dropout(0.5)

        # Input data
        input_data = np.arange(20)

        with brainstate.environ.context(fit=True):
            # Apply dropout
            output_data = dropout_layer(input_data)

            # Check that the output has the same shape as the input
            self.assertEqual(input_data.shape, output_data.shape)

            # Check that some elements are zeroed out
            self.assertTrue(np.any(output_data == 0))

            # Check that the non-zero elements are scaled by 1/(1-rate)
            scale_factor = 1 / (1 - 0.5)
            non_zero_elements = output_data[output_data != 0]
            expected_non_zero_elements = input_data[output_data != 0] * scale_factor
            np.testing.assert_almost_equal(non_zero_elements, expected_non_zero_elements)

    def test_DropoutFixed(self):
        dropout_layer = brainstate.nn.DropoutFixed(in_size=(2, 3), prob=0.5)
        dropout_layer.init_state(batch_size=2)
        input_data = np.random.randn(2, 2, 3)
        with brainstate.environ.context(fit=True):
            output_data = dropout_layer.update(input_data)
        self.assertEqual(input_data.shape, output_data.shape)
        self.assertTrue(np.any(output_data == 0))
        scale_factor = 1 / (1 - 0.5)
        non_zero_elements = output_data[output_data != 0]
        expected_non_zero_elements = input_data[output_data != 0] * scale_factor
        np.testing.assert_almost_equal(non_zero_elements, expected_non_zero_elements)

    # def test_Dropout1d(self):
    #     dropout_layer = brainstate.nn.Dropout1d(prob=0.5)
    #     input_data = np.random.randn(2, 3, 4)
    #     with brainstate.environ.context(fit=True):
    #         output_data = dropout_layer(input_data)
    #     self.assertEqual(input_data.shape, output_data.shape)
    #     self.assertTrue(np.any(output_data == 0))
    #     scale_factor = 1 / (1 - 0.5)
    #     non_zero_elements = output_data[output_data != 0]
    #     expected_non_zero_elements = input_data[output_data != 0] * scale_factor
    #     np.testing.assert_almost_equal(non_zero_elements, expected_non_zero_elements, decimal=4)

    def test_Dropout2d(self):
        dropout_layer = brainstate.nn.Dropout2d(prob=0.5)
        input_data = np.random.randn(2, 3, 4, 5)
        with brainstate.environ.context(fit=True):
            output_data = dropout_layer(input_data)
        self.assertEqual(input_data.shape, output_data.shape)
        self.assertTrue(np.any(output_data == 0))
        scale_factor = 1 / (1 - 0.5)
        non_zero_elements = output_data[output_data != 0]
        expected_non_zero_elements = input_data[output_data != 0] * scale_factor
        np.testing.assert_almost_equal(non_zero_elements, expected_non_zero_elements, decimal=4)

    def test_Dropout3d(self):
        dropout_layer = brainstate.nn.Dropout3d(prob=0.5)
        input_data = np.random.randn(2, 3, 4, 5, 6)
        with brainstate.environ.context(fit=True):
            output_data = dropout_layer(input_data)
        self.assertEqual(input_data.shape, output_data.shape)
        self.assertTrue(np.any(output_data == 0))
        scale_factor = 1 / (1 - 0.5)
        non_zero_elements = output_data[output_data != 0]
        expected_non_zero_elements = input_data[output_data != 0] * scale_factor
        np.testing.assert_almost_equal(non_zero_elements, expected_non_zero_elements, decimal=4)


if __name__ == '__main__':
    unittest.main()
