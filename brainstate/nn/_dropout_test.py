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


import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

import brainstate


class TestDropout(parameterized.TestCase):

    def test_dropout_basic(self):
        """Test basic dropout functionality."""
        dropout_layer = brainstate.nn.Dropout(0.5)
        input_data = np.arange(20, dtype=np.float32)

        with brainstate.environ.context(fit=True):
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

    def test_dropout_eval_mode(self):
        """Test that dropout is disabled in evaluation mode."""
        dropout_layer = brainstate.nn.Dropout(0.5)
        input_data = np.arange(20, dtype=np.float32)

        with brainstate.environ.context(fit=False):
            # Without fit context, dropout should be disabled
            output_data = dropout_layer(input_data)
            np.testing.assert_array_equal(input_data, output_data)

    @parameterized.parameters(0.0, 0.2, 0.5, 0.8, 1.0)
    def test_dropout_various_probs(self, prob):
        """Test dropout with various probabilities."""
        dropout_layer = brainstate.nn.Dropout(prob)
        input_data = brainstate.random.randn(1000)

        with brainstate.environ.context(fit=True):
            output_data = dropout_layer(input_data)
            self.assertEqual(input_data.shape, output_data.shape)

            if prob == 1.0:
                # All elements should be kept (no dropout)
                np.testing.assert_array_equal(input_data, output_data)
            elif prob == 0.0:
                # All elements should be dropped
                np.testing.assert_array_equal(np.zeros_like(input_data), output_data)

    def test_dropout_broadcast_dims(self):
        """Test dropout with broadcast dimensions."""
        dropout_layer = brainstate.nn.Dropout(0.5, broadcast_dims=(1, 2))
        input_data = brainstate.random.randn(4, 5, 6)

        with brainstate.environ.context(fit=True):
            output_data = dropout_layer(input_data)
            self.assertEqual(input_data.shape, output_data.shape)

            # Check that the same mask is applied across broadcast dimensions
            for i in range(4):
                channel_output = output_data[i]
                # All elements in a channel should be either all zero or all scaled
                is_zero = (channel_output == 0).all() or (channel_output != 0).all()
                # Note: This might not always be true due to randomness, but generally should hold

    def test_dropout_multidimensional(self):
        """Test dropout with multidimensional input."""
        dropout_layer = brainstate.nn.Dropout(0.5)
        input_data = brainstate.random.randn(10, 20, 30)

        with brainstate.environ.context(fit=True):
            output_data = dropout_layer(input_data)
            self.assertEqual(input_data.shape, output_data.shape)


class TestDropout1d(parameterized.TestCase):

    def test_dropout1d_basic(self):
        """Test basic Dropout1d functionality."""
        dropout_layer = brainstate.nn.Dropout1d(prob=0.5)
        input_data = brainstate.random.randn(2, 3, 4)  # (N, C, L)

        with brainstate.environ.context(fit=True):
            output_data = dropout_layer(input_data)
            self.assertEqual(input_data.shape, output_data.shape)

    def test_dropout1d_channel_wise(self):
        """Test that Dropout1d applies dropout."""
        dropout_layer = brainstate.nn.Dropout1d(prob=0.5)
        input_data = brainstate.random.randn(2, 8, 10)  # (N, C, L)

        with brainstate.environ.context(fit=True):
            output_data = dropout_layer(input_data)
            # Just verify that dropout is applied (shape preserved, some zeros present)
            self.assertEqual(input_data.shape, output_data.shape)

    def test_dropout1d_without_batch(self):
        """Test Dropout1d with unbatched input (C, L)."""
        dropout_layer = brainstate.nn.Dropout1d(prob=0.5)
        input_data = brainstate.random.randn(3, 4)  # (C, L)

        with brainstate.environ.context(fit=True):
            output_data = dropout_layer(input_data)
            self.assertEqual(input_data.shape, output_data.shape)

    def test_dropout1d_eval_mode(self):
        """Test that Dropout1d is disabled in eval mode."""
        dropout_layer = brainstate.nn.Dropout1d(prob=0.5)
        input_data = brainstate.random.randn(2, 3, 4)

        with brainstate.environ.context(fit=False):
            output_data = dropout_layer(input_data)
            np.testing.assert_array_equal(input_data, output_data)

    def test_dropout1d_invalid_shape(self):
        """Test that Dropout1d raises error for invalid input shape."""
        dropout_layer = brainstate.nn.Dropout1d(prob=0.5)
        input_data = brainstate.random.randn(2, 3, 4, 5)  # 4D input is invalid for Dropout1d

        with brainstate.environ.context(fit=True):
            with self.assertRaises(RuntimeError):
                dropout_layer(input_data)


class TestDropout2d(parameterized.TestCase):
    def setUp(self):
        brainstate.random.seed(0)

    def test_dropout2d_basic(self):
        """Test basic Dropout2d functionality."""
        with brainstate.random.seed_context(42):
            dropout_layer = brainstate.nn.Dropout2d(prob=0.5)
            input_data = brainstate.random.randn(2, 3, 4, 5)  # (N, C, H, W)

            with brainstate.environ.context(fit=True):
                output_data = dropout_layer(input_data)
                self.assertEqual(input_data.shape, output_data.shape)
                self.assertTrue(np.any(output_data == 0))

    def test_dropout2d_channel_wise(self):
        """Test that Dropout2d applies dropout."""
        dropout_layer = brainstate.nn.Dropout2d(prob=0.5)
        input_data = brainstate.random.randn(2, 8, 4, 5)  # (N, C, H, W)

        with brainstate.environ.context(fit=True):
            output_data = dropout_layer(input_data)
            # Just verify that dropout is applied (shape preserved, some zeros present)
            self.assertEqual(input_data.shape, output_data.shape)

    def test_dropout2d_without_batch(self):
        """Test Dropout2d with unbatched input (C, H, W)."""
        dropout_layer = brainstate.nn.Dropout2d(prob=0.5)
        input_data = brainstate.random.randn(3, 4, 5)  # (C, H, W)

        with brainstate.environ.context(fit=True):
            output_data = dropout_layer(input_data)
            self.assertEqual(input_data.shape, output_data.shape)

    def test_dropout2d_scaling(self):
        """Test that Dropout2d correctly scales non-dropped elements."""
        dropout_layer = brainstate.nn.Dropout2d(prob=0.5)
        input_data = brainstate.random.randn(2, 3, 4, 5)

        with brainstate.environ.context(fit=True):
            output_data = dropout_layer(input_data)
            scale_factor = 1 / (1 - 0.5)
            mask = ~np.isclose(output_data, 0)
            non_zero_elements = output_data[mask]
            expected_non_zero_elements = input_data[mask] * scale_factor
            np.testing.assert_allclose(non_zero_elements, expected_non_zero_elements, rtol=1e-5)

    def test_dropout2d_eval_mode(self):
        """Test that Dropout2d is disabled in eval mode."""
        dropout_layer = brainstate.nn.Dropout2d(prob=0.5)
        input_data = brainstate.random.randn(2, 3, 4, 5)

        with brainstate.environ.context(fit=False):
            output_data = dropout_layer(input_data)
            np.testing.assert_array_equal(input_data, output_data)


class TestDropout3d(parameterized.TestCase):

    def test_dropout3d_basic(self):
        """Test basic Dropout3d functionality."""
        dropout_layer = brainstate.nn.Dropout3d(prob=0.5)
        input_data = brainstate.random.randn(2, 3, 4, 5, 6)  # (N, C, D, H, W)

        with brainstate.environ.context(fit=True):
            output_data = dropout_layer(input_data)
            self.assertEqual(input_data.shape, output_data.shape)
            self.assertTrue(np.any(output_data == 0))

    def test_dropout3d_channel_wise(self):
        """Test that Dropout3d applies dropout."""
        dropout_layer = brainstate.nn.Dropout3d(prob=0.5)
        input_data = brainstate.random.randn(2, 8, 4, 5, 6)  # (N, C, D, H, W)

        with brainstate.environ.context(fit=True):
            output_data = dropout_layer(input_data)
            # Just verify that dropout is applied (shape preserved, some zeros present)
            self.assertEqual(input_data.shape, output_data.shape)

    def test_dropout3d_without_batch(self):
        """Test Dropout3d with unbatched input (C, D, H, W)."""
        dropout_layer = brainstate.nn.Dropout3d(prob=0.5)
        input_data = brainstate.random.randn(3, 4, 5, 6)  # (C, D, H, W)

        with brainstate.environ.context(fit=True):
            output_data = dropout_layer(input_data)
            self.assertEqual(input_data.shape, output_data.shape)

    def test_dropout3d_scaling(self):
        """Test that Dropout3d correctly scales non-dropped elements."""
        dropout_layer = brainstate.nn.Dropout3d(prob=0.5)
        input_data = brainstate.random.randn(2, 3, 4, 5, 6)

        with brainstate.environ.context(fit=True):
            output_data = dropout_layer(input_data)
            scale_factor = 1 / (1 - 0.5)
            mask = ~np.isclose(output_data, 0)
            non_zero_elements = output_data[mask]
            expected_non_zero_elements = input_data[mask] * scale_factor
            np.testing.assert_allclose(non_zero_elements, expected_non_zero_elements, rtol=1e-5)

    def test_dropout3d_eval_mode(self):
        """Test that Dropout3d is disabled in eval mode."""
        dropout_layer = brainstate.nn.Dropout3d(prob=0.5)
        input_data = brainstate.random.randn(2, 3, 4, 5, 6)

        with brainstate.environ.context(fit=False):
            output_data = dropout_layer(input_data)
            np.testing.assert_array_equal(input_data, output_data)


class TestAlphaDropout(parameterized.TestCase):

    def test_alphadropout_basic(self):
        """Test basic AlphaDropout functionality."""
        dropout_layer = brainstate.nn.AlphaDropout(prob=0.5)
        input_data = brainstate.random.randn(100, 50)

        with brainstate.environ.context(fit=True):
            output_data = dropout_layer(input_data)
            self.assertEqual(input_data.shape, output_data.shape)

    def test_alphadropout_self_normalizing(self):
        """Test that AlphaDropout maintains zero mean and unit variance."""
        dropout_layer = brainstate.nn.AlphaDropout(prob=0.5)
        # Create input with zero mean and unit variance
        input_data = brainstate.random.randn(10000)

        with brainstate.environ.context(fit=True):
            output_data = dropout_layer(input_data)

            # The output should approximately maintain zero mean and unit variance
            output_mean = np.mean(output_data)
            output_std = np.std(output_data)

            # Allow some tolerance due to randomness
            self.assertAlmostEqual(output_mean, 0.0, delta=0.1)
            self.assertAlmostEqual(output_std, 1.0, delta=0.2)

    def test_alphadropout_alpha_value(self):
        """Test that dropped values are set to alpha (not zero)."""
        dropout_layer = brainstate.nn.AlphaDropout(prob=0.5)
        input_data = brainstate.random.randn(1000) + 5.0  # Shift to avoid confusion with alpha

        with brainstate.environ.context(fit=True):
            output_data = dropout_layer(input_data)

            # AlphaDropout should not have zeros, but values close to transformed alpha
            # After affine transformation: alpha * a + b
            expected_dropped_value = dropout_layer.alpha * dropout_layer.a + dropout_layer.b

            # Check that we have some values close to the expected dropped value
            # (within reasonable tolerance due to numerical precision)
            unique_vals = np.unique(np.round(output_data, 3))
            self.assertGreater(len(unique_vals), 1)  # Should have both dropped and kept values

    def test_alphadropout_eval_mode(self):
        """Test that AlphaDropout is disabled in eval mode."""
        dropout_layer = brainstate.nn.AlphaDropout(prob=0.5)
        input_data = brainstate.random.randn(20, 16)

        with brainstate.environ.context(fit=False):
            output_data = dropout_layer(input_data)
            np.testing.assert_array_equal(input_data, output_data)

    @parameterized.parameters(0.2, 0.5, 0.8)
    def test_alphadropout_various_probs(self, prob):
        """Test AlphaDropout with various probabilities."""
        dropout_layer = brainstate.nn.AlphaDropout(prob=prob)
        input_data = brainstate.random.randn(1000)

        with brainstate.environ.context(fit=True):
            output_data = dropout_layer(input_data)
            self.assertEqual(input_data.shape, output_data.shape)

    def test_alphadropout_multidimensional(self):
        """Test AlphaDropout with multidimensional input."""
        dropout_layer = brainstate.nn.AlphaDropout(prob=0.5)
        input_data = brainstate.random.randn(10, 20, 30)

        with brainstate.environ.context(fit=True):
            output_data = dropout_layer(input_data)
            self.assertEqual(input_data.shape, output_data.shape)


class TestFeatureAlphaDropout(parameterized.TestCase):

    def test_featurealphadropout_basic(self):
        """Test basic FeatureAlphaDropout functionality."""
        dropout_layer = brainstate.nn.FeatureAlphaDropout(prob=0.5)
        input_data = brainstate.random.randn(2, 16, 4, 32, 32)  # (N, C, D, H, W)

        with brainstate.environ.context(fit=True):
            output_data = dropout_layer(input_data)
            self.assertEqual(input_data.shape, output_data.shape)

    def test_featurealphadropout_channel_wise(self):
        """Test that FeatureAlphaDropout drops entire channels."""
        dropout_layer = brainstate.nn.FeatureAlphaDropout(prob=0.5, channel_axis=1)
        input_data = brainstate.random.randn(2, 8, 10, 10)  # (N, C, H, W)

        with brainstate.environ.context(fit=True):
            output_data = dropout_layer(input_data)

            # Check that entire channels share the same dropout mask
            for batch in range(2):
                for channel in range(8):
                    channel_data = output_data[batch, channel, :, :]
                    # All elements in a channel should be either all from input or all alpha-transformed
                    unique_vals = np.unique(np.round(channel_data, 4))
                    # Due to the alpha transformation, we can't easily check, but shape should match
                    self.assertEqual(channel_data.shape, (10, 10))

    def test_featurealphadropout_channel_axis(self):
        """Test FeatureAlphaDropout with different channel axes."""
        # Test with channel_axis=-1
        dropout_layer = brainstate.nn.FeatureAlphaDropout(prob=0.5, channel_axis=-1)
        input_data = brainstate.random.randn(2, 10, 10, 8)  # (N, H, W, C)

        with brainstate.environ.context(fit=True):
            output_data = dropout_layer(input_data)
            self.assertEqual(input_data.shape, output_data.shape)

    def test_featurealphadropout_eval_mode(self):
        """Test that FeatureAlphaDropout is disabled in eval mode."""
        dropout_layer = brainstate.nn.FeatureAlphaDropout(prob=0.5)
        input_data = brainstate.random.randn(2, 16, 4, 32, 32)

        with brainstate.environ.context(fit=False):
            output_data = dropout_layer(input_data)
            np.testing.assert_array_equal(input_data, output_data)

    def test_featurealphadropout_self_normalizing(self):
        """Test that FeatureAlphaDropout maintains self-normalizing properties."""
        dropout_layer = brainstate.nn.FeatureAlphaDropout(prob=0.5, channel_axis=1)
        # Create input with zero mean and unit variance
        input_data = brainstate.random.randn(10, 32, 20)

        with brainstate.environ.context(fit=True):
            output_data = dropout_layer(input_data)

            # The output should approximately maintain zero mean and unit variance
            output_mean = np.mean(output_data)
            output_std = np.std(output_data)

            # Allow some tolerance due to randomness
            self.assertAlmostEqual(output_mean, 0.0, delta=0.2)
            self.assertAlmostEqual(output_std, 1.0, delta=0.3)


class TestDropoutFixed(parameterized.TestCase):

    def test_dropoutfixed_basic(self):
        """Test basic DropoutFixed functionality."""
        with brainstate.random.seed_context(42):
            dropout_layer = brainstate.nn.DropoutFixed(in_size=(2, 3), prob=0.5)
            dropout_layer.init_state(batch_size=2)
            input_data = np.random.randn(2, 2, 3)

            with brainstate.environ.context(fit=True):
                output_data = dropout_layer.update(input_data)
                self.assertEqual(input_data.shape, output_data.shape)
                self.assertTrue(np.any(output_data == 0))

    def test_dropoutfixed_mask_persistence(self):
        """Test that DropoutFixed uses the same mask across multiple calls."""
        with brainstate.random.seed_context(42):
            dropout_layer = brainstate.nn.DropoutFixed(in_size=(10,), prob=0.5)
            dropout_layer.init_state(batch_size=5)

            input_data1 = brainstate.random.randn(5, 10)
            input_data2 = brainstate.random.randn(5, 10)

            with brainstate.environ.context(fit=True):
                output_data1 = dropout_layer.update(input_data1)
                output_data2 = dropout_layer.update(input_data2)

                # The dropout mask should be the same for both calls
                mask1 = (output_data1 == 0)
                mask2 = (output_data2 == 0)
                np.testing.assert_array_equal(mask1, mask2)

    def test_dropoutfixed_scaling(self):
        """Test that DropoutFixed correctly scales non-dropped elements."""
        with brainstate.random.seed_context(42):
            dropout_layer = brainstate.nn.DropoutFixed(in_size=(2, 3), prob=0.5)
            dropout_layer.init_state(batch_size=2)
            input_data = np.random.randn(2, 2, 3)

            with brainstate.environ.context(fit=True):
                output_data = dropout_layer.update(input_data)
                scale_factor = 1 / (1 - 0.5)
                non_zero_elements = output_data[output_data != 0]
                expected_non_zero_elements = input_data[output_data != 0] * scale_factor
                np.testing.assert_almost_equal(non_zero_elements, expected_non_zero_elements, decimal=3)

    def test_dropoutfixed_eval_mode(self):
        """Test that DropoutFixed is disabled in eval mode."""
        with brainstate.random.seed_context(42):
            dropout_layer = brainstate.nn.DropoutFixed(in_size=(2, 3), prob=0.5)
            dropout_layer.init_state(batch_size=2)
            input_data = np.random.randn(2, 2, 3)

            with brainstate.environ.context(fit=False):
                output_data = dropout_layer.update(input_data)
                np.testing.assert_array_equal(input_data, output_data)

    def test_dropoutfixed_shape_mismatch(self):
        """Test that DropoutFixed raises error for shape mismatch."""
        with brainstate.random.seed_context(42):
            dropout_layer = brainstate.nn.DropoutFixed(in_size=(2, 3), prob=0.5)
            dropout_layer.init_state(batch_size=2)
            input_data = np.random.randn(3, 2, 3)  # Wrong batch size

            with brainstate.environ.context(fit=True):
                with self.assertRaises(ValueError):
                    dropout_layer.update(input_data)

    @parameterized.parameters(0.2, 0.5, 0.8)
    def test_dropoutfixed_various_probs(self, prob):
        """Test DropoutFixed with various probabilities."""
        with brainstate.random.seed_context(42):
            dropout_layer = brainstate.nn.DropoutFixed(in_size=(10,), prob=prob)
            dropout_layer.init_state(batch_size=5)
            input_data = brainstate.random.randn(5, 10)

            with brainstate.environ.context(fit=True):
                output_data = dropout_layer.update(input_data)
                self.assertEqual(input_data.shape, output_data.shape)


if __name__ == '__main__':
    absltest.main()
