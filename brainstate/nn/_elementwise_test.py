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

import unittest
from absl.testing import absltest
from absl.testing import parameterized

import jax
import jax.numpy as jnp
import numpy as np

import brainstate
import brainstate.nn as nn


class TestActivationFunctions(parameterized.TestCase):
    """Comprehensive tests for activation functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.seed = 42
        self.key = jax.random.PRNGKey(self.seed)

    def _check_shape_preservation(self, layer, input_shape):
        """Helper to check if layer preserves input shape."""
        x = jax.random.normal(self.key, input_shape)
        output = layer(x)
        self.assertEqual(output.shape, x.shape)

    def _check_gradient_flow(self, layer, input_shape):
        """Helper to check if gradients can flow through the layer."""
        x = jax.random.normal(self.key, input_shape)

        def loss_fn(x):
            return jnp.sum(layer(x))

        grad = jax.grad(loss_fn)(x)
        self.assertEqual(grad.shape, x.shape)
        # Check that gradients are not all zeros (for most activations)
        if not isinstance(layer, (nn.Threshold, nn.Hardtanh, nn.ReLU6)):
            self.assertFalse(jnp.allclose(grad, 0.0))

    # Test Threshold
    def test_threshold_functionality(self):
        """Test Threshold activation function."""
        layer = nn.Threshold(threshold=0.5, value=0.0)

        # Test with values above and below threshold
        x = jnp.array([-1.0, 0.0, 0.3, 0.7, 1.0])
        output = layer(x)
        expected = jnp.array([0.0, 0.0, 0.0, 0.7, 1.0])
        np.testing.assert_allclose(output, expected, rtol=1e-5)

    @parameterized.parameters(
        ((2,), ),
        ((3, 4), ),
        ((2, 3, 4), ),
        ((2, 3, 4, 5), ),
    )
    def test_threshold_shapes(self, shape):
        """Test Threshold with different input shapes."""
        layer = nn.Threshold(threshold=0.1, value=20)
        self._check_shape_preservation(layer, shape)

    # Test ReLU
    def test_relu_functionality(self):
        """Test ReLU activation function."""
        layer = nn.ReLU()

        x = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        output = layer(x)
        expected = jnp.array([0.0, 0.0, 0.0, 1.0, 2.0])
        np.testing.assert_allclose(output, expected, rtol=1e-5)

    @parameterized.parameters(
        ((10,), ),
        ((5, 10), ),
        ((3, 5, 10), ),
    )
    def test_relu_shapes_and_gradients(self, shape):
        """Test ReLU shapes and gradients."""
        layer = nn.ReLU()
        self._check_shape_preservation(layer, shape)
        self._check_gradient_flow(layer, shape)

    # Test RReLU
    def test_rrelu_functionality(self):
        """Test RReLU activation function."""
        layer = nn.RReLU(lower=0.1, upper=0.3)

        # Test positive and negative values
        x = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        output = layer(x)

        # Positive values should remain unchanged
        self.assertTrue(jnp.all(output[x > 0] == x[x > 0]))
        # Negative values should be scaled by a factor in [lower, upper]
        negative_mask = x < 0
        if jnp.any(negative_mask):
            scaled = output[negative_mask] / x[negative_mask]
            self.assertTrue(jnp.all((scaled >= 0.1) & (scaled <= 0.3)))

    # Test Hardtanh
    def test_hardtanh_functionality(self):
        """Test Hardtanh activation function."""
        layer = nn.Hardtanh(min_val=-1.0, max_val=1.0)

        x = jnp.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
        output = layer(x)
        expected = jnp.array([-1.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.0])
        np.testing.assert_allclose(output, expected, rtol=1e-5)

    def test_hardtanh_custom_bounds(self):
        """Test Hardtanh with custom bounds."""
        layer = nn.Hardtanh(min_val=-2.0, max_val=3.0)

        x = jnp.array([-3.0, -2.0, 0.0, 3.0, 4.0])
        output = layer(x)
        expected = jnp.array([-2.0, -2.0, 0.0, 3.0, 3.0])
        np.testing.assert_allclose(output, expected, rtol=1e-5)

    # Test ReLU6
    def test_relu6_functionality(self):
        """Test ReLU6 activation function."""
        layer = nn.ReLU6()

        x = jnp.array([-2.0, 0.0, 3.0, 6.0, 8.0])
        output = layer(x)
        expected = jnp.array([0.0, 0.0, 3.0, 6.0, 6.0])
        np.testing.assert_allclose(output, expected, rtol=1e-5)

    # Test Sigmoid
    def test_sigmoid_functionality(self):
        """Test Sigmoid activation function."""
        layer = nn.Sigmoid()

        x = jnp.array([-10.0, -1.0, 0.0, 1.0, 10.0])
        output = layer(x)

        # Check sigmoid properties
        self.assertTrue(jnp.all((output >= 0.0) & (output <= 1.0)))
        np.testing.assert_allclose(output[2], 0.5, rtol=1e-5)  # sigmoid(0) = 0.5

    @parameterized.parameters(
        ((10,), ),
        ((5, 10), ),
        ((3, 5, 10), ),
    )
    def test_sigmoid_shapes_and_gradients(self, shape):
        """Test Sigmoid shapes and gradients."""
        layer = nn.Sigmoid()
        self._check_shape_preservation(layer, shape)
        self._check_gradient_flow(layer, shape)

    # Test Hardsigmoid
    def test_hardsigmoid_functionality(self):
        """Test Hardsigmoid activation function."""
        layer = nn.Hardsigmoid()

        x = jnp.array([-4.0, -3.0, -1.0, 0.0, 1.0, 3.0, 4.0])
        output = layer(x)

        # Check bounds
        self.assertTrue(jnp.all((output >= 0.0) & (output <= 1.0)))
        # Check specific values
        np.testing.assert_allclose(output[1], 0.0, rtol=1e-5)  # x=-3
        np.testing.assert_allclose(output[3], 0.5, rtol=1e-5)  # x=0
        np.testing.assert_allclose(output[5], 1.0, rtol=1e-5)  # x=3

    # Test Tanh
    def test_tanh_functionality(self):
        """Test Tanh activation function."""
        layer = nn.Tanh()

        x = jnp.array([-10.0, -1.0, 0.0, 1.0, 10.0])
        output = layer(x)

        # Check tanh properties
        self.assertTrue(jnp.all((output >= -1.0) & (output <= 1.0)))
        np.testing.assert_allclose(output[2], 0.0, rtol=1e-5)  # tanh(0) = 0

    # Test SiLU (Swish)
    def test_silu_functionality(self):
        """Test SiLU activation function."""
        layer = nn.SiLU()

        x = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        output = layer(x)

        # SiLU(x) = x * sigmoid(x)
        expected = x * jax.nn.sigmoid(x)
        np.testing.assert_allclose(output, expected, rtol=1e-5)

    # Test Mish
    def test_mish_functionality(self):
        """Test Mish activation function."""
        layer = nn.Mish()

        x = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        output = layer(x)

        # Mish(x) = x * tanh(softplus(x))
        expected = x * jnp.tanh(jax.nn.softplus(x))
        np.testing.assert_allclose(output, expected, rtol=1e-5)

    # Test Hardswish
    def test_hardswish_functionality(self):
        """Test Hardswish activation function."""
        layer = nn.Hardswish()

        x = jnp.array([-4.0, -3.0, -1.0, 0.0, 1.0, 3.0, 4.0])
        output = layer(x)

        # Check boundary conditions
        np.testing.assert_allclose(output[1], 0.0, rtol=1e-5)  # x=-3
        np.testing.assert_allclose(output[5], 3.0, rtol=1e-5)  # x=3

    # Test ELU
    def test_elu_functionality(self):
        """Test ELU activation function."""
        layer = nn.ELU(alpha=1.0)

        x = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        output = layer(x)

        # Positive values should remain unchanged
        self.assertTrue(jnp.all(output[x > 0] == x[x > 0]))
        # Check ELU formula for negative values
        negative_mask = x <= 0
        expected_negative = 1.0 * (jnp.exp(x[negative_mask]) - 1)
        np.testing.assert_allclose(output[negative_mask], expected_negative, rtol=1e-5)

    def test_elu_with_different_alpha(self):
        """Test ELU with different alpha values."""
        alpha = 2.0
        layer = nn.ELU(alpha=alpha)

        x = jnp.array([-1.0])
        output = layer(x)
        expected = alpha * (jnp.exp(x) - 1)
        np.testing.assert_allclose(output, expected, rtol=1e-5)

    # Test CELU
    def test_celu_functionality(self):
        """Test CELU activation function."""
        layer = nn.CELU(alpha=1.0)

        x = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        output = layer(x)

        # Positive values should remain unchanged
        self.assertTrue(jnp.all(output[x > 0] == x[x > 0]))

    # Test SELU
    def test_selu_functionality(self):
        """Test SELU activation function."""
        layer = nn.SELU()

        x = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        output = layer(x)

        # Check that output is scaled ELU
        # SELU has specific scale and alpha values
        scale = 1.0507009873554804934193349852946
        alpha = 1.6732632423543772848170429916717

        positive_mask = x > 0
        self.assertTrue(jnp.all(output[positive_mask] == scale * x[positive_mask]))

    # Test GLU
    def test_glu_functionality(self):
        """Test GLU activation function."""
        layer = nn.GLU(dim=-1)

        # GLU splits input in half along specified dimension
        x = jnp.array([[1.0, 2.0, 3.0, 4.0],
                       [5.0, 6.0, 7.0, 8.0]])
        output = layer(x)

        # Output should have half the size along the split dimension
        self.assertEqual(output.shape, (2, 2))

    def test_glu_different_dimensions(self):
        """Test GLU with different split dimensions."""
        # Test splitting along different dimensions
        x = jax.random.normal(self.key, (4, 6, 8))

        layer_0 = nn.GLU(dim=0)
        output_0 = layer_0(x)
        self.assertEqual(output_0.shape, (2, 6, 8))

        layer_1 = nn.GLU(dim=1)
        output_1 = layer_1(x)
        self.assertEqual(output_1.shape, (4, 3, 8))

        layer_2 = nn.GLU(dim=2)
        output_2 = layer_2(x)
        self.assertEqual(output_2.shape, (4, 6, 4))

    # Test GELU
    def test_gelu_functionality(self):
        """Test GELU activation function."""
        layer = nn.GELU(approximate=False)

        x = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        output = layer(x)

        # GELU should be smooth and differentiable everywhere
        np.testing.assert_allclose(output[2], 0.0, rtol=1e-5)  # GELU(0) ≈ 0

    def test_gelu_approximate(self):
        """Test GELU with tanh approximation."""
        layer_exact = nn.GELU(approximate=False)
        layer_approx = nn.GELU(approximate=True)

        x = jnp.array([-1.0, 0.0, 1.0])
        output_exact = layer_exact(x)
        output_approx = layer_approx(x)

        # Approximation should be close but not exactly equal
        np.testing.assert_allclose(output_exact, output_approx, rtol=1e-2)

    # Test Hardshrink
    def test_hardshrink_functionality(self):
        """Test Hardshrink activation function."""
        lambd = 0.5
        layer = nn.Hardshrink(lambd=lambd)

        x = jnp.array([-1.0, -0.6, -0.5, -0.3, 0.0, 0.3, 0.5, 0.6, 1.0])
        output = layer(x)

        # Check each value according to hardshrink formula
        expected = []
        for xi in x:
            if xi > lambd:
                expected.append(xi)
            elif xi < -lambd:
                expected.append(xi)
            else:
                expected.append(0.0)
        expected = jnp.array(expected)

        np.testing.assert_allclose(output, expected, rtol=1e-5)

    # Test LeakyReLU
    def test_leaky_relu_functionality(self):
        """Test LeakyReLU activation function."""
        negative_slope = 0.01
        layer = nn.LeakyReLU(negative_slope=negative_slope)

        x = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        output = layer(x)

        # Positive values should remain unchanged
        self.assertTrue(jnp.all(output[x > 0] == x[x > 0]))
        # Negative values should be scaled
        negative_mask = x < 0
        expected_negative = negative_slope * x[negative_mask]
        np.testing.assert_allclose(output[negative_mask], expected_negative, rtol=1e-5)

    def test_leaky_relu_custom_slope(self):
        """Test LeakyReLU with custom negative slope."""
        negative_slope = 0.2
        layer = nn.LeakyReLU(negative_slope=negative_slope)

        x = jnp.array([-5.0])
        output = layer(x)
        expected = negative_slope * x
        np.testing.assert_allclose(output, expected, rtol=1e-5)

    # Test LogSigmoid
    def test_log_sigmoid_functionality(self):
        """Test LogSigmoid activation function."""
        layer = nn.LogSigmoid()

        x = jnp.array([-10.0, -1.0, 0.0, 1.0, 10.0])
        output = layer(x)

        # LogSigmoid(x) = log(sigmoid(x))
        expected = jnp.log(jax.nn.sigmoid(x))
        np.testing.assert_allclose(output, expected, rtol=1e-2)

        # Output should always be negative or zero
        self.assertTrue(jnp.all(output <= 0.0))

    # Test Softplus
    def test_softplus_functionality(self):
        """Test Softplus activation function."""
        layer = nn.Softplus()

        x = jnp.array([-10.0, -1.0, 0.0, 1.0, 10.0])
        output = layer(x)

        # Softplus is a smooth approximation to ReLU
        # Should always be positive
        self.assertTrue(jnp.all(output > 0.0))

        # For large positive values, should approximate x
        np.testing.assert_allclose(output[-1], x[-1], rtol=1e-2)

    # Test Softshrink
    def test_softshrink_functionality(self):
        """Test Softshrink activation function."""
        lambd = 0.5
        layer = nn.Softshrink(lambd=lambd)

        x = jnp.array([-1.0, -0.5, -0.3, 0.0, 0.3, 0.5, 1.0])
        output = layer(x)

        # Check the softshrink formula
        for i in range(len(x)):
            if x[i] > lambd:
                expected = x[i] - lambd
            elif x[i] < -lambd:
                expected = x[i] + lambd
            else:
                expected = 0.0
            np.testing.assert_allclose(output[i], expected, rtol=1e-5)

    # Test PReLU
    def test_prelu_functionality(self):
        """Test PReLU activation function."""
        layer = nn.PReLU(num_parameters=1, init=0.25)

        x = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        output = layer(x)

        # Positive values should remain unchanged
        self.assertTrue(jnp.all(output[x > 0] == x[x > 0]))
        # Negative values should be scaled by learned parameter
        negative_mask = x < 0
        # Check that negative values are scaled
        self.assertTrue(jnp.all(output[negative_mask] != x[negative_mask]))

    def test_prelu_multi_channel(self):
        """Test PReLU with multiple channels."""
        num_channels = 3
        layer = nn.PReLU(num_parameters=num_channels, init=0.25)

        # Input shape: (batch, channels, height, width)
        x = jax.random.normal(self.key, (2, 4, 4, num_channels))
        output = layer(x)

        self.assertEqual(output.shape, x.shape)

    # Test Softsign
    def test_softsign_functionality(self):
        """Test Softsign activation function."""
        layer = nn.Softsign()

        x = jnp.array([-10.0, -1.0, 0.0, 1.0, 10.0])
        output = layer(x)

        # Softsign(x) = x / (1 + |x|)
        expected = x / (1 + jnp.abs(x))
        np.testing.assert_allclose(output, expected, rtol=1e-5)

        # Output should be bounded between -1 and 1
        self.assertTrue(jnp.all((output >= -1.0) & (output <= 1.0)))

    # Test Tanhshrink
    def test_tanhshrink_functionality(self):
        """Test Tanhshrink activation function."""
        layer = nn.Tanhshrink()

        x = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        output = layer(x)

        # Tanhshrink(x) = x - tanh(x)
        expected = x - jnp.tanh(x)
        np.testing.assert_allclose(output, expected, rtol=1e-5)

    # Test Softmin
    def test_softmin_functionality(self):
        """Test Softmin activation function."""
        layer = nn.Softmin(dim=-1)

        x = jnp.array([[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0]])
        output = layer(x)

        # Softmin should sum to 1 along the specified dimension
        sums = jnp.sum(output, axis=-1)
        np.testing.assert_allclose(sums, jnp.ones_like(sums), rtol=1e-5)

        # Higher values should have lower probabilities
        self.assertTrue(jnp.all(output[:, 0] > output[:, 1]))
        self.assertTrue(jnp.all(output[:, 1] > output[:, 2]))

    # Test Softmax
    def test_softmax_functionality(self):
        """Test Softmax activation function."""
        layer = nn.Softmax(dim=-1)

        x = jnp.array([[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0]])
        output = layer(x)

        # Softmax should sum to 1 along the specified dimension
        sums = jnp.sum(output, axis=-1)
        np.testing.assert_allclose(sums, jnp.ones_like(sums), rtol=1e-5)

        # Higher values should have higher probabilities
        self.assertTrue(jnp.all(output[:, 2] > output[:, 1]))
        self.assertTrue(jnp.all(output[:, 1] > output[:, 0]))

    def test_softmax_numerical_stability(self):
        """Test Softmax numerical stability with large values."""
        layer = nn.Softmax(dim=-1)

        # Test with large values that could cause overflow
        x = jnp.array([[1000.0, 1000.0, 1000.0]])
        output = layer(x)

        # Should still sum to 1 and have equal probabilities
        np.testing.assert_allclose(jnp.sum(output), 1.0, rtol=1e-5)
        np.testing.assert_allclose(output[0, 0], 1/3, rtol=1e-5)

    # Test Softmax2d
    def test_softmax2d_functionality(self):
        """Test Softmax2d activation function."""
        layer = nn.Softmax2d()

        # Input shape: (batch, channels, height, width)
        x = jax.random.normal(self.key, (2, 3, 4, 5))
        output = layer(x)

        self.assertEqual(output.shape, x.shape)

        # Should sum to 1 across channels for each spatial location
        channel_sums = jnp.sum(output, axis=1)
        np.testing.assert_allclose(channel_sums, jnp.ones_like(channel_sums), rtol=1e-5)

    def test_softmax2d_3d_input(self):
        """Test Softmax2d with 3D input."""
        layer = nn.Softmax2d()

        # Input shape: (channels, height, width)
        x = jax.random.normal(self.key, (3, 4, 5))
        output = layer(x)

        self.assertEqual(output.shape, x.shape)

    # Test LogSoftmax
    def test_log_softmax_functionality(self):
        """Test LogSoftmax activation function."""
        layer = nn.LogSoftmax(dim=-1)

        x = jnp.array([[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0]])
        output = layer(x)

        # LogSoftmax = log(softmax(x))
        softmax_output = jax.nn.softmax(x, axis=-1)
        expected = jnp.log(softmax_output)
        np.testing.assert_allclose(output, expected, rtol=1e-5)

        # Output should be all negative or zero
        self.assertTrue(jnp.all(output <= 0.0))

    def test_log_softmax_numerical_stability(self):
        """Test LogSoftmax numerical stability."""
        layer = nn.LogSoftmax(dim=-1)

        # Test with values that could cause numerical issues
        x = jnp.array([[1000.0, 0.0, -1000.0]])
        output = layer(x)

        # Should not contain NaN or Inf
        self.assertFalse(jnp.any(jnp.isnan(output)))
        self.assertFalse(jnp.any(jnp.isinf(output)))

    # Test Identity
    def test_identity_functionality(self):
        """Test Identity activation function."""
        layer = nn.Identity()

        x = jax.random.normal(self.key, (3, 4, 5))
        output = layer(x)

        # Should be exactly equal to input
        np.testing.assert_array_equal(output, x)

    def test_identity_gradient(self):
        """Test Identity gradient flow."""
        layer = nn.Identity()

        x = jax.random.normal(self.key, (3, 4))

        def loss_fn(x):
            return jnp.sum(layer(x))

        grad = jax.grad(loss_fn)(x)

        # Gradient should be all ones
        np.testing.assert_allclose(grad, jnp.ones_like(x), rtol=1e-5)

    # Test SpikeBitwise
    def test_spike_bitwise_add(self):
        """Test SpikeBitwise with ADD operation."""
        layer = nn.SpikeBitwise(op='and')

        x = jnp.array([[1.0, 0.0], [1.0, 1.0]])
        y = jnp.array([[1.0, 1.0], [0.0, 1.0]])
        output = layer(x, y)

        expected = jnp.logical_and(x, y)
        np.testing.assert_allclose(output, expected, rtol=1e-5)

    def test_spike_bitwise_and(self):
        """Test SpikeBitwise with AND operation."""
        layer = nn.SpikeBitwise(op='and')

        x = jnp.array([[1.0, 0.0], [1.0, 1.0]])
        y = jnp.array([[1.0, 1.0], [0.0, 1.0]])
        output = layer(x, y)

        expected = x * y
        np.testing.assert_allclose(output, expected, rtol=1e-5)

    def test_spike_bitwise_iand(self):
        """Test SpikeBitwise with IAND operation."""
        layer = nn.SpikeBitwise(op='iand')

        x = jnp.array([[1.0, 0.0], [1.0, 1.0]])
        y = jnp.array([[1.0, 1.0], [0.0, 1.0]])
        output = layer(x, y)

        expected = (1 - x) * y
        np.testing.assert_allclose(output, expected, rtol=1e-5)

    def test_spike_bitwise_or(self):
        """Test SpikeBitwise with OR operation."""
        layer = nn.SpikeBitwise(op='or')

        x = jnp.array([[1.0, 0.0], [1.0, 1.0]])
        y = jnp.array([[1.0, 1.0], [0.0, 1.0]])
        output = layer(x, y)

        expected = (x + y) - (x * y)
        np.testing.assert_allclose(output, expected, rtol=1e-5)


class TestEdgeCases(parameterized.TestCase):
    """Test edge cases and boundary conditions."""

    def test_zero_input(self):
        """Test all activations with zero input."""
        x = jnp.zeros((3, 4))

        activations = [
            nn.ReLU(),
            nn.Sigmoid(),
            nn.Tanh(),
            nn.SiLU(),
            nn.ELU(),
            nn.GELU(),
            nn.Softplus(),
            nn.Softsign(),
        ]

        for activation in activations:
            output = activation(x)
            self.assertEqual(output.shape, x.shape)
            self.assertFalse(jnp.any(jnp.isnan(output)))

    def test_large_positive_input(self):
        """Test activations with very large positive values."""
        x = jnp.ones((2, 3)) * 1000.0

        activations = [
            nn.ReLU(),
            nn.Sigmoid(),
            nn.Tanh(),
            nn.Hardsigmoid(),
            nn.Hardswish(),
        ]

        for activation in activations:
            output = activation(x)
            self.assertFalse(jnp.any(jnp.isnan(output)))
            self.assertFalse(jnp.any(jnp.isinf(output)))

    def test_large_negative_input(self):
        """Test activations with very large negative values."""
        x = jnp.ones((2, 3)) * -1000.0

        activations = [
            nn.ReLU(),
            nn.Sigmoid(),
            nn.Tanh(),
            nn.Hardsigmoid(),
            nn.Hardswish(),
        ]

        for activation in activations:
            output = activation(x)
            self.assertFalse(jnp.any(jnp.isnan(output)))
            self.assertFalse(jnp.any(jnp.isinf(output)))

    def test_nan_propagation(self):
        """Test that NaN inputs produce NaN outputs (where appropriate)."""
        x = jnp.array([jnp.nan, 1.0, 2.0])

        activations = [
            nn.ReLU(),
            nn.Sigmoid(),
            nn.Tanh(),
        ]

        for activation in activations:
            output = activation(x)
            self.assertTrue(jnp.isnan(output[0]))

    def test_inf_handling(self):
        """Test handling of infinite values."""
        x = jnp.array([jnp.inf, -jnp.inf, 1.0])

        # ReLU should handle inf properly
        relu = nn.ReLU()
        output = relu(x)
        self.assertEqual(output[0], jnp.inf)
        self.assertEqual(output[1], 0.0)

        # Sigmoid should saturate
        sigmoid = nn.Sigmoid()
        output = sigmoid(x)
        np.testing.assert_allclose(output[0], 1.0, rtol=1e-5)
        np.testing.assert_allclose(output[1], 0.0, rtol=1e-5)


class TestBatchProcessing(parameterized.TestCase):
    """Test batch processing capabilities."""

    @parameterized.parameters(
        (nn.ReLU(), ),
        (nn.Sigmoid(), ),
        (nn.Tanh(), ),
        (nn.GELU(), ),
        (nn.SiLU(), ),
        (nn.ELU(), ),
    )
    def test_batch_consistency(self, activation):
        """Test that batch processing gives same results as individual processing."""
        # Process as batch
        batch_input = jax.random.normal(jax.random.PRNGKey(42), (5, 10))
        batch_output = activation(batch_input)

        # Process individually
        individual_outputs = []
        for i in range(5):
            individual_output = activation(batch_input[i])
            individual_outputs.append(individual_output)
        individual_outputs = jnp.stack(individual_outputs)

        np.testing.assert_allclose(batch_output, individual_outputs, rtol=1e-5)

    def test_different_batch_sizes(self):
        """Test activations with different batch sizes."""
        activation = nn.ReLU()

        for batch_size in [1, 10, 100]:
            x = jax.random.normal(jax.random.PRNGKey(42), (batch_size, 20))
            output = activation(x)
            self.assertEqual(output.shape[0], batch_size)


class TestMemoryAndPerformance(parameterized.TestCase):
    """Test memory and performance characteristics."""

    def test_in_place_operations(self):
        """Test that activations don't modify input in-place."""
        x_original = jax.random.normal(jax.random.PRNGKey(42), (10, 10))
        x = x_original.copy()

        activations = [
            nn.ReLU(),
            nn.Sigmoid(),
            nn.Tanh(),
        ]

        for activation in activations:
            output = activation(x)
            np.testing.assert_array_equal(x, x_original)

    def test_jit_compilation(self):
        """Test that activations work with JIT compilation."""
        @jax.jit
        def forward(x):
            relu = nn.ReLU()
            return relu(x)

        x = jax.random.normal(jax.random.PRNGKey(42), (10, 10))
        output = forward(x)

        # Should not raise any errors and produce valid output
        self.assertEqual(output.shape, x.shape)

    @parameterized.parameters(
        (nn.ReLU(), ),
        (nn.Sigmoid(), ),
        (nn.Tanh(), ),
    )
    def test_vmap_compatibility(self, activation):
        """Test compatibility with vmap."""
        def single_forward(x):
            return activation(x)

        batch_forward = jax.vmap(single_forward)

        x = jax.random.normal(jax.random.PRNGKey(42), (5, 10, 20))
        output = batch_forward(x)

        self.assertEqual(output.shape, x.shape)


if __name__ == '__main__':
    absltest.main()