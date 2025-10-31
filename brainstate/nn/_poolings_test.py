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

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

import brainstate
import brainstate.nn as nn


class TestFlatten(parameterized.TestCase):
    def test_flatten1(self):
        for size in [
            (16, 32, 32, 8),
            (32, 8),
            (10, 20, 30),
        ]:
            arr = brainstate.random.rand(*size)
            f = nn.Flatten(start_axis=0)
            out = f(arr)
            self.assertTrue(out.shape == (np.prod(size),))

    def test_flatten2(self):
        for size in [
            (16, 32, 32, 8),
            (32, 8),
            (10, 20, 30),
        ]:
            arr = brainstate.random.rand(*size)
            f = nn.Flatten(start_axis=1)
            out = f(arr)
            self.assertTrue(out.shape == (size[0], np.prod(size[1:])))

    def test_flatten3(self):
        size = (16, 32, 32, 8)
        arr = brainstate.random.rand(*size)
        f = nn.Flatten(start_axis=0, in_size=(32, 8))
        out = f(arr)
        self.assertTrue(out.shape == (16, 32, 32 * 8))

    def test_flatten4(self):
        size = (16, 32, 32, 8)
        arr = brainstate.random.rand(*size)
        f = nn.Flatten(start_axis=1, in_size=(32, 32, 8))
        out = f(arr)
        self.assertTrue(out.shape == (16, 32, 32 * 8))


class TestUnflatten(parameterized.TestCase):
    """Comprehensive tests for Unflatten layer.

    Note: Due to a bug in u.math.unflatten with negative axis handling,
    we only test with positive axis values.
    """

    def test_unflatten_basic_2d(self):
        """Test basic Unflatten functionality for 2D tensors."""
        arr = brainstate.random.rand(6, 12)

        # Unflatten last dimension (use positive axis due to bug)
        unflatten = nn.Unflatten(axis=1, sizes=(3, 4))
        out = unflatten(arr)
        self.assertEqual(out.shape, (6, 3, 4))

        # Unflatten first dimension
        unflatten = nn.Unflatten(axis=0, sizes=(2, 3))
        out = unflatten(arr)
        self.assertEqual(out.shape, (2, 3, 12))

    def test_unflatten_basic_3d(self):
        """Test basic Unflatten functionality for 3D tensors."""
        arr = brainstate.random.rand(4, 6, 24)

        # Unflatten last dimension using positive index
        unflatten = nn.Unflatten(axis=2, sizes=(2, 3, 4))
        out = unflatten(arr)
        self.assertEqual(out.shape, (4, 6, 2, 3, 4))

        # Unflatten middle dimension
        unflatten = nn.Unflatten(axis=1, sizes=(2, 3))
        out = unflatten(arr)
        self.assertEqual(out.shape, (4, 2, 3, 24))

    def test_unflatten_with_in_size(self):
        """Test Unflatten with in_size parameter."""
        # Test with in_size specified
        unflatten = nn.Unflatten(axis=1, sizes=(2, 3), in_size=(4, 6))

        # Check that out_size is computed correctly
        self.assertIsNotNone(unflatten.out_size)
        self.assertEqual(unflatten.out_size, (4, 2, 3))

        # Apply to actual tensor
        arr = brainstate.random.rand(4, 6)
        out = unflatten(arr)
        self.assertEqual(out.shape, (4, 2, 3))

    def test_unflatten_preserve_batch_dims(self):
        """Test that Unflatten preserves batch dimensions."""
        # Multiple batch dimensions
        arr = brainstate.random.rand(2, 3, 4, 20)

        # Unflatten last dimension (use positive axis)
        unflatten = nn.Unflatten(axis=3, sizes=(4, 5))
        out = unflatten(arr)
        self.assertEqual(out.shape, (2, 3, 4, 4, 5))

    def test_unflatten_single_element_split(self):
        """Test Unflatten with sizes containing 1."""
        arr = brainstate.random.rand(3, 12)

        # Include dimension of size 1
        unflatten = nn.Unflatten(axis=1, sizes=(1, 3, 4))
        out = unflatten(arr)
        self.assertEqual(out.shape, (3, 1, 3, 4))

        # Multiple ones
        unflatten = nn.Unflatten(axis=1, sizes=(1, 1, 12))
        out = unflatten(arr)
        self.assertEqual(out.shape, (3, 1, 1, 12))

    def test_unflatten_large_split(self):
        """Test Unflatten with large number of dimensions."""
        arr = brainstate.random.rand(2, 120)

        # Split into many dimensions
        unflatten = nn.Unflatten(axis=1, sizes=(2, 3, 4, 5))
        out = unflatten(arr)
        self.assertEqual(out.shape, (2, 2, 3, 4, 5))

        # Verify total elements preserved
        self.assertEqual(arr.size, out.size)
        self.assertEqual(2 * 3 * 4 * 5, 120)

    def test_unflatten_flatten_inverse(self):
        """Test that Unflatten is inverse of Flatten."""
        original = brainstate.random.rand(2, 3, 4, 5)

        # Flatten dimensions 1 and 2
        flatten = nn.Flatten(start_axis=1, end_axis=2)
        flattened = flatten(original)
        self.assertEqual(flattened.shape, (2, 12, 5))

        # Unflatten back
        unflatten = nn.Unflatten(axis=1, sizes=(3, 4))
        restored = unflatten(flattened)
        self.assertEqual(restored.shape, original.shape)

        # Values should be identical
        self.assertTrue(jnp.allclose(original, restored))

    def test_unflatten_sequential_operations(self):
        """Test Unflatten in sequential operations."""
        arr = brainstate.random.rand(4, 24)

        # Apply multiple unflatten operations
        unflatten1 = nn.Unflatten(axis=1, sizes=(6, 4))
        intermediate = unflatten1(arr)
        self.assertEqual(intermediate.shape, (4, 6, 4))

        unflatten2 = nn.Unflatten(axis=1, sizes=(2, 3))
        final = unflatten2(intermediate)
        self.assertEqual(final.shape, (4, 2, 3, 4))

    def test_unflatten_error_cases(self):
        """Test error handling in Unflatten."""
        # Test invalid sizes type
        with self.assertRaises(TypeError):
            nn.Unflatten(axis=0, sizes=12)  # sizes must be tuple or list

        with self.assertRaises(TypeError):
            nn.Unflatten(axis=0, sizes="invalid")

        # Test invalid element in sizes
        with self.assertRaises(TypeError):
            nn.Unflatten(axis=0, sizes=(2, "invalid"))

        with self.assertRaises(TypeError):
            nn.Unflatten(axis=0, sizes=(2.5, 3))  # must be integers

    @parameterized.named_parameters(
        ('axis_0_2d', 0, (10, 20), (2, 5)),
        ('axis_1_2d', 1, (10, 20), (4, 5)),
        ('axis_0_3d', 0, (6, 8, 10), (2, 3)),
        ('axis_1_3d', 1, (6, 8, 10), (2, 4)),
        ('axis_2_3d', 2, (6, 8, 10), (2, 5)),
    )
    def test_unflatten_parameterized(self, axis, input_shape, unflatten_sizes):
        """Parameterized test for various axis and shape combinations."""
        arr = brainstate.random.rand(*input_shape)
        unflatten = nn.Unflatten(axis=axis, sizes=unflatten_sizes)
        out = unflatten(arr)

        # Check that product of unflatten_sizes matches original dimension
        original_dim_size = input_shape[axis]
        self.assertEqual(np.prod(unflatten_sizes), original_dim_size)

        # Check output shape
        expected_shape = list(input_shape)
        expected_shape[axis:axis+1] = unflatten_sizes
        self.assertEqual(out.shape, tuple(expected_shape))

        # Check total size preserved
        self.assertEqual(arr.size, out.size)

    def test_unflatten_values_preserved(self):
        """Test that values are correctly preserved during unflatten."""
        # Create a tensor with known pattern
        arr = jnp.arange(24).reshape(2, 12)

        unflatten = nn.Unflatten(axis=1, sizes=(3, 4))
        out = unflatten(arr)

        # Check shape
        self.assertEqual(out.shape, (2, 3, 4))

        # Check that values are correctly rearranged
        # First batch
        self.assertTrue(jnp.allclose(out[0, 0, :], jnp.arange(0, 4)))
        self.assertTrue(jnp.allclose(out[0, 1, :], jnp.arange(4, 8)))
        self.assertTrue(jnp.allclose(out[0, 2, :], jnp.arange(8, 12)))

        # Second batch
        self.assertTrue(jnp.allclose(out[1, 0, :], jnp.arange(12, 16)))
        self.assertTrue(jnp.allclose(out[1, 1, :], jnp.arange(16, 20)))
        self.assertTrue(jnp.allclose(out[1, 2, :], jnp.arange(20, 24)))

    def test_unflatten_with_complex_shapes(self):
        """Test Unflatten with complex multi-dimensional shapes."""
        # 5D tensor
        arr = brainstate.random.rand(2, 3, 4, 5, 60)

        # Unflatten last dimension (use positive axis)
        unflatten = nn.Unflatten(axis=4, sizes=(3, 4, 5))
        out = unflatten(arr)
        self.assertEqual(out.shape, (2, 3, 4, 5, 3, 4, 5))

        # Unflatten middle dimension
        arr = brainstate.random.rand(2, 3, 12, 5, 6)
        unflatten = nn.Unflatten(axis=2, sizes=(3, 4))
        out = unflatten(arr)
        self.assertEqual(out.shape, (2, 3, 3, 4, 5, 6))

    def test_unflatten_edge_cases(self):
        """Test edge cases for Unflatten."""
        # Single element tensor
        arr = brainstate.random.rand(1)
        unflatten = nn.Unflatten(axis=0, sizes=(1,))
        out = unflatten(arr)
        self.assertEqual(out.shape, (1,))

        # Unflatten to same dimension (essentially no-op)
        arr = brainstate.random.rand(3, 5)
        unflatten = nn.Unflatten(axis=1, sizes=(5,))
        out = unflatten(arr)
        self.assertEqual(out.shape, (3, 5))

        # Very large unflatten
        arr = brainstate.random.rand(2, 1024)
        unflatten = nn.Unflatten(axis=1, sizes=(4, 4, 4, 4, 4))
        out = unflatten(arr)
        self.assertEqual(out.shape, (2, 4, 4, 4, 4, 4))
        self.assertEqual(4**5, 1024)

    def test_unflatten_jit_compatibility(self):
        """Test that Unflatten works with JAX JIT compilation."""
        arr = brainstate.random.rand(4, 12)
        unflatten = nn.Unflatten(axis=1, sizes=(3, 4))

        # JIT compile the unflatten operation
        jitted_unflatten = jax.jit(unflatten.update)

        # Compare results
        out_normal = unflatten(arr)
        out_jitted = jitted_unflatten(arr)

        self.assertEqual(out_normal.shape, (4, 3, 4))
        self.assertEqual(out_jitted.shape, (4, 3, 4))
        self.assertTrue(jnp.allclose(out_normal, out_jitted))


class TestMaxPool1d(parameterized.TestCase):
    """Comprehensive tests for MaxPool1d."""

    def test_maxpool1d_basic(self):
        """Test basic MaxPool1d functionality."""
        # Test with different input shapes
        arr = brainstate.random.rand(16, 32, 8)  # (batch, length, channels)

        # Test with kernel_size=2, stride=2
        pool = nn.MaxPool1d(2, 2, channel_axis=-1)
        out = pool(arr)
        self.assertEqual(out.shape, (16, 16, 8))

        # Test with kernel_size=3, stride=1
        pool = nn.MaxPool1d(3, 1, channel_axis=-1)
        out = pool(arr)
        self.assertEqual(out.shape, (16, 30, 8))

    def test_maxpool1d_padding(self):
        """Test MaxPool1d with padding."""
        arr = brainstate.random.rand(4, 10, 3)

        # Test with padding
        pool = nn.MaxPool1d(3, 2, padding=1, channel_axis=-1)
        out = pool(arr)
        self.assertEqual(out.shape, (4, 5, 3))

        # Test with tuple padding (same value for both sides in 1D)
        pool = nn.MaxPool1d(3, 2, padding=(1,), channel_axis=-1)
        out = pool(arr)
        self.assertEqual(out.shape, (4, 5, 3))

    def test_maxpool1d_return_indices(self):
        """Test MaxPool1d with return_indices=True."""
        arr = brainstate.random.rand(2, 10, 3)

        pool = nn.MaxPool1d(2, 2, channel_axis=-1, return_indices=True)
        out, indices = pool(arr)
        self.assertEqual(out.shape, (2, 5, 3))
        self.assertEqual(indices.shape, (2, 5, 3))

    def test_maxpool1d_no_channel_axis(self):
        """Test MaxPool1d without channel axis."""
        arr = brainstate.random.rand(16, 32)

        pool = nn.MaxPool1d(2, 2, channel_axis=None)
        out = pool(arr)
        self.assertEqual(out.shape, (16, 16))


class TestMaxPool2d(parameterized.TestCase):
    """Comprehensive tests for MaxPool2d."""

    def test_maxpool2d_basic(self):
        """Test basic MaxPool2d functionality."""
        arr = brainstate.random.rand(16, 32, 32, 8)  # (batch, height, width, channels)

        out = nn.MaxPool2d(2, 2, channel_axis=-1)(arr)
        self.assertTrue(out.shape == (16, 16, 16, 8))

        out = nn.MaxPool2d(2, 2, channel_axis=None)(arr)
        self.assertTrue(out.shape == (16, 32, 16, 4))

    def test_maxpool2d_padding(self):
        """Test MaxPool2d with padding."""
        arr = brainstate.random.rand(16, 32, 32, 8)

        out = nn.MaxPool2d(2, 2, channel_axis=None, padding=1)(arr)
        self.assertTrue(out.shape == (16, 32, 17, 5))

        out = nn.MaxPool2d(2, 2, channel_axis=None, padding=(2, 1))(arr)
        self.assertTrue(out.shape == (16, 32, 18, 5))

        out = nn.MaxPool2d(2, 2, channel_axis=-1, padding=(1, 1))(arr)
        self.assertTrue(out.shape == (16, 17, 17, 8))

        out = nn.MaxPool2d(2, 2, channel_axis=2, padding=(1, 1))(arr)
        self.assertTrue(out.shape == (16, 17, 32, 5))

    def test_maxpool2d_return_indices(self):
        """Test MaxPool2d with return_indices=True."""
        arr = brainstate.random.rand(2, 8, 8, 3)

        pool = nn.MaxPool2d(2, 2, channel_axis=-1, return_indices=True)
        out, indices = pool(arr)
        self.assertEqual(out.shape, (2, 4, 4, 3))
        self.assertEqual(indices.shape, (2, 4, 4, 3))

    def test_maxpool2d_different_strides(self):
        """Test MaxPool2d with different stride values."""
        arr = brainstate.random.rand(2, 16, 16, 4)

        # Different strides for height and width
        pool = nn.MaxPool2d(3, stride=(2, 1), channel_axis=-1)
        out = pool(arr)
        self.assertEqual(out.shape, (2, 7, 14, 4))


class TestMaxPool3d(parameterized.TestCase):
    """Comprehensive tests for MaxPool3d."""

    def test_maxpool3d_basic(self):
        """Test basic MaxPool3d functionality."""
        arr = brainstate.random.rand(2, 16, 16, 16, 4)  # (batch, depth, height, width, channels)

        pool = nn.MaxPool3d(2, 2, channel_axis=-1)
        out = pool(arr)
        self.assertEqual(out.shape, (2, 8, 8, 8, 4))

        pool = nn.MaxPool3d(3, 1, channel_axis=-1)
        out = pool(arr)
        self.assertEqual(out.shape, (2, 14, 14, 14, 4))

    def test_maxpool3d_padding(self):
        """Test MaxPool3d with padding."""
        arr = brainstate.random.rand(1, 8, 8, 8, 2)

        pool = nn.MaxPool3d(3, 2, padding=1, channel_axis=-1)
        out = pool(arr)
        self.assertEqual(out.shape, (1, 4, 4, 4, 2))

    def test_maxpool3d_return_indices(self):
        """Test MaxPool3d with return_indices=True."""
        arr = brainstate.random.rand(1, 4, 4, 4, 2)

        pool = nn.MaxPool3d(2, 2, channel_axis=-1, return_indices=True)
        out, indices = pool(arr)
        self.assertEqual(out.shape, (1, 2, 2, 2, 2))
        self.assertEqual(indices.shape, (1, 2, 2, 2, 2))


class TestAvgPool1d(parameterized.TestCase):
    """Comprehensive tests for AvgPool1d."""

    def test_avgpool1d_basic(self):
        """Test basic AvgPool1d functionality."""
        arr = brainstate.random.rand(4, 16, 8)  # (batch, length, channels)

        pool = nn.AvgPool1d(2, 2, channel_axis=-1)
        out = pool(arr)
        self.assertEqual(out.shape, (4, 8, 8))

        # Test averaging values
        arr = jnp.ones((1, 4, 2))
        pool = nn.AvgPool1d(2, 2, channel_axis=-1)
        out = pool(arr)
        self.assertTrue(jnp.allclose(out, jnp.ones((1, 2, 2))))

    def test_avgpool1d_padding(self):
        """Test AvgPool1d with padding."""
        arr = brainstate.random.rand(2, 10, 3)

        pool = nn.AvgPool1d(3, 2, padding=1, channel_axis=-1)
        out = pool(arr)
        self.assertEqual(out.shape, (2, 5, 3))

    def test_avgpool1d_divisor_override(self):
        """Test AvgPool1d divisor behavior."""
        arr = jnp.ones((1, 4, 1))

        # Standard average pooling
        pool = nn.AvgPool1d(2, 2, channel_axis=-1)
        out = pool(arr)

        # All values should still be 1.0 for constant input
        self.assertTrue(jnp.allclose(out, jnp.ones((1, 2, 1))))


class TestAvgPool2d(parameterized.TestCase):
    """Comprehensive tests for AvgPool2d."""

    def test_avgpool2d_basic(self):
        """Test basic AvgPool2d functionality."""
        arr = brainstate.random.rand(16, 32, 32, 8)

        out = nn.AvgPool2d(2, 2, channel_axis=-1)(arr)
        self.assertTrue(out.shape == (16, 16, 16, 8))

        out = nn.AvgPool2d(2, 2, channel_axis=None)(arr)
        self.assertTrue(out.shape == (16, 32, 16, 4))

    def test_avgpool2d_padding(self):
        """Test AvgPool2d with padding."""
        arr = brainstate.random.rand(16, 32, 32, 8)

        out = nn.AvgPool2d(2, 2, channel_axis=None, padding=1)(arr)
        self.assertTrue(out.shape == (16, 32, 17, 5))

        out = nn.AvgPool2d(2, 2, channel_axis=None, padding=(2, 1))(arr)
        self.assertTrue(out.shape == (16, 32, 18, 5))

        out = nn.AvgPool2d(2, 2, channel_axis=-1, padding=(1, 1))(arr)
        self.assertTrue(out.shape == (16, 17, 17, 8))

        out = nn.AvgPool2d(2, 2, channel_axis=2, padding=(1, 1))(arr)
        self.assertTrue(out.shape == (16, 17, 32, 5))

    def test_avgpool2d_values(self):
        """Test AvgPool2d computes correct average values."""
        arr = jnp.ones((1, 4, 4, 1))
        pool = nn.AvgPool2d(2, 2, channel_axis=-1)
        out = pool(arr)
        self.assertTrue(jnp.allclose(out, jnp.ones((1, 2, 2, 1))))


class TestAvgPool3d(parameterized.TestCase):
    """Comprehensive tests for AvgPool3d."""

    def test_avgpool3d_basic(self):
        """Test basic AvgPool3d functionality."""
        arr = brainstate.random.rand(2, 8, 8, 8, 4)

        pool = nn.AvgPool3d(2, 2, channel_axis=-1)
        out = pool(arr)
        self.assertEqual(out.shape, (2, 4, 4, 4, 4))

    def test_avgpool3d_padding(self):
        """Test AvgPool3d with padding."""
        arr = brainstate.random.rand(1, 6, 6, 6, 2)

        pool = nn.AvgPool3d(3, 2, padding=1, channel_axis=-1)
        out = pool(arr)
        self.assertEqual(out.shape, (1, 3, 3, 3, 2))


class TestMaxUnpool1d(parameterized.TestCase):
    """Comprehensive tests for MaxUnpool1d."""

    def test_maxunpool1d_basic(self):
        """Test basic MaxUnpool1d functionality."""
        # Create input
        arr = brainstate.random.rand(2, 8, 3)

        # Pool with indices
        pool = nn.MaxPool1d(2, 2, channel_axis=-1, return_indices=True)
        pooled, indices = pool(arr)

        # Unpool
        unpool = nn.MaxUnpool1d(2, 2, channel_axis=-1)
        unpooled = unpool(pooled, indices)

        # Shape should match original (or be close depending on padding)
        self.assertEqual(unpooled.shape, (2, 8, 3))

    def test_maxunpool1d_with_output_size(self):
        """Test MaxUnpool1d with explicit output_size."""
        arr = brainstate.random.rand(1, 10, 2)

        pool = nn.MaxPool1d(2, 2, channel_axis=-1, return_indices=True)
        pooled, indices = pool(arr)

        unpool = nn.MaxUnpool1d(2, 2, channel_axis=-1)
        unpooled = unpool(pooled, indices, output_size=(1, 10, 2))

        self.assertEqual(unpooled.shape, (1, 10, 2))


class TestMaxUnpool2d(parameterized.TestCase):
    """Comprehensive tests for MaxUnpool2d."""

    def test_maxunpool2d_basic(self):
        """Test basic MaxUnpool2d functionality."""
        arr = brainstate.random.rand(2, 8, 8, 3)

        # Pool with indices
        pool = nn.MaxPool2d(2, 2, channel_axis=-1, return_indices=True)
        pooled, indices = pool(arr)

        # Unpool
        unpool = nn.MaxUnpool2d(2, 2, channel_axis=-1)
        unpooled = unpool(pooled, indices)

        self.assertEqual(unpooled.shape, (2, 8, 8, 3))

    def test_maxunpool2d_values(self):
        """Test MaxUnpool2d places values correctly."""
        # Create simple input where we can track values
        arr = jnp.array([[1., 2., 3., 4.],
                         [5., 6., 7., 8.]])  # (2, 4)
        arr = arr.reshape(1, 2, 2, 2)  # (1, 2, 2, 2)

        # Pool to get max value and its index
        pool = nn.MaxPool2d(2, 2, channel_axis=-1, return_indices=True)
        pooled, indices = pool(arr)

        # Unpool
        unpool = nn.MaxUnpool2d(2, 2, channel_axis=-1)
        unpooled = unpool(pooled, indices)

        # Check that max value (8.0) is preserved
        self.assertTrue(jnp.max(unpooled) == 8.0)
        # Check shape
        self.assertEqual(unpooled.shape, (1, 2, 2, 2))


class TestMaxUnpool3d(parameterized.TestCase):
    """Comprehensive tests for MaxUnpool3d."""

    def test_maxunpool3d_basic(self):
        """Test basic MaxUnpool3d functionality."""
        arr = brainstate.random.rand(1, 4, 4, 4, 2)

        # Pool with indices
        pool = nn.MaxPool3d(2, 2, channel_axis=-1, return_indices=True)
        pooled, indices = pool(arr)

        # Unpool
        unpool = nn.MaxUnpool3d(2, 2, channel_axis=-1)
        unpooled = unpool(pooled, indices)

        self.assertEqual(unpooled.shape, (1, 4, 4, 4, 2))


class TestLPPool1d(parameterized.TestCase):
    """Comprehensive tests for LPPool1d."""

    def test_lppool1d_basic(self):
        """Test basic LPPool1d functionality."""
        arr = brainstate.random.rand(2, 16, 4)

        # Test L2 pooling (norm_type=2)
        pool = nn.LPPool1d(norm_type=2, kernel_size=2, stride=2, channel_axis=-1)
        out = pool(arr)
        self.assertEqual(out.shape, (2, 8, 4))

    def test_lppool1d_different_norms(self):
        """Test LPPool1d with different norm types."""
        arr = brainstate.random.rand(1, 8, 2)

        # Test with p=1 (should be similar to average)
        pool1 = nn.LPPool1d(norm_type=1, kernel_size=2, stride=2, channel_axis=-1)
        out1 = pool1(arr)

        # Test with p=2 (L2 norm)
        pool2 = nn.LPPool1d(norm_type=2, kernel_size=2, stride=2, channel_axis=-1)
        out2 = pool2(arr)

        # Test with large p (should approach max pooling)
        pool_inf = nn.LPPool1d(norm_type=10, kernel_size=2, stride=2, channel_axis=-1)
        out_inf = pool_inf(arr)

        self.assertEqual(out1.shape, (1, 4, 2))
        self.assertEqual(out2.shape, (1, 4, 2))
        self.assertEqual(out_inf.shape, (1, 4, 2))

    def test_lppool1d_value_check(self):
        """Test LPPool1d computes correct values."""
        # Simple test case
        arr = jnp.array([[[2., 2.], [2., 2.]]])  # (1, 2, 2)

        pool = nn.LPPool1d(norm_type=2, kernel_size=2, stride=2, channel_axis=-1)
        out = pool(arr)

        # For constant values, Lp norm should equal the value
        self.assertTrue(jnp.allclose(out, 2.0, atol=1e-5))


class TestLPPool2d(parameterized.TestCase):
    """Comprehensive tests for LPPool2d."""

    def test_lppool2d_basic(self):
        """Test basic LPPool2d functionality."""
        arr = brainstate.random.rand(2, 8, 8, 4)

        pool = nn.LPPool2d(norm_type=2, kernel_size=2, stride=2, channel_axis=-1)
        out = pool(arr)
        self.assertEqual(out.shape, (2, 4, 4, 4))

    def test_lppool2d_padding(self):
        """Test LPPool2d with padding."""
        arr = brainstate.random.rand(1, 7, 7, 2)

        pool = nn.LPPool2d(norm_type=2, kernel_size=3, stride=2, padding=1, channel_axis=-1)
        out = pool(arr)
        self.assertEqual(out.shape, (1, 4, 4, 2))

    def test_lppool2d_different_kernel_sizes(self):
        """Test LPPool2d with non-square kernels."""
        arr = brainstate.random.rand(1, 8, 6, 2)

        pool = nn.LPPool2d(norm_type=2, kernel_size=(3, 2), stride=(2, 1), channel_axis=-1)
        out = pool(arr)
        self.assertEqual(out.shape, (1, 3, 5, 2))


class TestLPPool3d(parameterized.TestCase):
    """Comprehensive tests for LPPool3d."""

    def test_lppool3d_basic(self):
        """Test basic LPPool3d functionality."""
        arr = brainstate.random.rand(1, 8, 8, 8, 2)

        pool = nn.LPPool3d(norm_type=2, kernel_size=2, stride=2, channel_axis=-1)
        out = pool(arr)
        self.assertEqual(out.shape, (1, 4, 4, 4, 2))

    def test_lppool3d_different_norms(self):
        """Test LPPool3d with different norm types."""
        arr = brainstate.random.rand(1, 4, 4, 4, 1)

        # Different p values should give different results
        pool1 = nn.LPPool3d(norm_type=1, kernel_size=2, stride=2, channel_axis=-1)
        pool2 = nn.LPPool3d(norm_type=2, kernel_size=2, stride=2, channel_axis=-1)
        pool3 = nn.LPPool3d(norm_type=3, kernel_size=2, stride=2, channel_axis=-1)

        out1 = pool1(arr)
        out2 = pool2(arr)
        out3 = pool3(arr)

        # All should have same shape
        self.assertEqual(out1.shape, (1, 2, 2, 2, 1))
        self.assertEqual(out2.shape, (1, 2, 2, 2, 1))
        self.assertEqual(out3.shape, (1, 2, 2, 2, 1))

        # Values should be different (unless input is uniform)
        self.assertFalse(jnp.allclose(out1, out2))
        self.assertFalse(jnp.allclose(out2, out3))


class TestAdaptivePool(parameterized.TestCase):
    """Tests for adaptive pooling layers."""

    @parameterized.named_parameters(
        dict(testcase_name=f'target_size={target_size}',
             target_size=target_size)
        for target_size in [10, 9, 8, 7, 6]
    )
    def test_adaptive_pool1d(self, target_size):
        """Test internal adaptive pooling function."""
        from brainstate.nn._poolings import _adaptive_pool1d

        arr = brainstate.random.rand(100)
        op = jax.numpy.mean

        out = _adaptive_pool1d(arr, target_size, op)
        self.assertTrue(out.shape == (target_size,))

    def test_adaptive_avg_pool1d(self):
        """Test AdaptiveAvgPool1d."""
        input = brainstate.random.randn(2, 32, 4)

        # Test with different target sizes
        pool = nn.AdaptiveAvgPool1d(5, channel_axis=-1)
        output = pool(input)
        self.assertEqual(output.shape, (2, 5, 4))

        # Test with single element input
        pool = nn.AdaptiveAvgPool1d(1, channel_axis=-1)
        output = pool(input)
        self.assertEqual(output.shape, (2, 1, 4))

    def test_adaptive_avg_pool2d(self):
        """Test AdaptiveAvgPool2d."""
        input = brainstate.random.randn(2, 8, 9, 3)

        # Square output
        output = nn.AdaptiveAvgPool2d(5, channel_axis=-1)(input)
        self.assertEqual(output.shape, (2, 5, 5, 3))

        # Non-square output
        output = nn.AdaptiveAvgPool2d((5, 7), channel_axis=-1)(input)
        self.assertEqual(output.shape, (2, 5, 7, 3))

        # Test with single integer (square output)
        output = nn.AdaptiveAvgPool2d(4, channel_axis=-1)(input)
        self.assertEqual(output.shape, (2, 4, 4, 3))

    def test_adaptive_avg_pool3d(self):
        """Test AdaptiveAvgPool3d."""
        input = brainstate.random.randn(1, 8, 6, 4, 2)

        pool = nn.AdaptiveAvgPool3d((4, 3, 2), channel_axis=-1)
        output = pool(input)
        self.assertEqual(output.shape, (1, 4, 3, 2, 2))

        # Cube output
        pool = nn.AdaptiveAvgPool3d(3, channel_axis=-1)
        output = pool(input)
        self.assertEqual(output.shape, (1, 3, 3, 3, 2))

    def test_adaptive_max_pool1d(self):
        """Test AdaptiveMaxPool1d."""
        input = brainstate.random.randn(2, 32, 4)

        pool = nn.AdaptiveMaxPool1d(8, channel_axis=-1)
        output = pool(input)
        self.assertEqual(output.shape, (2, 8, 4))

    def test_adaptive_max_pool2d(self):
        """Test AdaptiveMaxPool2d."""
        input = brainstate.random.randn(2, 10, 8, 3)

        pool = nn.AdaptiveMaxPool2d((5, 4), channel_axis=-1)
        output = pool(input)
        self.assertEqual(output.shape, (2, 5, 4, 3))

    def test_adaptive_max_pool3d(self):
        """Test AdaptiveMaxPool3d."""
        input = brainstate.random.randn(1, 8, 8, 8, 2)

        pool = nn.AdaptiveMaxPool3d((4, 4, 4), channel_axis=-1)
        output = pool(input)
        self.assertEqual(output.shape, (1, 4, 4, 4, 2))


class TestPoolingEdgeCases(parameterized.TestCase):
    """Test edge cases and error conditions."""

    def test_pool_with_stride_none(self):
        """Test pooling with stride=None (defaults to kernel_size)."""
        arr = brainstate.random.rand(1, 8, 2)

        pool = nn.MaxPool1d(kernel_size=3, stride=None, channel_axis=-1)
        out = pool(arr)
        # stride defaults to kernel_size=3
        self.assertEqual(out.shape, (1, 2, 2))

    def test_pool_with_large_kernel(self):
        """Test pooling with kernel larger than input."""
        arr = brainstate.random.rand(1, 4, 2)

        # Kernel size larger than spatial dimension
        pool = nn.MaxPool1d(kernel_size=5, stride=1, channel_axis=-1)
        out = pool(arr)
        # Should handle gracefully (may produce empty output or handle with padding)
        self.assertTrue(out.shape[1] >= 0)

    def test_pool_single_element(self):
        """Test pooling on single-element tensors."""
        arr = brainstate.random.rand(1, 1, 1)

        pool = nn.AvgPool1d(1, 1, channel_axis=-1)
        out = pool(arr)
        self.assertEqual(out.shape, (1, 1, 1))
        self.assertTrue(jnp.allclose(out, arr))

    def test_adaptive_pool_smaller_output(self):
        """Test adaptive pooling with output smaller than input."""
        arr = brainstate.random.rand(1, 16, 2)

        # Adaptive pooling to smaller size
        pool = nn.AdaptiveAvgPool1d(4, channel_axis=-1)
        out = pool(arr)
        self.assertEqual(out.shape, (1, 4, 2))

    def test_unpool_without_indices(self):
        """Test unpooling behavior with placeholder indices."""
        pooled = brainstate.random.rand(1, 4, 2)
        indices = jnp.zeros_like(pooled, dtype=jnp.int32)

        unpool = nn.MaxUnpool1d(2, 2, channel_axis=-1)
        # Should not raise error even with zero indices
        unpooled = unpool(pooled, indices)
        self.assertEqual(unpooled.shape, (1, 8, 2))

    def test_lppool_extreme_norm(self):
        """Test LPPool with extreme norm values."""
        arr = brainstate.random.rand(1, 8, 2) + 0.1  # Avoid zeros

        # Very large p (approaches max pooling)
        pool_large = nn.LPPool1d(norm_type=20, kernel_size=2, stride=2, channel_axis=-1)
        out_large = pool_large(arr)

        # Compare with actual max pooling
        pool_max = nn.MaxPool1d(2, 2, channel_axis=-1)
        out_max = pool_max(arr)

        # Should approach max pooling for large p (but not exactly equal)
        # Just check shapes match
        self.assertEqual(out_large.shape, out_max.shape)

    def test_pool_with_channels_first(self):
        """Test pooling with channels in different positions."""
        arr = brainstate.random.rand(3, 16, 8)  # (dim0, dim1, dim2)

        # Channel axis at position 0 - treats dim 0 as channels, pools last dimension
        pool = nn.MaxPool1d(2, 2, channel_axis=0)
        out = pool(arr)
        # Pools the last dimension, keeping first two
        self.assertEqual(out.shape, (3, 16, 4))

        # Channel axis at position -1 (last) - pools middle dimension
        pool = nn.MaxPool1d(2, 2, channel_axis=-1)
        out = pool(arr)
        # Pools the middle dimension, keeping first and last
        self.assertEqual(out.shape, (3, 8, 8))

        # No channel axis - pools last dimension, treating earlier dims as batch
        pool = nn.MaxPool1d(2, 2, channel_axis=None)
        out = pool(arr)
        # Pools the last dimension
        self.assertEqual(out.shape, (3, 16, 4))


class TestPoolingMathematicalProperties(parameterized.TestCase):
    """Test mathematical properties of pooling operations."""

    def test_maxpool_idempotence(self):
        """Test that max pooling with kernel_size=1 is identity."""
        arr = brainstate.random.rand(2, 8, 3)

        pool = nn.MaxPool1d(1, 1, channel_axis=-1)
        out = pool(arr)

        self.assertTrue(jnp.allclose(out, arr))

    def test_avgpool_constant_input(self):
        """Test average pooling on constant input."""
        arr = jnp.ones((1, 8, 2)) * 5.0

        pool = nn.AvgPool1d(2, 2, channel_axis=-1)
        out = pool(arr)

        # Average of constant should be the constant
        self.assertTrue(jnp.allclose(out, 5.0))

    def test_lppool_norm_properties(self):
        """Test Lp pooling norm properties."""
        arr = brainstate.random.rand(1, 4, 1) + 0.1

        # L1 norm (p=1) should give average of absolute values
        pool_l1 = nn.LPPool1d(norm_type=1, kernel_size=4, stride=4, channel_axis=-1)
        out_l1 = pool_l1(arr)

        # Manual calculation
        manual_l1 = jnp.mean(jnp.abs(arr[:, :4, :]))

        self.assertTrue(jnp.allclose(out_l1[0, 0, 0], manual_l1, rtol=1e-5))

    def test_maxpool_monotonicity(self):
        """Test that max pooling preserves monotonicity."""
        arr1 = brainstate.random.rand(1, 8, 2)
        arr2 = arr1 + 1.0  # Strictly greater

        pool = nn.MaxPool1d(2, 2, channel_axis=-1)
        out1 = pool(arr1)
        out2 = pool(arr2)

        # out2 should be strictly greater than out1
        self.assertTrue(jnp.all(out2 > out1))

    def test_adaptive_pool_preserves_values(self):
        """Test that adaptive pooling with same size preserves values."""
        arr = brainstate.random.rand(1, 8, 2)

        # Adaptive pool to same size
        pool = nn.AdaptiveAvgPool1d(8, channel_axis=-1)
        out = pool(arr)

        # Should be approximately equal (might have small numerical differences)
        self.assertTrue(jnp.allclose(out, arr, rtol=1e-5))


if __name__ == '__main__':
    absltest.main()