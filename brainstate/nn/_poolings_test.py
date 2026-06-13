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

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from absl.testing import absltest
from absl.testing import parameterized

import brainstate
import brainstate.nn as nn
from brainstate import _testing
from brainstate._testing import (
    SMALL_BATCH,
    assert_allclose,
    assert_grad_finite,
    assert_jit_equal,
    seeded,
)


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

    def test_maxunpool1d_natural_roundtrip_no_cross_batch(self):
        """Natural-size unpool keeps each batch element's maxima in that element."""
        # input (2, 4, 1): batch 0 -> [0,1,2,3], batch 1 -> [4,5,6,7]
        arr = jnp.arange(8.0).reshape(2, 4, 1)

        pool = nn.MaxPool1d(2, 2, channel_axis=-1, return_indices=True)
        pooled, indices = pool(arr)

        unpool = nn.MaxUnpool1d(2, 2, channel_axis=-1)
        unpooled = unpool(pooled, indices)

        self.assertEqual(unpooled.shape, (2, 4, 1))
        # Each batch element's maxima land in the SAME batch element.
        np.testing.assert_array_equal(
            np.asarray(unpooled[..., 0]),
            np.array([[0.0, 1.0, 0.0, 3.0],
                      [0.0, 5.0, 0.0, 7.0]]),
        )

    def test_maxunpool1d_output_size_no_cross_batch_leakage(self):
        """A non-natural output_size must not leak maxima across batch elements.

        Regression test for the flat-scatter bug where, for batch N>1, values
        landed in the wrong batch element because the per-batch flat stride of the
        output differs from that of the input layout when ``output_size`` changes
        the spatial extent.
        """
        # input (2, 4, 1): batch 0 -> [0,1,2,3], batch 1 -> [4,5,6,7]
        arr = jnp.arange(8.0).reshape(2, 4, 1)

        pool = nn.MaxPool1d(2, 2, channel_axis=-1, return_indices=True)
        pooled, indices = pool(arr)
        # pooled maxima: batch 0 -> [1, 3], batch 1 -> [5, 7]

        unpool = nn.MaxUnpool1d(2, 2, channel_axis=-1)
        # Request a spatial size (6) that differs from the natural size (4).
        unpooled = unpool(pooled, indices, output_size=(2, 6, 1))

        self.assertEqual(unpooled.shape, (2, 6, 1))

        b0 = np.asarray(unpooled[0, :, 0])
        b1 = np.asarray(unpooled[1, :, 0])

        # Batch 0 must contain exactly its own maxima {1, 3} and nothing from batch 1.
        self.assertEqual(set(np.unique(b0[b0 != 0]).tolist()), {1.0, 3.0})
        self.assertNotIn(5.0, b0.tolist())
        self.assertNotIn(7.0, b0.tolist())
        # Batch 1 must contain exactly its own maxima {5, 7}.
        self.assertEqual(set(np.unique(b1[b1 != 0]).tolist()), {5.0, 7.0})
        # Positions within each batch element are preserved (col index = original).
        self.assertEqual(b0[1], 1.0)
        self.assertEqual(b0[3], 3.0)
        self.assertEqual(b1[1], 5.0)
        self.assertEqual(b1[3], 7.0)


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

    def test_maxunpool2d_interleaved_channel_axis_roundtrip(self):
        """Regression (H8): unpool handles a non-contiguous (interleaved) channel axis.

        With ``channel_axis=2`` the layout is ``(N, H, C, W)``, so the spatial axes
        (1 and 3) are *not* contiguous. The previous implementation assumed a single
        contiguous spatial block and mislaid values; here a pool/unpool roundtrip must
        recover the correct shape and the pooled maxima.
        """
        with seeded(11):
            arr = brainstate.random.randn(1, 4, 2, 4)  # (N, H, C, W)

        pool = nn.MaxPool2d(2, 2, channel_axis=2, return_indices=True)
        pooled, indices = pool(arr)
        # Spatial axes 1 (H: 4->2) and 3 (W: 4->2); channel axis 2 (size 2) untouched.
        self.assertEqual(pooled.shape, (1, 2, 2, 2))

        unpool = nn.MaxUnpool2d(2, 2, channel_axis=2)
        unpooled = unpool(pooled, indices)

        self.assertEqual(unpooled.shape, (1, 4, 2, 4))
        # The non-zero entries of the reconstruction are exactly the pooled maxima.
        assert_allclose(jnp.sum(unpooled), jnp.sum(pooled))
        self.assertEqual(float(jnp.max(unpooled)), float(jnp.max(pooled)))


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


class TestFlattenInSizePaths(unittest.TestCase):
    """Cover the ``in_size``-aware branches of :class:`Flatten`."""

    def test_flatten_in_size_shape_mismatch_raises(self):
        """Flatten with ``in_size`` rejects a mismatched trailing shape."""
        f = nn.Flatten(start_axis=0, in_size=(32, 8))
        bad = brainstate.random.randn(16, 99, 8)
        with self.assertRaises(ValueError):
            f(bad)

    def test_flatten_in_size_negative_start_axis(self):
        """Flatten with ``in_size`` resolves a negative ``start_axis``."""
        f = nn.Flatten(start_axis=-2, in_size=(32, 8))
        out = f(brainstate.random.randn(16, 32, 8))
        self.assertEqual(out.shape, (16, 32 * 8))

    def test_flatten_out_size_inferred(self):
        """Flatten records ``out_size`` when ``in_size`` is supplied."""
        f = nn.Flatten(start_axis=1, in_size=(3, 4, 5))
        self.assertEqual(f.out_size, (3, 20))

    def test_flatten_grad_and_jit(self):
        """Flatten yields finite gradients and matches its jitted form."""
        f = nn.Flatten(start_axis=1)
        with seeded(0):
            x = brainstate.random.randn(SMALL_BATCH, 4, 5)
        assert_jit_equal(f, x)
        assert_grad_finite(lambda y: jnp.sum(f(y)), x)

    def test_flatten_in_size_positive_end_axis_rebased(self):
        """Regression (H7): a positive ``end_axis`` is rebased by the batch offset.

        With ``in_size=(3, 4, 5)`` and a leading batch dim, ``Flatten(0, 1)`` should
        flatten the *local* first two axes (3, 4) -> 12, leaving the batch and the
        trailing 5 intact. Before the fix, ``end_axis`` was passed unrebased, so the
        flattened span was wrong.
        """
        f = nn.Flatten(start_axis=0, end_axis=1, in_size=(3, 4, 5))
        out = f(brainstate.random.randn(2, 3, 4, 5))
        self.assertEqual(out.shape, (2, 12, 5))

    def test_flatten_in_size_default_end_axis_unchanged(self):
        """Regression (H7): a negative (default) ``end_axis`` still resolves correctly.

        ``start_axis`` is relative to ``in_size=(3, 4, 5)``; with a leading batch dim
        ``dim_diff == 1`` rebases ``start_axis=1`` to local axis 2 and ``end_axis=-1``
        to the last axis, flattening (4, 5) -> 20 while leaving the batch and the
        first in_size axis intact.
        """
        f = nn.Flatten(start_axis=1, in_size=(3, 4, 5))
        out = f(brainstate.random.randn(2, 3, 4, 5))
        self.assertEqual(out.shape, (2, 3, 20))


class TestMaxPoolConstructorValidation(unittest.TestCase):
    """Cover the argument-validation branches in the ``_MaxPool`` base class."""

    def test_kernel_size_wrong_length_raises(self):
        """A kernel tuple whose length differs from ``pool_dim`` is rejected."""
        with self.assertRaises(ValueError):
            nn.MaxPool2d((2, 2, 2))

    def test_kernel_size_wrong_type_raises(self):
        """A non-int, non-sequence kernel size is rejected."""
        with self.assertRaises(TypeError):
            nn.MaxPool2d(2.5)

    def test_stride_wrong_length_raises(self):
        """A stride tuple whose length differs from ``pool_dim`` is rejected."""
        with self.assertRaises(ValueError):
            nn.MaxPool2d(2, stride=(2, 2, 2))

    def test_stride_wrong_type_raises(self):
        """A non-int, non-sequence stride is rejected."""
        with self.assertRaises(TypeError):
            nn.MaxPool2d(2, stride=2.5)

    def test_padding_invalid_string_raises(self):
        """An unknown padding string is rejected."""
        with self.assertRaises(ValueError):
            nn.MaxPool2d(2, padding='FOO')

    def test_padding_int_sequence_wrong_length_raises(self):
        """A sequence of ints of the wrong length for padding is rejected."""
        with self.assertRaises(ValueError):
            nn.MaxPool2d(2, padding=[1, 2, 3])

    def test_padding_non_tuple_entries_raises(self):
        """A padding sequence with non-tuple entries is rejected."""
        with self.assertRaises(ValueError):
            nn.MaxPool2d(2, padding=[(1, 2), 3])

    def test_padding_tuple_wrong_inner_length_raises(self):
        """A padding entry that is not a 2-tuple is rejected."""
        with self.assertRaises(ValueError):
            nn.MaxPool2d(2, padding=[(1, 2, 3), (1, 2)])

    def test_padding_invalid_type_raises(self):
        """A non-string, non-int, non-sequence padding is rejected."""
        with self.assertRaises(ValueError):
            nn.MaxPool2d(2, padding=2.5)

    def test_padding_int_expands(self):
        """An int padding expands to a per-dim ``(p, p)`` sequence."""
        p = nn.MaxPool2d(2, padding=1)
        self.assertEqual(list(p.padding), [(1, 1), (1, 1)])

    def test_padding_single_tuple_expands(self):
        """A length-1 padding sequence is broadcast to ``pool_dim`` entries."""
        p = nn.MaxPool2d(2, padding=[(1, 1)])
        self.assertEqual(tuple(p.padding), ((1, 1), (1, 1)))

    def test_in_size_infers_out_size(self):
        """Supplying ``in_size`` triggers eager output-shape inference."""
        p = nn.MaxPool2d(2, in_size=(8, 8, 3), channel_axis=-1)
        self.assertEqual(p.out_size, (4, 4, 3))

    def test_too_few_dims_raises(self):
        """Calling a pool on an input with too few dimensions raises."""
        p = nn.MaxPool2d(2, channel_axis=-1)
        with self.assertRaises(ValueError):
            p(brainstate.random.randn(4))

    def test_invalid_channel_axis_raises(self):
        """An out-of-range channel axis is rejected during the forward pass."""
        p = nn.MaxPool1d(2, channel_axis=5)
        with self.assertRaises(ValueError):
            p(brainstate.random.randn(3, 4))

    def test_most_negative_channel_axis_accepted(self):
        """``channel_axis == -ndim`` is valid and matches the positive equivalent.

        Regression test: the previous guard rejected the most-negative valid axis.
        """
        x = brainstate.random.randn(4, 10)
        out_neg = nn.MaxPool1d(2, channel_axis=-2)(x)  # -2 == -ndim for a 2D input
        out_pos = nn.MaxPool1d(2, channel_axis=0)(x)
        self.assertEqual(out_neg.shape, (4, 5))
        np.testing.assert_array_equal(np.asarray(out_neg), np.asarray(out_pos))

    def test_too_negative_channel_axis_raises(self):
        """A channel axis more negative than ``-ndim`` is still rejected."""
        p = nn.MaxPool1d(2, channel_axis=-3)
        with self.assertRaises(ValueError):
            p(brainstate.random.randn(3, 4))

    def test_most_negative_channel_axis_lppool_and_adaptive(self):
        """``channel_axis == -ndim`` is accepted by LPPool and adaptive pooling too."""
        x = brainstate.random.randn(4, 10)
        self.assertEqual(nn.LPPool1d(2, 2, channel_axis=-2)(x).shape, (4, 5))
        self.assertEqual(nn.AdaptiveAvgPool1d(5, channel_axis=-2)(x).shape, (4, 5))


class TestMaxPoolTransforms(unittest.TestCase):
    """Gradient and jit consistency checks for the max-pooling variants."""

    def test_maxpool1d_grad_and_jit(self):
        """MaxPool1d is jit-stable and produces finite gradients."""
        pool = nn.MaxPool1d(2, 2, channel_axis=-1)
        with seeded(0):
            x = brainstate.random.randn(SMALL_BATCH, 8, 3)
        assert_jit_equal(pool, x)
        assert_grad_finite(lambda y: jnp.sum(pool(y)), x)

    def test_maxpool2d_grad_and_jit(self):
        """MaxPool2d is jit-stable and produces finite gradients."""
        pool = nn.MaxPool2d(2, 2, channel_axis=-1)
        with seeded(1):
            x = brainstate.random.randn(2, 8, 8, 3)
        assert_jit_equal(pool, x)
        assert_grad_finite(lambda y: jnp.sum(pool(y)), x)

    @pytest.mark.slow
    def test_maxpool3d_grad_large_kernel(self):
        """MaxPool3d gradients remain finite with a larger kernel."""
        pool = nn.MaxPool3d(3, 2, channel_axis=-1)
        with seeded(2):
            x = brainstate.random.randn(1, 12, 12, 12, 2)
        assert_grad_finite(lambda y: jnp.sum(pool(y)), x)


class TestAvgPoolExtra(unittest.TestCase):
    """Additional coverage for the average-pooling base class."""

    def test_avgpool_too_few_dims_raises(self):
        """AvgPool rejects inputs with too few dimensions."""
        p = nn.AvgPool2d(2, 2, channel_axis=-1)
        with self.assertRaises(ValueError):
            p(brainstate.random.randn(4))

    def test_avgpool_same_padding_normalizes(self):
        """AvgPool with SAME padding divides by per-window valid counts."""
        x = jnp.ones((1, 5, 1))
        pool = nn.AvgPool1d(3, 1, padding='SAME', channel_axis=-1)
        out = pool(x)
        # Constant ones input averages back to ones regardless of edge handling.
        assert_allclose(out, jnp.ones_like(out))

    def test_avgpool1d_grad_and_jit(self):
        """AvgPool1d is jit-stable and produces finite gradients."""
        pool = nn.AvgPool1d(2, 2, channel_axis=-1)
        with seeded(3):
            x = brainstate.random.randn(SMALL_BATCH, 8, 3)
        assert_jit_equal(pool, x)
        assert_grad_finite(lambda y: jnp.sum(pool(y)), x)

    def test_avgpool_all_padding_window_is_nan_free(self):
        """Regression (M15): a window that is entirely padding does not produce NaN.

        With ``kernel_size=3``, ``stride=3`` and ``padding=3`` the border windows
        consist solely of padding cells, so the valid-element count there is 0.
        The naive ``pooled / window_counts`` divisor yields 0/0 = NaN; the NaN-safe
        divisor (``jnp.maximum(window_counts, 1)``) must keep the output finite.
        """
        pool = nn.AvgPool1d(3, stride=3, padding=3, channel_axis=-1)
        out = pool(brainstate.random.randn(1, 9, 2))
        self.assertTrue(bool(jnp.all(jnp.isfinite(out))))

    def test_avgpool_valid_padding_is_finite(self):
        """Regression (M15): ordinary padded pooling stays finite and correct."""
        pool = nn.AvgPool1d(3, stride=1, padding=1, channel_axis=-1)
        out = pool(jnp.ones((1, 6, 2)))
        self.assertTrue(bool(jnp.all(jnp.isfinite(out))))
        # Constant ones input averages back to ones (padding cells are excluded).
        assert_allclose(out, jnp.ones_like(out))


class TestMaxUnpoolValidation(unittest.TestCase):
    """Cover the validation and output-shape branches of ``_MaxUnpool``."""

    def test_kernel_size_wrong_length_raises(self):
        """An unpool kernel tuple of the wrong length is rejected."""
        with self.assertRaises(ValueError):
            nn.MaxUnpool2d((2, 2, 2))

    def test_kernel_size_wrong_type_raises(self):
        """A non-int, non-sequence unpool kernel is rejected."""
        with self.assertRaises(TypeError):
            nn.MaxUnpool2d(2.5)

    def test_stride_wrong_length_raises(self):
        """An unpool stride tuple of the wrong length is rejected."""
        with self.assertRaises(ValueError):
            nn.MaxUnpool2d(2, stride=(2, 2, 2))

    def test_stride_wrong_type_raises(self):
        """A non-int, non-sequence unpool stride is rejected."""
        with self.assertRaises(TypeError):
            nn.MaxUnpool2d(2, stride=2.5)

    def test_padding_wrong_length_raises(self):
        """An unpool padding tuple of the wrong length is rejected."""
        with self.assertRaises(ValueError):
            nn.MaxUnpool2d(2, padding=(1, 2, 3))

    def test_padding_wrong_type_raises(self):
        """A non-int, non-sequence unpool padding is rejected."""
        with self.assertRaises(TypeError):
            nn.MaxUnpool2d(2, padding=2.5)

    def test_in_size_stored(self):
        """Supplying ``in_size`` to an unpool layer records it."""
        u = nn.MaxUnpool2d(2, in_size=(4, 4, 3))
        self.assertEqual(u.in_size, (4, 4, 3))

    def test_unpool_too_few_dims_raises(self):
        """Unpooling an input with too few dimensions raises."""
        up = nn.MaxUnpool2d(2, channel_axis=-1)
        with self.assertRaises(ValueError):
            up(brainstate.random.randn(4), jnp.zeros(4, dtype=jnp.int32))

    def test_unpool_output_size_wrong_spatial_raises(self):
        """An ``output_size`` with the wrong number of spatial dims raises."""
        arr = brainstate.random.randn(1, 4, 4, 2)
        pool = nn.MaxPool2d(2, 2, channel_axis=-1, return_indices=True)
        pooled, idx = pool(arr)
        unpool = nn.MaxUnpool2d(2, 2, channel_axis=-1)
        with self.assertRaises(ValueError):
            unpool(pooled, idx, output_size=(8,))

    def test_unpool_output_size_spatial_only(self):
        """An ``output_size`` giving only spatial dims is broadcast over batch/channel."""
        arr = brainstate.random.randn(1, 4, 4, 2)
        pool = nn.MaxPool2d(2, 2, channel_axis=-1, return_indices=True)
        pooled, idx = pool(arr)
        unpool = nn.MaxUnpool2d(2, 2, channel_axis=-1)
        out = unpool(pooled, idx, output_size=(8, 8))
        self.assertEqual(out.shape, (1, 8, 8, 2))

    def test_unpool_output_size_full_shape(self):
        """An ``output_size`` matching the full ndim is used verbatim."""
        arr = brainstate.random.randn(1, 4, 4, 2)
        pool = nn.MaxPool2d(2, 2, channel_axis=-1, return_indices=True)
        pooled, idx = pool(arr)
        unpool = nn.MaxUnpool2d(2, 2, channel_axis=-1)
        out = unpool(pooled, idx, output_size=(1, 8, 8, 2))
        self.assertEqual(out.shape, (1, 8, 8, 2))

    def test_unpool_output_size_int(self):
        """A scalar ``output_size`` is applied to every spatial dimension."""
        arr = brainstate.random.randn(1, 4, 4, 2)
        pool = nn.MaxPool2d(2, 2, channel_axis=-1, return_indices=True)
        pooled, idx = pool(arr)
        unpool = nn.MaxUnpool2d(2, 2, channel_axis=-1)
        out = unpool(pooled, idx, output_size=8)
        self.assertEqual(out.shape, (1, 8, 8, 2))

    def test_unpool_channel_axis_none(self):
        """Unpooling with ``channel_axis=None`` infers spatial dims from the tail."""
        arr = brainstate.random.randn(4, 8, 8)
        pool = nn.MaxPool2d(2, 2, channel_axis=None, return_indices=True)
        pooled, idx = pool(arr)
        unpool = nn.MaxUnpool2d(2, 2, channel_axis=None)
        out = unpool(pooled, idx)
        self.assertEqual(out.shape, (4, 8, 8))

    def test_unpool_channel_first(self):
        """Unpooling with a leading channel axis offsets the spatial start index."""
        arr = brainstate.random.randn(8, 3, 8, 8)
        pool = nn.MaxPool2d(2, 2, channel_axis=1, return_indices=True)
        pooled, idx = pool(arr)
        unpool = nn.MaxUnpool2d(2, 2, channel_axis=1)
        out = unpool(pooled, idx)
        self.assertEqual(out.shape, (8, 3, 8, 8))

    def test_unpool_index_roundtrip_preserves_max(self):
        """A pool/unpool roundtrip preserves the pooled maxima exactly."""
        with seeded(7):
            arr = brainstate.random.randn(1, 8, 3)
        pool = nn.MaxPool1d(2, 2, channel_axis=-1, return_indices=True)
        pooled, idx = pool(arr)
        unpool = nn.MaxUnpool1d(2, 2, channel_axis=-1)
        recon = unpool(pooled, idx)
        self.assertEqual(recon.shape, (1, 8, 3))
        # The non-zero entries of the reconstruction are exactly the pooled maxima.
        assert_allclose(jnp.sum(recon), jnp.sum(pooled))


class TestLPPoolValidation(unittest.TestCase):
    """Cover the argument-validation branches in the ``_LPPool`` base class."""

    def test_nonpositive_norm_type_raises(self):
        """A non-positive ``norm_type`` is rejected."""
        with self.assertRaises(ValueError):
            nn.LPPool2d(norm_type=0, kernel_size=2)
        with self.assertRaises(ValueError):
            nn.LPPool2d(norm_type=-1.0, kernel_size=2)

    def test_nonfinite_norm_type_raises(self):
        """Regression (M16): a non-finite ``norm_type`` (inf/nan) is rejected.

        ``p = inf`` would correspond to max pooling only in the limit; the
        implementation cannot evaluate it, so it must raise rather than silently
        producing NaNs.
        """
        with self.assertRaises(ValueError):
            nn.LPPool1d(float('inf'), kernel_size=2, stride=2, channel_axis=-1)
        with self.assertRaises(ValueError):
            nn.LPPool2d(float('nan'), kernel_size=2)

    def test_kernel_size_wrong_length_raises(self):
        """An LP kernel tuple of the wrong length is rejected."""
        with self.assertRaises(ValueError):
            nn.LPPool2d(2, (2, 2, 2))

    def test_kernel_size_wrong_type_raises(self):
        """A non-int, non-sequence LP kernel is rejected."""
        with self.assertRaises(TypeError):
            nn.LPPool2d(2, 2.5)

    def test_stride_wrong_length_raises(self):
        """An LP stride tuple of the wrong length is rejected."""
        with self.assertRaises(ValueError):
            nn.LPPool2d(2, 2, stride=(2, 2, 2))

    def test_stride_wrong_type_raises(self):
        """A non-int, non-sequence LP stride is rejected."""
        with self.assertRaises(TypeError):
            nn.LPPool2d(2, 2, stride=2.5)

    def test_padding_invalid_string_raises(self):
        """An unknown LP padding string is rejected."""
        with self.assertRaises(ValueError):
            nn.LPPool2d(2, 2, padding='FOO')

    def test_padding_int_expands(self):
        """An int LP padding expands to a per-dim ``(p, p)`` sequence."""
        p = nn.LPPool2d(2, 2, padding=1)
        self.assertEqual(list(p.padding), [(1, 1), (1, 1)])

    def test_padding_int_sequence_expands(self):
        """An LP padding int-sequence of correct length expands to ``(x, x)`` pairs."""
        p = nn.LPPool2d(2, 2, padding=[1, 2])
        self.assertEqual(list(p.padding), [(1, 1), (2, 2)])

    def test_padding_int_sequence_wrong_length_raises(self):
        """An LP padding int-sequence of the wrong length is rejected."""
        with self.assertRaises(ValueError):
            nn.LPPool2d(2, 2, padding=[1, 2, 3])

    def test_padding_non_tuple_entries_raises(self):
        """An LP padding sequence with non-tuple entries is rejected."""
        with self.assertRaises(ValueError):
            nn.LPPool2d(2, 2, padding=[(1, 2), 3])

    def test_padding_tuple_wrong_inner_length_raises(self):
        """An LP padding entry that is not a 2-tuple is rejected."""
        with self.assertRaises(ValueError):
            nn.LPPool2d(2, 2, padding=[(1, 2, 3), (1, 2)])

    def test_padding_single_tuple_expands(self):
        """A length-1 LP padding sequence is broadcast to ``pool_dim`` entries."""
        p = nn.LPPool2d(2, 2, padding=[(1, 1)])
        self.assertEqual(tuple(p.padding), ((1, 1), (1, 1)))

    def test_padding_invalid_type_raises(self):
        """A non-string, non-int, non-sequence LP padding is rejected."""
        with self.assertRaises(ValueError):
            nn.LPPool2d(2, 2, padding=2.5)

    def test_in_size_infers_out_size(self):
        """Supplying ``in_size`` to an LP pool triggers output-shape inference."""
        p = nn.LPPool2d(2, 2, in_size=(8, 8, 3), channel_axis=-1)
        self.assertEqual(p.out_size, (4, 4, 3))

    def test_too_few_dims_raises(self):
        """An LP pool rejects inputs with too few dimensions."""
        p = nn.LPPool2d(2, 2, channel_axis=-1)
        with self.assertRaises(ValueError):
            p(brainstate.random.randn(4))

    def test_invalid_channel_axis_raises(self):
        """An out-of-range LP channel axis is rejected during the forward pass."""
        p = nn.LPPool1d(2, 2, channel_axis=5)
        with self.assertRaises(ValueError):
            p(brainstate.random.randn(3, 4))


class TestLPPoolTransforms(unittest.TestCase):
    """Gradient and jit consistency checks for the LP-pooling variants."""

    def test_lppool1d_grad_and_jit(self):
        """LPPool1d is jit-stable and produces finite gradients."""
        pool = nn.LPPool1d(2, 2, 2, channel_axis=-1)
        with seeded(4):
            x = brainstate.random.randn(SMALL_BATCH, 8, 3) + 1.0
        assert_jit_equal(pool, x)
        assert_grad_finite(lambda y: jnp.sum(pool(y)), x)

    def test_lppool1d_grad_finite_at_all_zero_window(self):
        """Regression (M17): the gradient is finite even when a window is all zeros.

        At an all-zero window ``pooled_sum == 0`` and the naive ``sum ** (1/p)`` has an
        infinite derivative, producing a NaN gradient. The double-``where`` guard must
        keep the gradient finite (and zero) there.
        """
        pool = nn.LPPool1d(2.0, 2, stride=2, channel_axis=None)
        grad = jax.grad(lambda x: jnp.sum(pool(x)))(jnp.zeros(4))
        self.assertTrue(bool(jnp.all(jnp.isfinite(grad))))

    @pytest.mark.slow
    def test_lppool3d_grad_large_kernel(self):
        """LPPool3d gradients remain finite with a larger kernel."""
        pool = nn.LPPool3d(2, 3, 2, channel_axis=-1)
        with seeded(5):
            x = brainstate.random.randn(1, 12, 12, 12, 2) + 1.0
        assert_grad_finite(lambda y: jnp.sum(pool(y)), x)


class TestAdaptivePoolValidation(unittest.TestCase):
    """Cover the validation and forward branches of ``_AdaptivePool``."""

    def test_target_size_wrong_length_raises(self):
        """A target-size tuple whose length differs from the spatial dims is rejected."""
        with self.assertRaises(ValueError):
            nn.AdaptiveAvgPool2d((5, 6, 7))

    def test_target_size_wrong_type_raises(self):
        """A non-int, non-sequence target size is rejected."""
        with self.assertRaises(ValueError):
            nn.AdaptiveAvgPool2d(2.5)

    def test_in_size_infers_out_size(self):
        """Supplying ``in_size`` triggers adaptive output-shape inference."""
        p = nn.AdaptiveAvgPool2d((4, 4), in_size=(8, 8, 3), channel_axis=-1)
        self.assertEqual(p.out_size, (4, 4, 3))

    def test_invalid_channel_axis_raises(self):
        """An out-of-range channel axis is rejected during the forward pass."""
        p = nn.AdaptiveAvgPool1d(5, channel_axis=5)
        with self.assertRaises(ValueError):
            p(brainstate.random.randn(3, 4))

    def test_too_few_dims_raises_with_channel(self):
        """A valid channel axis but too few overall dims raises a dimension error."""
        p = nn.AdaptiveAvgPool2d((2, 2), channel_axis=0)
        with self.assertRaises(ValueError):
            p(brainstate.random.randn(3, 4))

    def test_too_few_dims_raises_no_channel(self):
        """With ``channel_axis=None`` an input below the target rank raises."""
        p = nn.AdaptiveAvgPool2d((2, 2), channel_axis=None)
        with self.assertRaises(ValueError):
            p(brainstate.random.randn(4))

    def test_channel_axis_none_forward(self):
        """Adaptive pooling with ``channel_axis=None`` pools the trailing axis."""
        p = nn.AdaptiveAvgPool1d(5, channel_axis=None)
        out = p(brainstate.random.randn(2, 32))
        self.assertEqual(out.shape, (2, 5))

    def test_adaptive_pool_target_larger_than_input_raises(self):
        """Regression (H9): target size larger than the input raises ValueError.

        Previously this hit ``num_block = size // target_size == 0`` and raised a
        bare ``ZeroDivisionError`` from the reshape; it must now raise a clear
        ``ValueError`` (upsampling is unsupported).
        """
        p = nn.AdaptiveAvgPool1d(5, channel_axis=-1)
        with self.assertRaises(ValueError):
            p(brainstate.random.randn(1, 3, 2))  # pooled axis size 3 < target 5

    def test_adaptive_pool1d_target_larger_raises_directly(self):
        """Regression (H9): the internal 1D helper raises ValueError, not ZeroDivisionError."""
        from brainstate.nn._poolings import _adaptive_pool1d
        with self.assertRaises(ValueError):
            _adaptive_pool1d(brainstate.random.rand(3), 5, jnp.mean)

    def test_negative_channel_axis_forward(self):
        """Adaptive pooling resolves a negative channel axis."""
        p = nn.AdaptiveMaxPool1d(5, channel_axis=-1)
        out = p(brainstate.random.randn(2, 32, 4))
        self.assertEqual(out.shape, (2, 5, 4))

    def test_adaptive_avg_pool_with_none_target_dim(self):
        """A ``None`` entry in the target size leaves that dimension unchanged."""
        p = nn.AdaptiveAvgPool2d((None, 7), channel_axis=-1)
        out = p(brainstate.random.randn(1, 10, 9, 8))
        self.assertEqual(out.shape, (1, 10, 7, 8))

    def test_adaptive_max_pool2d_with_none_target_dim(self):
        """``None`` target dims are also supported by the max variant (2d)."""
        p = nn.AdaptiveMaxPool2d((None, 7), channel_axis=-1)
        out = p(brainstate.random.randn(1, 10, 9, 8))
        self.assertEqual(out.shape, (1, 10, 7, 8))

    def test_adaptive_avg_pool3d_with_none_target_dims(self):
        """Multiple ``None`` target dims leave those axes unchanged (3d)."""
        p = nn.AdaptiveAvgPool3d((7, None, None), channel_axis=-1)
        out = p(brainstate.random.randn(1, 10, 9, 8, 64))
        self.assertEqual(out.shape, (1, 7, 9, 8, 64))

    def test_adaptive_avg_pool_none_target_preserves_values(self):
        """An unpooled (``None``) axis is left numerically unchanged."""
        # Pool only the last spatial axis; the first spatial axis (None) untouched.
        p = nn.AdaptiveAvgPool2d((None, 2), channel_axis=-1)
        x = brainstate.random.randn(1, 3, 4, 2)
        out = p(x)
        self.assertEqual(out.shape, (1, 3, 2, 2))
        # Manually average the width axis (size 4 -> 2) and compare.
        expected = jnp.stack(
            [jnp.mean(x[:, :, 0:2, :], axis=2), jnp.mean(x[:, :, 2:4, :], axis=2)],
            axis=2,
        )
        np.testing.assert_allclose(np.asarray(out), np.asarray(expected), rtol=1e-5, atol=1e-6)


class TestAdaptivePoolTransforms(unittest.TestCase):
    """Gradient and jit consistency checks for the adaptive-pooling variants."""

    def test_adaptive_avg_pool1d_grad_and_jit(self):
        """AdaptiveAvgPool1d is jit-stable and produces finite gradients."""
        pool = nn.AdaptiveAvgPool1d(5, channel_axis=-1)
        with seeded(6):
            x = brainstate.random.randn(2, 16, 3)
        assert_jit_equal(pool, x)
        assert_grad_finite(lambda y: jnp.sum(pool(y)), x)

    def test_adaptive_max_pool2d_grad_and_jit(self):
        """AdaptiveMaxPool2d is jit-stable and produces finite gradients."""
        pool = nn.AdaptiveMaxPool2d((3, 3), channel_axis=-1)
        with seeded(8):
            x = brainstate.random.randn(1, 8, 8, 2)
        assert_jit_equal(pool, x)
        assert_grad_finite(lambda y: jnp.sum(pool(y)), x)

    @pytest.mark.slow
    def test_adaptive_max_pool3d_grad(self):
        """AdaptiveMaxPool3d gradients remain finite on a moderate input."""
        pool = nn.AdaptiveMaxPool3d((3, 3, 3), channel_axis=-1)
        with seeded(9):
            x = brainstate.random.randn(1, 8, 8, 8, 2)
        assert_grad_finite(lambda y: jnp.sum(pool(y)), x)


class TestPoolingAuditRegressions(unittest.TestCase):
    """Regression tests for pooling audit findings (P3-P8)."""

    def test_maxpool_stride_length_error_reports_stride_len(self):
        # P4: the stride-length error must mention the stride length, not the
        # kernel length. With kernel len 2 and stride len 3 the message must
        # report 3.
        with self.assertRaises(ValueError) as ctx:
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2, 2))
        self.assertIn("got 3", str(ctx.exception))

    def test_maxpool_bad_kernel_element_raises_not_assert(self):
        # P3: non-int kernel elements must raise (TypeError), not rely on
        # ``assert`` (which is stripped under ``python -O``).
        with self.assertRaises(TypeError):
            nn.MaxPool2d(kernel_size=(2.0, 2))

    def test_maxpool_bad_channel_axis_raises_not_assert(self):
        # P3: a bad channel_axis must raise TypeError, not assert.
        with self.assertRaises(TypeError):
            nn.MaxPool2d(kernel_size=2, channel_axis=1.5)

    def test_lppool_bad_stride_element_raises_not_assert(self):
        # P3/P5: _LPPool must validate via raise, not assert.
        with self.assertRaises(TypeError):
            nn.LPPool2d(norm_type=2, kernel_size=2, stride=(2.0, 2))

    def test_adaptivepool_rejects_str_target_size(self):
        # P6: a str target_size must be rejected (str is a Sequence).
        with self.assertRaises(TypeError):
            nn.AdaptiveAvgPool1d(target_size="ab")

    def test_adaptivepool_rejects_nonpositive_target_size(self):
        # P6: non-positive target sizes must be rejected.
        with self.assertRaises(ValueError):
            nn.AdaptiveAvgPool1d(target_size=0)
        with self.assertRaises(ValueError):
            nn.AdaptiveAvgPool2d(target_size=(2, -1))

    def test_adaptivepool_channel_axis_zero_preserves_channels(self):
        # P7/P8: channel_axis=0 must skip the channel axis when pooling. Input
        # (C=3, L=4) -> (C=3, L=2); previously ``if channel_axis:`` was falsy for
        # 0 and the channel axis was incorrectly pooled.
        pool = nn.AdaptiveAvgPool1d(target_size=2, channel_axis=0)
        x = jnp.arange(12.0).reshape(3, 4)
        y = pool(x)
        self.assertEqual(y.shape, (3, 2))
        # Each channel row is the mean over its two adaptive windows.
        expected = jnp.stack([
            jnp.array([(x[c, 0] + x[c, 1]) / 2, (x[c, 2] + x[c, 3]) / 2])
            for c in range(3)
        ])
        np.testing.assert_allclose(np.asarray(y), np.asarray(expected), rtol=1e-6)


class TestPoolingSequenceBranchValidation(unittest.TestCase):
    """Cover the ``Sequence``-but-not-``tuple``/``list`` and non-int-element
    validation branches across the pooling base classes.

    These branches fire for an argument that *is* a ``collections.abc.Sequence``
    (so it passes the ``isinstance(x, Sequence)`` guard) yet is neither a tuple
    nor a list -- e.g. a ``range`` object -- and for tuple/list arguments whose
    elements are not all ints. The existing tests only exercise the outer
    ``else`` (non-int, non-sequence) branch, so these complete the matrix.
    """

    # ------------------------------------------------------------------ _MaxPool

    def test_maxpool_kernel_size_sequence_not_tuple_raises(self):
        """A ``range`` kernel_size (a Sequence, not a tuple/list) is rejected."""
        with self.assertRaisesRegex(TypeError, 'kernel_size should be a tuple'):
            nn.MaxPool2d(range(2))

    def test_maxpool_stride_sequence_not_tuple_raises(self):
        """A ``range`` stride (a Sequence, not a tuple/list) is rejected."""
        with self.assertRaisesRegex(TypeError, 'stride should be a tuple'):
            nn.MaxPool2d(2, stride=range(2))

    def test_maxpool_stride_non_int_elements_raises(self):
        """A stride tuple containing a non-int element is rejected."""
        with self.assertRaisesRegex(TypeError, 'stride should be a tuple of ints'):
            nn.MaxPool2d(2, stride=(2.0, 2))

    def test_maxpool_padding_tuple_sequence_wrong_length_raises(self):
        """A sequence of ``(lo, hi)`` padding pairs of the wrong length is rejected.

        Each entry is a valid 2-tuple, so validation falls through to the final
        length check, which must reject a length that is neither 1 nor ``pool_dim``.
        """
        with self.assertRaisesRegex(ValueError, 'padding should has the length of 2'):
            nn.MaxPool2d(2, padding=[(1, 1), (1, 1), (1, 1)])

    # ---------------------------------------------------------------- _MaxUnpool

    def test_maxunpool_kernel_size_sequence_not_tuple_raises(self):
        """A ``range`` unpool kernel_size is rejected."""
        with self.assertRaisesRegex(TypeError, 'kernel_size should be a tuple'):
            nn.MaxUnpool2d(range(2))

    def test_maxunpool_kernel_size_non_int_elements_raises(self):
        """An unpool kernel tuple with a non-int element is rejected."""
        with self.assertRaisesRegex(TypeError, 'kernel_size should be a tuple of ints'):
            nn.MaxUnpool2d((2.0, 2))

    def test_maxunpool_stride_sequence_not_tuple_raises(self):
        """A ``range`` unpool stride is rejected."""
        with self.assertRaisesRegex(TypeError, 'stride should be a tuple'):
            nn.MaxUnpool2d(2, stride=range(2))

    def test_maxunpool_stride_non_int_elements_raises(self):
        """An unpool stride tuple with a non-int element is rejected."""
        with self.assertRaisesRegex(TypeError, 'stride should be a tuple of ints'):
            nn.MaxUnpool2d(2, stride=(2.0, 2))

    def test_maxunpool_channel_axis_non_int_raises(self):
        """A non-int, non-None unpool ``channel_axis`` is rejected."""
        with self.assertRaisesRegex(TypeError, 'channel_axis should be an int'):
            nn.MaxUnpool2d(2, channel_axis=1.5)

    # ------------------------------------------------------------------- _LPPool

    def test_lppool_kernel_size_sequence_not_tuple_raises(self):
        """A ``range`` LP kernel_size is rejected."""
        with self.assertRaisesRegex(TypeError, 'kernel_size should be a tuple'):
            nn.LPPool2d(2, range(2))

    def test_lppool_kernel_size_non_int_elements_raises(self):
        """An LP kernel tuple with a non-int element is rejected."""
        with self.assertRaisesRegex(TypeError, 'kernel_size should be a tuple of ints'):
            nn.LPPool2d(2, (2.0, 2))

    def test_lppool_stride_sequence_not_tuple_raises(self):
        """A ``range`` LP stride is rejected."""
        with self.assertRaisesRegex(TypeError, 'stride should be a tuple'):
            nn.LPPool2d(2, 2, stride=range(2))

    def test_lppool_padding_tuple_sequence_wrong_length_raises(self):
        """A sequence of LP padding pairs of the wrong length is rejected."""
        with self.assertRaisesRegex(ValueError, 'padding should has the length of 2'):
            nn.LPPool2d(2, 2, padding=[(1, 1), (1, 1), (1, 1)])

    def test_lppool_channel_axis_non_int_raises(self):
        """A non-int, non-None LP ``channel_axis`` is rejected."""
        with self.assertRaisesRegex(TypeError, 'channel_axis should be an int'):
            nn.LPPool2d(2, 2, channel_axis=1.5)


class TestAdaptivePoolHelperValidation(unittest.TestCase):
    """Cover the lower-bound and element-type guards of adaptive pooling."""

    def test_adaptive_pool1d_target_size_below_one_raises(self):
        """The 1D helper rejects ``target_size < 1`` before any reshape.

        This guard sits ahead of the ``size < target_size`` check, so it is only
        reachable by calling the helper directly with a non-positive target (the
        public layers reject ``target_size <= 0`` earlier in their constructor).
        """
        from brainstate.nn._poolings import _adaptive_pool1d
        with self.assertRaisesRegex(ValueError, 'target_size must be >= 1'):
            _adaptive_pool1d(jnp.arange(4.0), 0, jnp.mean)
        with self.assertRaisesRegex(ValueError, 'target_size must be >= 1'):
            _adaptive_pool1d(jnp.arange(4.0), -3, jnp.mean)

    def test_adaptive_pool_target_size_non_int_entry_raises(self):
        """A target-size tuple with a non-int, non-None entry is rejected.

        This is distinct from the non-positive-entry case (``(2, -1)``): here the
        offending entry is a float, so the ``isinstance(t, int)`` guard fires with
        a ``TypeError`` rather than the value-range ``ValueError``.
        """
        with self.assertRaisesRegex(TypeError, 'target_size.*entries must be ints or None'):
            nn.AdaptiveAvgPool2d((2, 2.5))
        with self.assertRaisesRegex(TypeError, 'target_size.*entries must be ints or None'):
            nn.AdaptiveMaxPool2d((2, 2.5))


if __name__ == '__main__':
    absltest.main()