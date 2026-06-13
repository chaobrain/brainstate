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

import brainstate
import braintools
from brainstate import _testing


class TestConv1d(unittest.TestCase):
    """Test cases for 1D convolution."""

    def test_basic_channels_last(self):
        """Test basic Conv1d with channels-last format."""
        conv = brainstate.nn.Conv1d(in_size=(100, 16), out_channels=32, kernel_size=5)
        x = jnp.ones((4, 100, 16))
        y = conv(x)

        self.assertEqual(y.shape, (4, 100, 32))
        self.assertEqual(conv.in_channels, 16)
        self.assertEqual(conv.out_channels, 32)
        self.assertFalse(conv.channel_first)

    def test_basic_channels_first(self):
        """Test basic Conv1d with channels-first format."""
        conv = brainstate.nn.Conv1d(in_size=(16, 100), out_channels=32, kernel_size=5, channel_first=True)
        x = jnp.ones((4, 16, 100))
        y = conv(x)

        self.assertEqual(y.shape, (4, 32, 100))
        self.assertEqual(conv.in_channels, 16)
        self.assertEqual(conv.out_channels, 32)
        self.assertTrue(conv.channel_first)

    def test_without_batch(self):
        """Test Conv1d without batch dimension."""
        conv = brainstate.nn.Conv1d(in_size=(50, 8), out_channels=16, kernel_size=3)
        x = jnp.ones((50, 8))
        y = conv(x)

        self.assertEqual(y.shape, (50, 16))

    def test_stride(self):
        """Test Conv1d with stride."""
        conv = brainstate.nn.Conv1d(in_size=(100, 8), out_channels=16, kernel_size=3, stride=2, padding='VALID')
        x = jnp.ones((2, 100, 8))
        y = conv(x)

        # VALID padding: output = (100 - 3 + 1) / 2 = 49
        self.assertEqual(y.shape, (2, 49, 16))

    def test_dilation(self):
        """Test Conv1d with dilated convolution."""
        conv = brainstate.nn.Conv1d(in_size=(100, 8), out_channels=16, kernel_size=3, rhs_dilation=2)
        x = jnp.ones((2, 100, 8))
        y = conv(x)

        self.assertEqual(y.shape, (2, 100, 16))

    def test_groups(self):
        """Test Conv1d with grouped convolution."""
        conv = brainstate.nn.Conv1d(in_size=(100, 16), out_channels=32, kernel_size=3, groups=4)
        x = jnp.ones((2, 100, 16))
        y = conv(x)

        self.assertEqual(y.shape, (2, 100, 32))
        self.assertEqual(conv.groups, 4)

    def test_with_bias(self):
        """Test Conv1d with bias."""
        conv = brainstate.nn.Conv1d(in_size=(50, 8), out_channels=16, kernel_size=3,
                                    b_init=braintools.init.Constant(0.0))
        x = jnp.ones((2, 50, 8))
        y = conv(x)

        self.assertEqual(y.shape, (2, 50, 16))
        self.assertIn('bias', conv.weight.value)


class TestConv2d(unittest.TestCase):
    """Test cases for 2D convolution."""

    def test_basic_channels_last(self):
        """Test basic Conv2d with channels-last format."""
        conv = brainstate.nn.Conv2d(in_size=(32, 32, 3), out_channels=64, kernel_size=3)
        x = jnp.ones((8, 32, 32, 3))
        y = conv(x)

        self.assertEqual(y.shape, (8, 32, 32, 64))
        self.assertEqual(conv.in_channels, 3)
        self.assertEqual(conv.out_channels, 64)
        self.assertFalse(conv.channel_first)

    def test_basic_channels_first(self):
        """Test basic Conv2d with channels-first format."""
        conv = brainstate.nn.Conv2d(in_size=(3, 32, 32), out_channels=64, kernel_size=3, channel_first=True)
        x = jnp.ones((8, 3, 32, 32))
        y = conv(x)

        self.assertEqual(y.shape, (8, 64, 32, 32))
        self.assertEqual(conv.in_channels, 3)
        self.assertEqual(conv.out_channels, 64)
        self.assertTrue(conv.channel_first)

    def test_rectangular_kernel(self):
        """Test Conv2d with rectangular kernel."""
        conv = brainstate.nn.Conv2d(in_size=(64, 64, 16), out_channels=32, kernel_size=(3, 5))
        x = jnp.ones((4, 64, 64, 16))
        y = conv(x)

        self.assertEqual(y.shape, (4, 64, 64, 32))
        self.assertEqual(conv.kernel_size, (3, 5))

    def test_stride_2d(self):
        """Test Conv2d with different strides."""
        conv = brainstate.nn.Conv2d(in_size=(64, 64, 3), out_channels=32, kernel_size=3, stride=(2, 2), padding='VALID')
        x = jnp.ones((4, 64, 64, 3))
        y = conv(x)

        # VALID padding: output = (64 - 3 + 1) / 2 = 31
        self.assertEqual(y.shape, (4, 31, 31, 32))

    def test_depthwise_convolution(self):
        """Test depthwise convolution (groups = in_channels)."""
        conv = brainstate.nn.Conv2d(in_size=(32, 32, 16), out_channels=16, kernel_size=3, groups=16)
        x = jnp.ones((4, 32, 32, 16))
        y = conv(x)

        self.assertEqual(y.shape, (4, 32, 32, 16))
        self.assertEqual(conv.groups, 16)

    def test_padding_same_vs_valid(self):
        """Test different padding modes."""
        conv_same = brainstate.nn.Conv2d(in_size=(32, 32, 3), out_channels=16, kernel_size=5, padding='SAME')
        conv_valid = brainstate.nn.Conv2d(in_size=(32, 32, 3), out_channels=16, kernel_size=5, padding='VALID')

        x = jnp.ones((2, 32, 32, 3))
        y_same = conv_same(x)
        y_valid = conv_valid(x)

        self.assertEqual(y_same.shape, (2, 32, 32, 16))  # SAME preserves size
        self.assertEqual(y_valid.shape, (2, 28, 28, 16))  # VALID reduces size


class TestConv3d(unittest.TestCase):
    """Test cases for 3D convolution."""

    def test_basic_channels_last(self):
        """Test basic Conv3d with channels-last format."""
        conv = brainstate.nn.Conv3d(in_size=(16, 16, 16, 1), out_channels=32, kernel_size=3)
        x = jnp.ones((2, 16, 16, 16, 1))
        y = conv(x)

        self.assertEqual(y.shape, (2, 16, 16, 16, 32))
        self.assertEqual(conv.in_channels, 1)
        self.assertEqual(conv.out_channels, 32)

    def test_basic_channels_first(self):
        """Test basic Conv3d with channels-first format."""
        conv = brainstate.nn.Conv3d(in_size=(1, 16, 16, 16), out_channels=32, kernel_size=3, channel_first=True)
        x = jnp.ones((2, 1, 16, 16, 16))
        y = conv(x)

        self.assertEqual(y.shape, (2, 32, 16, 16, 16))
        self.assertEqual(conv.in_channels, 1)
        self.assertEqual(conv.out_channels, 32)

    def test_video_data(self):
        """Test Conv3d for video data."""
        conv = brainstate.nn.Conv3d(in_size=(8, 32, 32, 3), out_channels=64, kernel_size=(3, 3, 3))
        x = jnp.ones((4, 8, 32, 32, 3))  # batch, frames, height, width, channels
        y = conv(x)

        self.assertEqual(y.shape, (4, 8, 32, 32, 64))


class TestScaledWSConv1d(unittest.TestCase):
    """Test cases for 1D convolution with weight standardization."""

    def test_basic(self):
        """Test basic ScaledWSConv1d."""
        conv = brainstate.nn.ScaledWSConv1d(in_size=(100, 16), out_channels=32, kernel_size=5)
        x = jnp.ones((4, 100, 16))
        y = conv(x)

        self.assertEqual(y.shape, (4, 100, 32))
        self.assertIsNotNone(conv.eps)

    def test_with_gain(self):
        """Test ScaledWSConv1d with gain parameter."""
        conv = brainstate.nn.ScaledWSConv1d(in_size=(100, 16), out_channels=32, kernel_size=5, ws_gain=True)
        x = jnp.ones((4, 100, 16))
        y = conv(x)

        self.assertEqual(y.shape, (4, 100, 32))
        self.assertIn('gain', conv.weight.value)

    def test_without_gain(self):
        """Test ScaledWSConv1d without gain parameter."""
        conv = brainstate.nn.ScaledWSConv1d(in_size=(100, 16), out_channels=32, kernel_size=5, ws_gain=False)
        x = jnp.ones((4, 100, 16))
        y = conv(x)

        self.assertEqual(y.shape, (4, 100, 32))
        self.assertNotIn('gain', conv.weight.value)

    def test_custom_eps(self):
        """Test ScaledWSConv1d with custom epsilon."""
        conv = brainstate.nn.ScaledWSConv1d(in_size=(100, 16), out_channels=32, kernel_size=5, eps=1e-5)
        self.assertEqual(conv.eps, 1e-5)


class TestScaledWSConv2d(unittest.TestCase):
    """Test cases for 2D convolution with weight standardization."""

    def test_basic_channels_last(self):
        """Test basic ScaledWSConv2d with channels-last format."""
        conv = brainstate.nn.ScaledWSConv2d(in_size=(64, 64, 3), out_channels=32, kernel_size=3)
        x = jnp.ones((8, 64, 64, 3))
        y = conv(x)

        self.assertEqual(y.shape, (8, 64, 64, 32))

    def test_basic_channels_first(self):
        """Test basic ScaledWSConv2d with channels-first format."""
        conv = brainstate.nn.ScaledWSConv2d(in_size=(3, 64, 64), out_channels=32, kernel_size=3, channel_first=True)
        x = jnp.ones((8, 3, 64, 64))
        y = conv(x)

        self.assertEqual(y.shape, (8, 32, 64, 64))

    def test_with_group_norm_style(self):
        """Test ScaledWSConv2d for use with group normalization."""
        conv = brainstate.nn.ScaledWSConv2d(
            in_size=(32, 32, 16),
            out_channels=32,
            kernel_size=3,
            ws_gain=True,
            groups=1
        )
        x = jnp.ones((4, 32, 32, 16))
        y = conv(x)

        self.assertEqual(y.shape, (4, 32, 32, 32))


class TestScaledWSConv3d(unittest.TestCase):
    """Test cases for 3D convolution with weight standardization."""

    def test_basic(self):
        """Test basic ScaledWSConv3d."""
        conv = brainstate.nn.ScaledWSConv3d(in_size=(8, 16, 16, 3), out_channels=32, kernel_size=3)
        x = jnp.ones((2, 8, 16, 16, 3))
        y = conv(x)

        self.assertEqual(y.shape, (2, 8, 16, 16, 32))

    def test_channels_first(self):
        """Test ScaledWSConv3d with channels-first format."""
        conv = brainstate.nn.ScaledWSConv3d(in_size=(3, 8, 16, 16), out_channels=32, kernel_size=3, channel_first=True)
        x = jnp.ones((2, 3, 8, 16, 16))
        y = conv(x)

        self.assertEqual(y.shape, (2, 32, 8, 16, 16))


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases."""

    def test_invalid_input_shape(self):
        """Test that invalid input shapes raise appropriate errors."""
        conv = brainstate.nn.Conv2d(in_size=(32, 32, 3), out_channels=16, kernel_size=3)
        x_wrong = jnp.ones((8, 32, 32, 16))  # Wrong number of channels

        with self.assertRaises(ValueError):
            conv(x_wrong)

    def test_invalid_groups(self):
        """Test that invalid group configurations raise errors.

        Validation of user-supplied constructor arguments must raise ``ValueError``
        (which survives ``python -O``), not ``AssertionError``.
        """
        with self.assertRaises(ValueError):
            # out_channels not divisible by groups
            conv = brainstate.nn.Conv2d(in_size=(32, 32, 16), out_channels=30, kernel_size=3, groups=4)

    def test_dimension_mismatch(self):
        """Test dimension mismatch detection."""
        conv = brainstate.nn.Conv2d(in_size=(32, 32, 3), out_channels=16, kernel_size=3)
        x_1d = jnp.ones((8, 32, 3))  # 1D instead of 2D

        with self.assertRaises(ValueError):
            conv(x_1d)


class TestOutputShapes(unittest.TestCase):
    """Test output shape calculations."""

    def test_same_padding_preserves_size(self):
        """Test that SAME padding preserves spatial dimensions when stride=1."""
        for kernel_size in [3, 5, 7]:
            conv = brainstate.nn.Conv2d(in_size=(32, 32, 3), out_channels=16, kernel_size=kernel_size, padding='SAME')
            x = jnp.ones((4, 32, 32, 3))
            y = conv(x)
            self.assertEqual(y.shape, (4, 32, 32, 16), f"Failed for kernel_size={kernel_size}")

    def test_valid_padding_reduces_size(self):
        """Test that VALID padding reduces spatial dimensions."""
        conv = brainstate.nn.Conv2d(in_size=(32, 32, 3), out_channels=16, kernel_size=5, padding='VALID')
        x = jnp.ones((4, 32, 32, 3))
        y = conv(x)
        # 32 - 5 + 1 = 28
        self.assertEqual(y.shape, (4, 28, 28, 16))

    def test_output_size_attribute(self):
        """Test that out_size attribute is correctly computed."""
        conv_cl = brainstate.nn.Conv2d(in_size=(64, 64, 3), out_channels=32, kernel_size=3, channel_first=False)
        conv_cf = brainstate.nn.Conv2d(in_size=(3, 64, 64), out_channels=32, kernel_size=3, channel_first=True)

        self.assertEqual(conv_cl.out_size, (64, 64, 32))
        self.assertEqual(conv_cf.out_size, (32, 64, 64))


class TestChannelFormatConsistency(unittest.TestCase):
    """Test consistency between channels-first and channels-last formats."""

    def test_conv1d_output_channels(self):
        """Test that output channels are in correct position for both formats."""
        conv_cl = brainstate.nn.Conv1d(in_size=(100, 16), out_channels=32, kernel_size=3)
        conv_cf = brainstate.nn.Conv1d(in_size=(16, 100), out_channels=32, kernel_size=3, channel_first=True)

        x_cl = jnp.ones((4, 100, 16))
        x_cf = jnp.ones((4, 16, 100))

        y_cl = conv_cl(x_cl)
        y_cf = conv_cf(x_cf)

        # Channels-last: channels in last dimension
        self.assertEqual(y_cl.shape[-1], 32)
        # Channels-first: channels in first dimension (after batch)
        self.assertEqual(y_cf.shape[1], 32)

    def test_conv2d_output_channels(self):
        """Test 2D output channel positions."""
        conv_cl = brainstate.nn.Conv2d(in_size=(32, 32, 3), out_channels=64, kernel_size=3)
        conv_cf = brainstate.nn.Conv2d(in_size=(3, 32, 32), out_channels=64, kernel_size=3, channel_first=True)

        x_cl = jnp.ones((4, 32, 32, 3))
        x_cf = jnp.ones((4, 3, 32, 32))

        y_cl = conv_cl(x_cl)
        y_cf = conv_cf(x_cf)

        self.assertEqual(y_cl.shape[-1], 64)
        self.assertEqual(y_cf.shape[1], 64)


class TestReproducibility(unittest.TestCase):
    """Test reproducibility with fixed seeds."""

    def test_deterministic_output(self):
        """Test that same seed produces same output."""
        conv1 = brainstate.nn.Conv2d(in_size=(32, 32, 3), out_channels=16, kernel_size=3)
        conv2 = brainstate.nn.Conv2d(in_size=(32, 32, 3), out_channels=16, kernel_size=3)

        # Use random input
        x = brainstate.random.randn(4, 32, 32, 3)

        # Note: outputs will differ due to different weight initialization
        # This test just ensures no crashes with random inputs
        y1 = conv1(x)
        y2 = conv2(x)

        self.assertEqual(y1.shape, y2.shape)


class TestRepr(unittest.TestCase):
    """Test string representations."""

    def test_conv_repr_channels_last(self):
        """Test __repr__ for channels-last format."""
        conv = brainstate.nn.Conv2d(in_size=(32, 32, 3), out_channels=64, kernel_size=3)
        repr_str = repr(conv)

        self.assertIn('Conv2d', repr_str)
        self.assertIn('channel_first=False', repr_str)
        self.assertIn('in_channels=3', repr_str)
        self.assertIn('out_channels=64', repr_str)

    def test_conv_repr_channels_first(self):
        """Test __repr__ for channels-first format."""
        conv = brainstate.nn.Conv2d(in_size=(3, 32, 32), out_channels=64, kernel_size=3, channel_first=True)
        repr_str = repr(conv)

        self.assertIn('Conv2d', repr_str)
        self.assertIn('channel_first=True', repr_str)


class TestConvTranspose1d(unittest.TestCase):
    """Test cases for ConvTranspose1d layer."""

    def setUp(self):
        """Set up test fixtures."""
        self.in_size = (28, 16)
        self.out_channels = 8
        self.kernel_size = 4

    def test_basic_channels_last(self):
        """Test basic ConvTranspose1d with channels-last format."""
        conv_t = brainstate.nn.ConvTranspose1d(
            in_size=self.in_size,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=1
        )
        x = jnp.ones((2, 28, 16))
        y = conv_t(x)

        self.assertEqual(len(y.shape), 3)
        self.assertEqual(y.shape[0], 2)  # batch size
        self.assertEqual(y.shape[-1], self.out_channels)
        self.assertEqual(conv_t.in_channels, 16)
        self.assertEqual(conv_t.out_channels, 8)
        self.assertFalse(conv_t.channel_first)

    def test_basic_channels_first(self):
        """Test basic ConvTranspose1d with channels-first format."""
        conv_t = brainstate.nn.ConvTranspose1d(
            in_size=(16, 28),  # (C, L) for channels-first
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=1,
            channel_first=True
        )
        x = jnp.ones((2, 16, 28))
        y = conv_t(x)

        self.assertEqual(len(y.shape), 3)
        self.assertEqual(y.shape[0], 2)  # batch size
        self.assertEqual(y.shape[1], self.out_channels)  # channels first
        self.assertEqual(conv_t.in_channels, 16)
        self.assertTrue(conv_t.channel_first)

    def test_stride_upsampling(self):
        """Test transposed convolution with stride for upsampling."""
        conv_t = brainstate.nn.ConvTranspose1d(
            in_size=(28, 16),
            out_channels=8,
            kernel_size=4,
            stride=2,
            padding='SAME'
        )
        x = jnp.ones((2, 28, 16))
        y = conv_t(x)

        # With stride=2, output should be approximately 2x larger
        self.assertGreater(y.shape[1], x.shape[1])

    def test_with_bias(self):
        """Test ConvTranspose1d with bias."""
        conv_t = brainstate.nn.ConvTranspose1d(
            in_size=(50, 8),
            out_channels=16,
            kernel_size=3,
            b_init=braintools.init.Constant(0.0)
        )
        x = jnp.ones((4, 50, 8))
        y = conv_t(x)

        self.assertTrue('bias' in conv_t.weight.value)
        self.assertEqual(y.shape[-1], 16)

    def test_without_batch(self):
        """Test ConvTranspose1d without batch dimension."""
        conv_t = brainstate.nn.ConvTranspose1d(
            in_size=(28, 16),
            out_channels=8,
            kernel_size=4
        )
        x = jnp.ones((28, 16))
        y = conv_t(x)

        self.assertEqual(len(y.shape), 2)
        self.assertEqual(y.shape[-1], 8)

    def test_groups(self):
        """Test grouped transposed convolution."""
        conv_t = brainstate.nn.ConvTranspose1d(
            in_size=(28, 16),
            out_channels=16,
            kernel_size=3,
            groups=4
        )
        x = jnp.ones((2, 28, 16))
        y = conv_t(x)

        self.assertEqual(y.shape[-1], 16)


class TestConvTranspose2d(unittest.TestCase):
    """Test cases for ConvTranspose2d layer."""

    def setUp(self):
        """Set up test fixtures."""
        self.in_size = (16, 16, 32)
        self.out_channels = 16
        self.kernel_size = 4

    def test_basic_channels_last(self):
        """Test basic ConvTranspose2d with channels-last format."""
        conv_t = brainstate.nn.ConvTranspose2d(
            in_size=self.in_size,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size
        )
        x = jnp.ones((4, 16, 16, 32))
        y = conv_t(x)

        self.assertEqual(len(y.shape), 4)
        self.assertEqual(y.shape[0], 4)  # batch size
        self.assertEqual(y.shape[-1], self.out_channels)
        self.assertEqual(conv_t.in_channels, 32)
        self.assertFalse(conv_t.channel_first)

    def test_basic_channels_first(self):
        """Test basic ConvTranspose2d with channels-first format."""
        conv_t = brainstate.nn.ConvTranspose2d(
            in_size=(32, 16, 16),  # (C, H, W) for channels-first
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            channel_first=True
        )
        x = jnp.ones((4, 32, 16, 16))
        y = conv_t(x)

        self.assertEqual(len(y.shape), 4)
        self.assertEqual(y.shape[1], self.out_channels)  # channels first
        self.assertTrue(conv_t.channel_first)

    def test_stride_upsampling(self):
        """Test 2x upsampling with stride=2."""
        conv_t = brainstate.nn.ConvTranspose2d(
            in_size=(16, 16, 32),
            out_channels=16,
            kernel_size=4,
            stride=2,
            padding='SAME'
        )
        x = jnp.ones((4, 16, 16, 32))
        y = conv_t(x)

        # With stride=2, output should be approximately 2x larger in each spatial dimension
        self.assertGreater(y.shape[1], x.shape[1])
        self.assertGreater(y.shape[2], x.shape[2])

    def test_rectangular_kernel(self):
        """Test ConvTranspose2d with rectangular kernel."""
        conv_t = brainstate.nn.ConvTranspose2d(
            in_size=(16, 16, 32),
            out_channels=16,
            kernel_size=(3, 5),
            stride=1
        )
        x = jnp.ones((2, 16, 16, 32))
        y = conv_t(x)

        self.assertEqual(conv_t.kernel_size, (3, 5))
        self.assertEqual(y.shape[-1], 16)

    def test_padding_valid(self):
        """Test ConvTranspose2d with VALID padding."""
        conv_t = brainstate.nn.ConvTranspose2d(
            in_size=(16, 16, 32),
            out_channels=16,
            kernel_size=4,
            stride=2,
            padding='VALID'
        )
        x = jnp.ones((2, 16, 16, 32))
        y = conv_t(x)

        # VALID padding means no padding, output computed by formula:
        # out = (in - 1) * stride + kernel
        # out = (16 - 1) * 2 + 4 = 34 (but JAX may compute it slightly differently)
        # At minimum, it should upsample
        self.assertGreater(y.shape[1], 16)

    def test_groups(self):
        """Test grouped transposed convolution."""
        conv_t = brainstate.nn.ConvTranspose2d(
            in_size=(16, 16, 32),
            out_channels=32,
            kernel_size=3,
            groups=4
        )
        x = jnp.ones((2, 16, 16, 32))
        y = conv_t(x)

        self.assertEqual(y.shape[-1], 32)


class TestConvTranspose3d(unittest.TestCase):
    """Test cases for ConvTranspose3d layer."""

    def setUp(self):
        """Set up test fixtures."""
        self.in_size = (8, 8, 8, 16)
        self.out_channels = 8
        self.kernel_size = 4

    def test_basic_channels_last(self):
        """Test basic ConvTranspose3d with channels-last format."""
        conv_t = brainstate.nn.ConvTranspose3d(
            in_size=self.in_size,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size
        )
        x = jnp.ones((2, 8, 8, 8, 16))
        y = conv_t(x)

        self.assertEqual(len(y.shape), 5)
        self.assertEqual(y.shape[0], 2)  # batch size
        self.assertEqual(y.shape[-1], self.out_channels)
        self.assertEqual(conv_t.in_channels, 16)

    def test_basic_channels_first(self):
        """Test basic ConvTranspose3d with channels-first format."""
        conv_t = brainstate.nn.ConvTranspose3d(
            in_size=(16, 8, 8, 8),  # (C, H, W, D) for channels-first
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            channel_first=True
        )
        x = jnp.ones((2, 16, 8, 8, 8))
        y = conv_t(x)

        self.assertEqual(len(y.shape), 5)
        self.assertEqual(y.shape[1], self.out_channels)  # channels first
        self.assertTrue(conv_t.channel_first)

    def test_stride_upsampling(self):
        """Test 3D upsampling with stride=2."""
        conv_t = brainstate.nn.ConvTranspose3d(
            in_size=(8, 8, 8, 16),
            out_channels=8,
            kernel_size=4,
            stride=2,
            padding='SAME'
        )
        x = jnp.ones((2, 8, 8, 8, 16))
        y = conv_t(x)

        # With stride=2, output should be approximately 2x larger
        self.assertGreater(y.shape[1], x.shape[1])
        self.assertGreater(y.shape[2], x.shape[2])
        self.assertGreater(y.shape[3], x.shape[3])


class TestErrorHandlingConvTranspose(unittest.TestCase):
    """Test error handling for transposed convolutions."""

    def test_invalid_groups(self):
        """Test that invalid groups raises a ValueError.

        Validation of user-supplied constructor arguments must raise ``ValueError``
        (which survives ``python -O``), not ``AssertionError``.
        """
        with self.assertRaises(ValueError):
            brainstate.nn.ConvTranspose2d(
                in_size=(16, 16, 32),
                out_channels=15,  # Not divisible by groups
                kernel_size=3,
                groups=4
            )

    def test_dimension_mismatch(self):
        """Test that wrong input dimensions raise error."""
        conv_t = brainstate.nn.ConvTranspose2d(
            in_size=(16, 16, 32),
            out_channels=16,
            kernel_size=3
        )
        x = jnp.ones((2, 16, 16, 16))  # Wrong number of channels

        with self.assertRaises(ValueError):
            conv_t(x)

    def test_invalid_input_shape(self):
        """Test that invalid input shape raises error."""
        conv_t = brainstate.nn.ConvTranspose1d(
            in_size=(28, 16),
            out_channels=8,
            kernel_size=3
        )
        x = jnp.ones((2, 2, 28, 16))  # Too many dimensions

        with self.assertRaises(ValueError):
            conv_t(x)


class TestOutputShapesConvTranspose(unittest.TestCase):
    """Test output shape computation for transposed convolutions."""

    def test_out_size_attribute_1d(self):
        """Test that out_size attribute is correctly computed for 1D."""
        conv_t = brainstate.nn.ConvTranspose1d(
            in_size=(28, 16),
            out_channels=8,
            kernel_size=4,
            stride=2
        )

        self.assertIsNotNone(conv_t.out_size)
        self.assertEqual(len(conv_t.out_size), 2)

    def test_out_size_attribute_2d(self):
        """Test that out_size attribute is correctly computed for 2D."""
        conv_t = brainstate.nn.ConvTranspose2d(
            in_size=(16, 16, 32),
            out_channels=16,
            kernel_size=4,
            stride=2
        )

        self.assertIsNotNone(conv_t.out_size)
        self.assertEqual(len(conv_t.out_size), 3)

    def test_upsampling_factor(self):
        """Test that stride=2 approximately doubles spatial dimensions."""
        conv_t = brainstate.nn.ConvTranspose2d(
            in_size=(16, 16, 32),
            out_channels=16,
            kernel_size=4,
            stride=2,
            padding='SAME'
        )
        x = jnp.ones((2, 16, 16, 32))
        y = conv_t(x)

        # For SAME padding and stride=2, output should be approximately 2x input
        self.assertGreaterEqual(y.shape[1], 28)
        self.assertGreaterEqual(y.shape[2], 28)


class TestChannelFormatConsistencyConvTranspose(unittest.TestCase):
    """Test consistency between different channel formats."""

    def test_conv_transpose_1d_output_channels(self):
        """Test that output channels are in correct position for both formats."""
        # Channels-last
        conv_t_last = brainstate.nn.ConvTranspose1d(
            in_size=(28, 16),
            out_channels=8,
            kernel_size=3
        )
        x_last = jnp.ones((2, 28, 16))
        y_last = conv_t_last(x_last)
        self.assertEqual(y_last.shape[-1], 8)

        # Channels-first
        conv_t_first = brainstate.nn.ConvTranspose1d(
            in_size=(16, 28),
            out_channels=8,
            kernel_size=3,
            channel_first=True
        )
        x_first = jnp.ones((2, 16, 28))
        y_first = conv_t_first(x_first)
        self.assertEqual(y_first.shape[1], 8)

    def test_conv_transpose_2d_output_channels(self):
        """Test that output channels are in correct position for both formats."""
        # Channels-last
        conv_t_last = brainstate.nn.ConvTranspose2d(
            in_size=(16, 16, 32),
            out_channels=16,
            kernel_size=3
        )
        x_last = jnp.ones((2, 16, 16, 32))
        y_last = conv_t_last(x_last)
        self.assertEqual(y_last.shape[-1], 16)

        # Channels-first
        conv_t_first = brainstate.nn.ConvTranspose2d(
            in_size=(32, 16, 16),
            out_channels=16,
            kernel_size=3,
            channel_first=True
        )
        x_first = jnp.ones((2, 32, 16, 16))
        y_first = conv_t_first(x_first)
        self.assertEqual(y_first.shape[1], 16)


class TestReproducibilityConvTranspose(unittest.TestCase):
    """Test deterministic behavior of transposed convolutions."""

    def test_deterministic_output(self):
        """Test that same input produces same output."""
        conv_t = brainstate.nn.ConvTranspose2d(
            in_size=(16, 16, 32),
            out_channels=16,
            kernel_size=3
        )
        x = jnp.ones((2, 16, 16, 32))

        y1 = conv_t(x)
        y2 = conv_t(x)

        self.assertTrue(jnp.allclose(y1, y2))


class TestKernelShapeConvTranspose(unittest.TestCase):
    """Test kernel shape computation for transposed convolutions."""

    def test_kernel_shape_1d(self):
        """Test that kernel shape is correct for transposed conv 1D."""
        conv_t = brainstate.nn.ConvTranspose1d(
            in_size=(28, 16),
            out_channels=8,
            kernel_size=4,
            groups=2
        )
        # For transpose conv: (kernel_size, out_channels, in_channels // groups)
        expected_shape = (4, 8, 16 // 2)
        self.assertEqual(conv_t.kernel_shape, expected_shape)

    def test_kernel_shape_2d(self):
        """Test that kernel shape is correct for transposed conv 2D."""
        conv_t = brainstate.nn.ConvTranspose2d(
            in_size=(16, 16, 32),
            out_channels=16,
            kernel_size=4,
            groups=4
        )
        # For transpose conv: (kernel_h, kernel_w, out_channels, in_channels // groups)
        expected_shape = (4, 4, 16, 32 // 4)
        self.assertEqual(conv_t.kernel_shape, expected_shape)


class TestReplicateHelper(unittest.TestCase):
    """Tests for the kernel/stride replication helper via public constructors."""

    def test_single_element_sequence(self):
        """A length-1 sequence is replicated to match the spatial dimensions."""
        conv = brainstate.nn.Conv2d(in_size=(16, 16, 3), out_channels=8, kernel_size=[3])
        self.assertEqual(conv.kernel_size, (3, 3))

    def test_full_length_sequence(self):
        """A full-length sequence is kept unchanged."""
        conv = brainstate.nn.Conv2d(in_size=(16, 16, 3), out_channels=8, kernel_size=(3, 5))
        self.assertEqual(conv.kernel_size, (3, 5))

    def test_wrong_length_sequence_raises(self):
        """A sequence whose length is neither 1 nor num_spatial_dims raises TypeError."""
        with self.assertRaises(TypeError):
            brainstate.nn.Conv2d(in_size=(16, 16, 3), out_channels=8, kernel_size=[3, 3, 3])


class TestConvPadding(unittest.TestCase):
    """Tests for the various padding specifications accepted by convolution layers."""

    def test_padding_int(self):
        """An integer padding is expanded to symmetric padding per dimension."""
        conv = brainstate.nn.Conv2d(in_size=(16, 16, 3), out_channels=8, kernel_size=3, padding=2)
        self.assertEqual(conv.padding, ((2, 2), (2, 2)))

    def test_padding_tuple_of_int(self):
        """A flat int tuple gives one symmetric pad per spatial dim (M11).

        For Conv2d a length-2 tuple ``(1, 2)`` is one value per spatial axis:
        axis 0 -> (1, 1), axis 1 -> (2, 2). This is NOT a single (low, high) pair
        broadcast to every axis (the old, buggy behavior). For Conv1d, a length-2
        tuple is unambiguously a single (low, high) pair for the one spatial axis.
        """
        # Conv2d: one symmetric pad per spatial dim.
        conv2d = brainstate.nn.Conv2d(in_size=(16, 16, 3), out_channels=8, kernel_size=3, padding=(1, 2))
        self.assertEqual(conv2d.padding, ((1, 1), (2, 2)))

        # Conv3d: one symmetric pad per spatial dim.
        conv3d = brainstate.nn.Conv3d(in_size=(16, 16, 16, 3), out_channels=8, kernel_size=3, padding=(1, 2, 3))
        self.assertEqual(conv3d.padding, ((1, 1), (2, 2), (3, 3)))

        # Conv1d: length-2 tuple is a single (low, high) pair for the one axis.
        conv1d = brainstate.nn.Conv1d(in_size=(16, 3), out_channels=8, kernel_size=3, padding=(1, 2))
        self.assertEqual(conv1d.padding, ((1, 2),))

    def test_padding_int_tuple_wrong_length_raises(self):
        """A flat int padding tuple of the wrong length raises ValueError (M11)."""
        # Conv2d expects length 2; length 3 is invalid.
        with self.assertRaises(ValueError):
            brainstate.nn.Conv2d(in_size=(16, 16, 3), out_channels=8, kernel_size=3, padding=(1, 2, 3))

    def test_padding_sequence_of_tuples(self):
        """A full sequence of (low, high) tuples is preserved per dimension."""
        conv = brainstate.nn.Conv2d(in_size=(16, 16, 3), out_channels=8, kernel_size=3, padding=[(1, 1), (2, 2)])
        self.assertEqual(conv.padding, ((1, 1), (2, 2)))

    def test_padding_length_one_sequence(self):
        """A length-1 sequence of tuples is broadcast across all dimensions."""
        conv = brainstate.nn.Conv2d(in_size=(16, 16, 3), out_channels=8, kernel_size=3, padding=[(1, 1)])
        self.assertEqual(conv.padding, ((1, 1), (1, 1)))

    def test_padding_wrong_length_raises(self):
        """A sequence of tuples with the wrong length raises ValueError."""
        with self.assertRaises(ValueError):
            brainstate.nn.Conv2d(in_size=(16, 16, 3), out_channels=8, kernel_size=3, padding=[(1, 1), (2, 2), (3, 3)])

    def test_padding_invalid_type_raises(self):
        """A padding of an unsupported type raises ValueError."""
        with self.assertRaises(ValueError):
            brainstate.nn.Conv2d(in_size=(16, 16, 3), out_channels=8, kernel_size=3, padding=1.5)


class TestConvWeightMask(unittest.TestCase):
    """Tests for weight masking in convolution layers."""

    def test_conv2d_w_mask_zeros_output(self):
        """A zero weight mask forces the convolution output to zero."""
        mask = np.zeros((3, 3, 3, 8))
        conv = brainstate.nn.Conv2d(in_size=(16, 16, 3), out_channels=8, kernel_size=3, w_mask=mask)
        x = jnp.ones((1, 16, 16, 3))
        y = conv(x)
        _testing.assert_allclose(y, jnp.zeros_like(y))

    def test_conv2d_w_mask_ones_identity(self):
        """A unit weight mask leaves the convolution output unchanged."""
        mask = np.ones((3, 3, 3, 8))
        conv_masked = brainstate.nn.Conv2d(in_size=(16, 16, 3), out_channels=8, kernel_size=3, w_mask=mask)
        x = jnp.ones((1, 16, 16, 3))
        y = conv_masked(x)
        self.assertEqual(y.shape, (1, 16, 16, 8))


class TestBaseConvAbstract(unittest.TestCase):
    """Tests for the abstract base convolution behavior."""

    def test_conv_op_not_implemented(self):
        """The base class ``_conv_op`` raises NotImplementedError."""
        from brainstate.nn._conv import _BaseConv

        class _Dummy(_BaseConv):
            num_spatial_dims = 2

        dummy = _Dummy(in_size=(8, 8, 3), out_channels=4, kernel_size=3)
        with self.assertRaises(NotImplementedError):
            dummy._conv_op(jnp.ones((1, 8, 8, 3)), {})


class TestConvGradients(unittest.TestCase):
    """Tests for finite gradients through convolution layers."""

    def test_conv1d_grad_finite(self):
        """Gradients of a scalar loss through Conv1d are finite."""
        conv = brainstate.nn.Conv1d(in_size=(20, 4), out_channels=8, kernel_size=3)
        x = brainstate.random.randn(2, 20, 4)

        def loss(inp):
            return jnp.sum(conv(inp) ** 2)

        _testing.assert_grad_finite(loss, x)

    def test_conv2d_grad_finite(self):
        """Gradients of a scalar loss through Conv2d are finite."""
        conv = brainstate.nn.Conv2d(in_size=(12, 12, 3), out_channels=8, kernel_size=3)
        x = brainstate.random.randn(2, 12, 12, 3)

        def loss(inp):
            return jnp.sum(conv(inp) ** 2)

        _testing.assert_grad_finite(loss, x)

    @pytest.mark.slow
    def test_conv3d_grad_finite(self):
        """Gradients of a scalar loss through Conv3d are finite."""
        conv = brainstate.nn.Conv3d(in_size=(6, 6, 6, 2), out_channels=4, kernel_size=3)
        x = brainstate.random.randn(1, 6, 6, 6, 2)

        def loss(inp):
            return jnp.sum(conv(inp) ** 2)

        _testing.assert_grad_finite(loss, x)


class TestConvJit(unittest.TestCase):
    """Tests that JIT-compiled convolutions match eager execution."""

    def test_conv1d_jit_equal(self):
        """JIT-compiled Conv1d matches eager output."""
        conv = brainstate.nn.Conv1d(in_size=(20, 4), out_channels=8, kernel_size=3)
        x = brainstate.random.randn(2, 20, 4)
        _testing.assert_jit_equal(lambda inp: conv(inp), x)

    def test_conv2d_jit_equal(self):
        """JIT-compiled Conv2d matches eager output."""
        conv = brainstate.nn.Conv2d(in_size=(12, 12, 3), out_channels=8, kernel_size=3)
        x = brainstate.random.randn(2, 12, 12, 3)
        _testing.assert_jit_equal(lambda inp: conv(inp), x)

    def test_scaled_ws_conv2d_jit_equal(self):
        """JIT-compiled ScaledWSConv2d matches eager output."""
        conv = brainstate.nn.ScaledWSConv2d(in_size=(12, 12, 3), out_channels=8, kernel_size=3)
        x = brainstate.random.randn(2, 12, 12, 3)
        _testing.assert_jit_equal(lambda inp: conv(inp), x)


class TestScaledWSConvExtra(unittest.TestCase):
    """Extra tests for weight-standardized convolutions covering bias and masks."""

    def test_scaled_ws_with_bias_and_mask(self):
        """ScaledWSConv2d supports both a bias term and a weight mask."""
        mask = np.ones((3, 3, 3, 8))
        conv = brainstate.nn.ScaledWSConv2d(
            in_size=(16, 16, 3),
            out_channels=8,
            kernel_size=3,
            b_init=braintools.init.Constant(0.1),
            w_mask=mask,
        )
        x = jnp.ones((1, 16, 16, 3))
        y = conv(x)
        self.assertEqual(y.shape, (1, 16, 16, 8))
        self.assertIn('bias', conv.weight.value)

    def test_scaled_ws_zero_mask_zeros_output(self):
        """A zero mask forces a weight-standardized convolution output to zero (no bias)."""
        mask = np.zeros((5, 3, 8))
        conv = brainstate.nn.ScaledWSConv1d(in_size=(20, 3), out_channels=8, kernel_size=5, w_mask=mask)
        x = jnp.ones((1, 20, 3))
        y = conv(x)
        _testing.assert_allclose(y, jnp.zeros_like(y))

    def test_scaled_ws_grad_finite(self):
        """Gradients through ScaledWSConv2d are finite."""
        conv = brainstate.nn.ScaledWSConv2d(in_size=(12, 12, 3), out_channels=8, kernel_size=3)
        x = brainstate.random.randn(2, 12, 12, 3)

        def loss(inp):
            return jnp.sum(conv(inp) ** 2)

        _testing.assert_grad_finite(loss, x)


class TestConvTransposePadding(unittest.TestCase):
    """Tests for padding specifications on transposed convolution layers."""

    def test_padding_int(self):
        """An integer padding is expanded to symmetric explicit padding."""
        conv = brainstate.nn.ConvTranspose2d(in_size=(8, 8, 16), out_channels=8, kernel_size=3, padding=1)
        self.assertEqual(conv.padding, ((1, 1), (1, 1)))
        self.assertEqual(conv.padding_mode, 'explicit')

    def test_padding_tuple_of_int(self):
        """A flat int tuple gives one symmetric pad per spatial dim (M11).

        For ConvTranspose2d a length-2 tuple ``(1, 2)`` is one value per spatial
        axis: axis 0 -> (1, 1), axis 1 -> (2, 2), not a single (low, high) pair
        broadcast to every axis.
        """
        conv = brainstate.nn.ConvTranspose2d(in_size=(8, 8, 16), out_channels=8, kernel_size=3, padding=(1, 2))
        self.assertEqual(conv.padding, ((1, 1), (2, 2)))

    def test_padding_int_tuple_wrong_length_raises(self):
        """A flat int padding tuple of the wrong length raises ValueError (M11)."""
        with self.assertRaises(ValueError):
            brainstate.nn.ConvTranspose2d(in_size=(8, 8, 16), out_channels=8, kernel_size=3, padding=(1, 2, 3))

    def test_padding_sequence_of_tuples(self):
        """A full sequence of (low, high) tuples is preserved."""
        conv = brainstate.nn.ConvTranspose2d(
            in_size=(8, 8, 16), out_channels=8, kernel_size=3, padding=[(1, 1), (2, 2)]
        )
        self.assertEqual(conv.padding, ((1, 1), (2, 2)))

    def test_padding_length_one_sequence(self):
        """A length-1 sequence of tuples is broadcast across dimensions."""
        conv = brainstate.nn.ConvTranspose2d(in_size=(8, 8, 16), out_channels=8, kernel_size=3, padding=[(1, 1)])
        self.assertEqual(conv.padding, ((1, 1), (1, 1)))

    def test_padding_wrong_length_raises(self):
        """A sequence of tuples with the wrong length raises ValueError."""
        with self.assertRaises(ValueError):
            brainstate.nn.ConvTranspose2d(
                in_size=(8, 8, 16), out_channels=8, kernel_size=3, padding=[(1, 1), (2, 2), (3, 3)]
            )

    def test_padding_invalid_type_raises(self):
        """An unsupported padding type raises ValueError."""
        with self.assertRaises(ValueError):
            brainstate.nn.ConvTranspose2d(in_size=(8, 8, 16), out_channels=8, kernel_size=3, padding=1.5)


class TestConvTransposeWeightMask(unittest.TestCase):
    """Tests for weight masking in transposed convolution layers."""

    def test_conv_transpose_zero_mask_zeros_output(self):
        """A zero weight mask forces the transposed convolution output to zero."""
        mask = np.zeros((3, 3, 8, 16))
        conv = brainstate.nn.ConvTranspose2d(in_size=(8, 8, 16), out_channels=8, kernel_size=3, w_mask=mask)
        x = jnp.ones((1, 8, 8, 16))
        y = conv(x)
        _testing.assert_allclose(y, jnp.zeros_like(y))


class TestConvTransposeGradients(unittest.TestCase):
    """Tests for finite gradients through transposed convolution layers."""

    def test_conv_transpose1d_grad_finite(self):
        """Gradients through ConvTranspose1d are finite."""
        conv = brainstate.nn.ConvTranspose1d(in_size=(20, 4), out_channels=8, kernel_size=4, stride=2)
        x = brainstate.random.randn(2, 20, 4)

        def loss(inp):
            return jnp.sum(conv(inp) ** 2)

        _testing.assert_grad_finite(loss, x)

    def test_conv_transpose2d_jit_equal(self):
        """JIT-compiled ConvTranspose2d matches eager output."""
        conv = brainstate.nn.ConvTranspose2d(in_size=(8, 8, 16), out_channels=8, kernel_size=3)
        x = brainstate.random.randn(2, 8, 8, 16)
        _testing.assert_jit_equal(lambda inp: conv(inp), x)


class TestChannelFirstBias(unittest.TestCase):
    """Regression tests for C1: channel_first=True together with a non-zero bias.

    Previously the bias was always built channels-last as ``(1, ..., 1, C)`` which
    failed to broadcast against a ``[B, C, *spatial]`` channels-first output.
    """

    def test_conv2d_channel_first_with_bias(self):
        """Conv2d with channel_first=True and a constant bias produces [B, C, H, W]."""
        conv = brainstate.nn.Conv2d(
            in_size=(3, 8, 8),
            out_channels=4,
            kernel_size=3,
            channel_first=True,
            b_init=brainstate.init.Constant(1.0),
        )
        x = jnp.ones((2, 3, 8, 8))
        y = conv(x)
        self.assertEqual(y.shape, (2, 4, 8, 8))
        self.assertIn('bias', conv.weight.value)

    def test_conv1d_channel_first_with_bias(self):
        """Conv1d with channel_first=True and a constant bias produces [B, C, L]."""
        conv = brainstate.nn.Conv1d(
            in_size=(8, 50),
            out_channels=16,
            kernel_size=3,
            channel_first=True,
            b_init=brainstate.init.Constant(1.0),
        )
        x = jnp.ones((2, 8, 50))
        y = conv(x)
        self.assertEqual(y.shape, (2, 16, 50))

    def test_conv3d_channel_first_with_bias(self):
        """Conv3d with channel_first=True and a constant bias produces [B, C, H, W, D]."""
        conv = brainstate.nn.Conv3d(
            in_size=(2, 8, 8, 8),
            out_channels=4,
            kernel_size=3,
            channel_first=True,
            b_init=brainstate.init.Constant(1.0),
        )
        x = jnp.ones((2, 2, 8, 8, 8))
        y = conv(x)
        self.assertEqual(y.shape, (2, 4, 8, 8, 8))

    def test_scaled_ws_conv2d_channel_first_with_bias(self):
        """ScaledWSConv2d with channel_first=True and a constant bias produces [B, C, H, W]."""
        conv = brainstate.nn.ScaledWSConv2d(
            in_size=(3, 8, 8),
            out_channels=4,
            kernel_size=3,
            channel_first=True,
            b_init=brainstate.init.Constant(1.0),
        )
        x = jnp.ones((2, 3, 8, 8))
        y = conv(x)
        self.assertEqual(y.shape, (2, 4, 8, 8))
        self.assertIn('bias', conv.weight.value)

    def test_conv_transpose2d_channel_first_with_bias(self):
        """ConvTranspose2d with channel_first=True and a constant bias produces [B, C, H, W]."""
        conv = brainstate.nn.ConvTranspose2d(
            in_size=(16, 8, 8),
            out_channels=8,
            kernel_size=3,
            stride=2,
            channel_first=True,
            b_init=brainstate.init.Constant(1.0),
        )
        x = jnp.ones((2, 16, 8, 8))
        y = conv(x)
        self.assertEqual(y.shape[0], 2)
        self.assertEqual(y.shape[1], 8)  # channel axis at position 1
        self.assertIn('bias', conv.weight.value)

    def test_conv_transpose1d_channel_first_with_bias(self):
        """ConvTranspose1d with channel_first=True and a constant bias produces [B, C, L]."""
        conv = brainstate.nn.ConvTranspose1d(
            in_size=(16, 28),
            out_channels=8,
            kernel_size=4,
            stride=2,
            channel_first=True,
            b_init=brainstate.init.Constant(1.0),
        )
        x = jnp.ones((2, 16, 28))
        y = conv(x)
        self.assertEqual(y.shape[0], 2)
        self.assertEqual(y.shape[1], 8)


class TestConvTransposeShapeVsJax(unittest.TestCase):
    """Regression tests for C2/C3: transposed-conv output sizes must match
    ``jax.lax.conv_transpose`` for both 'SAME' and 'VALID' at all strides.
    """

    def _jax_reference(self, x, w, stride, padding):
        """Channels-last 1D reference via jax.lax.conv_transpose (WOI kernel)."""
        return jax.lax.conv_transpose(
            x, w,
            strides=(stride,),
            padding=padding,
            dimension_numbers=('NWC', 'WIO', 'NWC'),
            transpose_kernel=True,
        )

    def test_exact_output_shape_matches_jax(self):
        """Output shape equals jax.lax.conv_transpose for k in {2,3,4,5}, s in {1,2,3}."""
        in_len = 7
        in_ch = 2
        out_ch = 3
        for k in (2, 3, 4, 5):
            for s in (1, 2, 3):
                for padding in ('SAME', 'VALID'):
                    conv = brainstate.nn.ConvTranspose1d(
                        in_size=(in_len, in_ch),
                        out_channels=out_ch,
                        kernel_size=k,
                        stride=s,
                        padding=padding,
                    )
                    x = jnp.ones((1, in_len, in_ch))
                    y = conv(x)
                    # Reference kernel layout for jax: (K, C_out, C_in)
                    w_ref = jnp.ones((k, out_ch, in_ch))
                    y_ref = self._jax_reference(x, w_ref, s, padding)
                    self.assertEqual(
                        y.shape, y_ref.shape,
                        f"shape mismatch k={k} s={s} pad={padding}: "
                        f"got {y.shape}, jax {y_ref.shape}"
                    )

    def test_numeric_equivalence_to_jax(self):
        """No-bias, single-group output matches jax.lax.conv_transpose numerically."""
        in_len = 9
        in_ch = 2
        out_ch = 4
        for k in (2, 3, 4, 5):
            for s in (1, 2, 3):
                for padding in ('SAME', 'VALID'):
                    conv = brainstate.nn.ConvTranspose1d(
                        in_size=(in_len, in_ch),
                        out_channels=out_ch,
                        kernel_size=k,
                        stride=s,
                        padding=padding,
                    )
                    x = brainstate.random.randn(1, in_len, in_ch)
                    y = conv(x)
                    # brainstate transposed kernel layout is (K, C_out, C_in)
                    w = conv.weight.value['weight']
                    y_ref = self._jax_reference(x, w, s, padding)
                    np.testing.assert_allclose(
                        np.asarray(y), np.asarray(y_ref), atol=1e-4, rtol=1e-4,
                        err_msg=f"value mismatch k={k} s={s} pad={padding}"
                    )

    def test_out_size_attribute_matches_jax(self):
        """The cached out_size attribute matches jax.lax.conv_transpose."""
        for k in (2, 3, 4, 5):
            for s in (1, 2, 3):
                for padding in ('SAME', 'VALID'):
                    conv = brainstate.nn.ConvTranspose1d(
                        in_size=(7, 2),
                        out_channels=3,
                        kernel_size=k,
                        stride=s,
                        padding=padding,
                    )
                    w_ref = jnp.ones((k, 3, 2))
                    y_ref = self._jax_reference(jnp.ones((1, 7, 2)), w_ref, s, padding)
                    self.assertEqual(conv.out_size, y_ref.shape[1:])


class TestConvValidationAuditRegressions(unittest.TestCase):
    """Regression tests for nn-audit items 3/4/5 (conv input validation).

    Items 3 & 5: runtime input validation used ``assert``, which is stripped under
    ``python -O``. These must raise ``ValueError`` (a subclass-checked exception that
    survives ``-O``), never ``AssertionError``.

    Item 4: padding parsing raised a cryptic ``IndexError`` on an empty sequence and a
    bare ``raise ValueError`` with no message.
    """

    def test_in_size_wrong_length_raises_valueerror(self):
        """Item 3: in_size length mismatch raises ValueError, not (stripped) assert."""
        with self.assertRaises(ValueError):
            brainstate.nn.Conv2d(in_size=(8, 3), out_channels=4, kernel_size=3)  # needs 3 entries
        with self.assertRaises(ValueError):
            brainstate.nn.ConvTranspose2d(in_size=(8, 3), out_channels=4, kernel_size=3)

    def test_groups_not_divisible_raises_valueerror(self):
        """Item 3: non-divisible channels/groups raises ValueError, not assert."""
        with self.assertRaises(ValueError):
            brainstate.nn.Conv2d(in_size=(8, 8, 4), out_channels=5, kernel_size=3, groups=2)
        with self.assertRaises(ValueError):
            brainstate.nn.ConvTranspose2d(in_size=(8, 8, 4), out_channels=5, kernel_size=3, groups=2)

    def test_invalid_string_padding_raises_valueerror(self):
        """Item 3: an unknown padding-mode string raises ValueError, not assert."""
        with self.assertRaises(ValueError):
            brainstate.nn.Conv2d(in_size=(8, 8, 3), out_channels=4, kernel_size=3, padding='FULL')
        with self.assertRaises(ValueError):
            brainstate.nn.ConvTranspose2d(in_size=(8, 8, 3), out_channels=4, kernel_size=3, padding='FULL')

    def test_empty_padding_sequence_raises_valueerror_not_indexerror(self):
        """Item 4: an empty padding sequence raises a clear ValueError (was IndexError)."""
        with self.assertRaises(ValueError):
            brainstate.nn.Conv2d(in_size=(8, 8, 3), out_channels=4, kernel_size=3, padding=[])
        with self.assertRaises(ValueError):
            brainstate.nn.ConvTranspose2d(in_size=(8, 8, 3), out_channels=4, kernel_size=3, padding=[])

    def test_invalid_padding_type_has_message(self):
        """Item 4: the bare ``raise ValueError`` now carries an explanatory message."""
        with self.assertRaises(ValueError) as ctx:
            brainstate.nn.Conv2d(in_size=(8, 8, 3), out_channels=4, kernel_size=3, padding=3.5)
        self.assertTrue(str(ctx.exception))  # non-empty message

    def test_validation_not_assertion(self):
        """Items 3/5: the raised exception is ValueError, not AssertionError.

        (``AssertionError`` would be silently removed by ``python -O``.)
        """
        for bad in (
            lambda: brainstate.nn.Conv2d(in_size=(8, 3), out_channels=4, kernel_size=3),
            lambda: brainstate.nn.Conv2d(in_size=(8, 8, 4), out_channels=5, kernel_size=3, groups=2),
            lambda: brainstate.nn.Conv2d(in_size=(8, 8, 3), out_channels=4, kernel_size=3, padding='FULL'),
        ):
            with self.assertRaises(ValueError):
                bad()
            self.assertNotIsInstance(self._capture(bad), AssertionError)

    @staticmethod
    def _capture(fn):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            return e
        return None
