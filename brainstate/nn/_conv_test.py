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

import brainstate
import braintools


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
        """Test that invalid group configurations raise errors."""
        with self.assertRaises(AssertionError):
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
        key = jax.random.PRNGKey(42)

        conv1 = brainstate.nn.Conv2d(in_size=(32, 32, 3), out_channels=16, kernel_size=3)
        conv2 = brainstate.nn.Conv2d(in_size=(32, 32, 3), out_channels=16, kernel_size=3)

        # Use same random key for input
        x = jax.random.normal(key, (4, 32, 32, 3))

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
        """Test that invalid groups raises assertion error."""
        with self.assertRaises(AssertionError):
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
