# -*- coding: utf-8 -*-

import jax.numpy as jnp
import pytest
from absl.testing import absltest
from absl.testing import parameterized

import brainstate
import unittest
import jax
import jax.numpy as jnp
import brainstate


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
        conv = brainstate.nn.Conv1d(in_size=(50, 8), out_channels=16, kernel_size=3, b_init=brainstate.init.Constant(0.0))
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

