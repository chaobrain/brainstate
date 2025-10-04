# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
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

"""
Comprehensive tests for padding layers.
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

import brainstate
from brainstate.nn._paddings import _format_padding


class TestPaddingFormatting(unittest.TestCase):
    """Test the _format_padding helper function."""

    def test_format_padding_1d_int(self):
        """Test formatting integer padding for 1D."""
        result = _format_padding(2, 1)
        self.assertEqual(result, [(2, 2)])

    def test_format_padding_2d_int(self):
        """Test formatting integer padding for 2D."""
        
        result = _format_padding(3, 2)
        self.assertEqual(result, [(3, 3), (3, 3)])

    def test_format_padding_3d_int(self):
        """Test formatting integer padding for 3D."""
        
        result = _format_padding(1, 3)
        self.assertEqual(result, [(1, 1), (1, 1), (1, 1)])

    def test_format_padding_1d_tuple_symmetric(self):
        """Test formatting symmetric tuple padding for 1D."""
        
        result = _format_padding([2], 1)
        self.assertEqual(result, [(2, 2)])

    def test_format_padding_2d_tuple_symmetric(self):
        """Test formatting symmetric tuple padding for 2D."""
        
        result = _format_padding([2, 3], 2)
        self.assertEqual(result, [(2, 2), (3, 3)])

    def test_format_padding_1d_tuple_asymmetric(self):
        """Test formatting asymmetric tuple padding for 1D."""
        
        result = _format_padding([1, 2], 1)
        self.assertEqual(result, [(1, 2)])

    def test_format_padding_2d_tuple_asymmetric(self):
        """Test formatting asymmetric tuple padding for 2D."""
        
        result = _format_padding([1, 2, 3, 4], 2)
        self.assertEqual(result, [(1, 2), (3, 4)])

    def test_format_padding_invalid_length(self):
        """Test that invalid padding length raises ValueError."""
        
        with self.assertRaises(ValueError):
            _format_padding([1, 2, 3], 2)  # Should be 2 or 4 elements for 2D


class TestReflectionPad1d(unittest.TestCase):
    """Test ReflectionPad1d class."""

    def test_reflection_pad1d_2d_input(self):
        """Test ReflectionPad1d with 2D input (length, channels)."""
        pad = brainstate.nn.ReflectionPad1d(2)
        x = jnp.array([[1, 2], [3, 4], [5, 6]])  # (3, 2)
        output = pad(x)
        self.assertEqual(output.shape, (7, 2))  # 3 + 2*2 = 7
        # Check reflection pattern for first channel
        expected_first_channel = jnp.array([5, 3, 1, 3, 5, 3, 1])
        np.testing.assert_array_equal(output[:, 0], expected_first_channel)

    def test_reflection_pad1d_3d_input(self):
        """Test ReflectionPad1d with 3D input (batch, length, channels)."""
        pad = brainstate.nn.ReflectionPad1d(1)
        x = jnp.ones((2, 4, 3))
        output = pad(x)
        self.assertEqual(output.shape, (2, 6, 3))  # 4 + 2*1 = 6

    def test_reflection_pad1d_asymmetric(self):
        """Test ReflectionPad1d with asymmetric padding."""
        pad = brainstate.nn.ReflectionPad1d([1, 2])
        x = jnp.ones((2, 5, 3))
        output = pad(x)
        self.assertEqual(output.shape, (2, 8, 3))  # 5 + 1 + 2 = 8

    def test_reflection_pad1d_with_in_size(self):
        """Test ReflectionPad1d with in_size parameter."""
        pad = brainstate.nn.ReflectionPad1d(2, in_size=(10, 3))
        self.assertEqual(pad.out_size, (14, 3))

    def test_reflection_pad1d_invalid_shape(self):
        """Test that ReflectionPad1d raises error for invalid input shape."""
        pad = brainstate.nn.ReflectionPad1d(2)
        x = jnp.ones((2, 3, 4, 5))  # 4D input
        with self.assertRaises(ValueError):
            pad(x)


class TestReflectionPad2d(unittest.TestCase):
    """Test ReflectionPad2d class."""

    def test_reflection_pad2d_3d_input(self):
        """Test ReflectionPad2d with 3D input (height, width, channels)."""
        pad = brainstate.nn.ReflectionPad2d(1)
        x = jnp.ones((4, 4, 3))
        output = pad(x)
        self.assertEqual(output.shape, (6, 6, 3))

    def test_reflection_pad2d_4d_input(self):
        """Test ReflectionPad2d with 4D input (batch, height, width, channels)."""
        pad = brainstate.nn.ReflectionPad2d(2)
        x = jnp.ones((2, 3, 3, 1))
        output = pad(x)
        self.assertEqual(output.shape, (2, 7, 7, 1))

    def test_reflection_pad2d_different_hw_padding(self):
        """Test ReflectionPad2d with different height and width padding."""
        pad = brainstate.nn.ReflectionPad2d([1, 2])
        x = jnp.ones((2, 4, 4, 3))
        output = pad(x)
        self.assertEqual(output.shape, (2, 6, 8, 3))  # 4+2*1, 4+2*2

    def test_reflection_pad2d_asymmetric(self):
        """Test ReflectionPad2d with asymmetric padding."""
        pad = brainstate.nn.ReflectionPad2d([1, 2, 3, 4])  # left, right, top, bottom
        x = jnp.ones((2, 5, 5, 3))
        output = pad(x)
        # For 2D: first pair is height (top/bottom), second is width (left/right)
        # Height: 5+1+2=8, Width: 5+3+4=12
        self.assertEqual(output.shape, (2, 8, 12, 3))

    def test_reflection_pad2d_with_in_size(self):
        """Test ReflectionPad2d with in_size parameter."""
        pad = brainstate.nn.ReflectionPad2d([1, 2], in_size=(10, 10, 3))
        self.assertEqual(pad.out_size, (12, 14, 3))


class TestReflectionPad3d(unittest.TestCase):
    """Test ReflectionPad3d class."""

    def test_reflection_pad3d_4d_input(self):
        """Test ReflectionPad3d with 4D input (depth, height, width, channels)."""
        pad = brainstate.nn.ReflectionPad3d(1)
        x = jnp.ones((3, 4, 4, 2))
        output = pad(x)
        self.assertEqual(output.shape, (5, 6, 6, 2))

    def test_reflection_pad3d_5d_input(self):
        """Test ReflectionPad3d with 5D input (batch, depth, height, width, channels)."""
        pad = brainstate.nn.ReflectionPad3d(2)
        x = jnp.ones((1, 3, 3, 3, 1))
        output = pad(x)
        self.assertEqual(output.shape, (1, 7, 7, 7, 1))

    def test_reflection_pad3d_different_padding(self):
        """Test ReflectionPad3d with different padding for each dimension."""
        pad = brainstate.nn.ReflectionPad3d([1, 2, 3])
        x = jnp.ones((2, 4, 4, 4, 3))
        output = pad(x)
        self.assertEqual(output.shape, (2, 6, 8, 10, 3))

    def test_reflection_pad3d_with_in_size(self):
        """Test ReflectionPad3d with in_size parameter."""
        pad = brainstate.nn.ReflectionPad3d(1, in_size=(1, 8, 8, 8, 3))
        self.assertEqual(pad.out_size, (1, 10, 10, 10, 3))


class TestReplicationPad1d(unittest.TestCase):
    """Test ReplicationPad1d class."""

    def test_replication_pad1d_2d_input(self):
        """Test ReplicationPad1d with 2D input."""
        pad = brainstate.nn.ReplicationPad1d(2)
        x = jnp.array([[1, 2], [3, 4], [5, 6]])
        output = pad(x)
        self.assertEqual(output.shape, (7, 2))
        # Check that edges are replicated
        np.testing.assert_array_equal(output[0, :], x[0, :])
        np.testing.assert_array_equal(output[1, :], x[0, :])
        np.testing.assert_array_equal(output[-1, :], x[-1, :])
        np.testing.assert_array_equal(output[-2, :], x[-1, :])

    def test_replication_pad1d_3d_input(self):
        """Test ReplicationPad1d with 3D input."""
        pad = brainstate.nn.ReplicationPad1d([1, 3])
        x = jnp.ones((2, 4, 3))
        output = pad(x)
        self.assertEqual(output.shape, (2, 8, 3))

    def test_replication_pad1d_with_in_size(self):
        """Test ReplicationPad1d with in_size parameter."""
        pad = brainstate.nn.ReplicationPad1d(3, in_size=(2, 10, 5))
        self.assertEqual(pad.out_size, (2, 16, 5))


class TestReplicationPad2d(unittest.TestCase):
    """Test ReplicationPad2d class."""

    def test_replication_pad2d_3d_input(self):
        """Test ReplicationPad2d with 3D input."""
        pad = brainstate.nn.ReplicationPad2d(1)
        x = jnp.arange(12).reshape(3, 4, 1)
        output = pad(x)
        self.assertEqual(output.shape, (5, 6, 1))
        # Check corners are replicated correctly
        self.assertEqual(output[0, 0, 0], x[0, 0, 0])
        self.assertEqual(output[-1, -1, 0], x[-1, -1, 0])

    def test_replication_pad2d_4d_input(self):
        """Test ReplicationPad2d with 4D input."""
        pad = brainstate.nn.ReplicationPad2d([2, 1])
        x = jnp.ones((2, 5, 5, 3))
        output = pad(x)
        self.assertEqual(output.shape, (2, 9, 7, 3))

    def test_replication_pad2d_asymmetric(self):
        """Test ReplicationPad2d with asymmetric padding."""
        pad = brainstate.nn.ReplicationPad2d([1, 3, 2, 4])
        x = jnp.ones((1, 4, 4, 2))
        output = pad(x)
        # Height: 4+1+3=8, Width: 4+2+4=10
        self.assertEqual(output.shape, (1, 8, 10, 2))


class TestReplicationPad3d(unittest.TestCase):
    """Test ReplicationPad3d class."""

    def test_replication_pad3d_4d_input(self):
        """Test ReplicationPad3d with 4D input."""
        pad = brainstate.nn.ReplicationPad3d(1)
        x = jnp.ones((3, 3, 3, 2))
        output = pad(x)
        self.assertEqual(output.shape, (5, 5, 5, 2))

    def test_replication_pad3d_5d_input(self):
        """Test ReplicationPad3d with 5D input."""
        pad = brainstate.nn.ReplicationPad3d([1, 2, 3])
        x = jnp.ones((2, 4, 4, 4, 1))
        output = pad(x)
        self.assertEqual(output.shape, (2, 6, 8, 10, 1))


class TestZeroPad1d(unittest.TestCase):
    """Test ZeroPad1d class."""

    def test_zero_pad1d_2d_input(self):
        """Test ZeroPad1d with 2D input."""
        pad = brainstate.nn.ZeroPad1d(2)
        x = jnp.ones((3, 2))
        output = pad(x)
        self.assertEqual(output.shape, (7, 2))
        # Check that padding is zeros
        np.testing.assert_array_equal(output[:2, :], jnp.zeros((2, 2)))
        np.testing.assert_array_equal(output[-2:, :], jnp.zeros((2, 2)))

    def test_zero_pad1d_3d_input(self):
        """Test ZeroPad1d with 3D input."""
        pad = brainstate.nn.ZeroPad1d([3, 1])
        x = jnp.ones((1, 5, 3))
        output = pad(x)
        self.assertEqual(output.shape, (1, 9, 3))
        # Check padding values
        np.testing.assert_array_equal(output[0, :3, :], jnp.zeros((3, 3)))
        np.testing.assert_array_equal(output[0, -1:, :], jnp.zeros((1, 3)))

    def test_zero_pad1d_with_in_size(self):
        """Test ZeroPad1d with in_size parameter."""
        pad = brainstate.nn.ZeroPad1d(5, in_size=(20, 10))
        self.assertEqual(pad.out_size, (30, 10))


class TestZeroPad2d(unittest.TestCase):
    """Test ZeroPad2d class."""

    def test_zero_pad2d_3d_input(self):
        """Test ZeroPad2d with 3D input."""
        pad = brainstate.nn.ZeroPad2d(1)
        x = jnp.ones((3, 3, 2))
        output = pad(x)
        self.assertEqual(output.shape, (5, 5, 2))
        # Check borders are zero
        np.testing.assert_array_equal(output[0, :, :], jnp.zeros((5, 2)))
        np.testing.assert_array_equal(output[-1, :, :], jnp.zeros((5, 2)))
        np.testing.assert_array_equal(output[:, 0, :], jnp.zeros((5, 2)))
        np.testing.assert_array_equal(output[:, -1, :], jnp.zeros((5, 2)))

    def test_zero_pad2d_4d_input(self):
        """Test ZeroPad2d with 4D input."""
        pad = brainstate.nn.ZeroPad2d([2, 3])
        x = jnp.ones((2, 4, 4, 3))
        output = pad(x)
        self.assertEqual(output.shape, (2, 8, 10, 3))

    def test_zero_pad2d_asymmetric(self):
        """Test ZeroPad2d with asymmetric padding."""
        pad = brainstate.nn.ZeroPad2d([1, 2, 3, 4])
        x = jnp.ones((1, 5, 5, 1))
        output = pad(x)
        # Height: 5+1+2=8, Width: 5+3+4=12
        self.assertEqual(output.shape, (1, 8, 12, 1))


class TestZeroPad3d(unittest.TestCase):
    """Test ZeroPad3d class."""

    def test_zero_pad3d_4d_input(self):
        """Test ZeroPad3d with 4D input."""
        pad = brainstate.nn.ZeroPad3d(1)
        x = jnp.ones((2, 2, 2, 3))
        output = pad(x)
        self.assertEqual(output.shape, (4, 4, 4, 3))
        # Check that corners are zero
        self.assertEqual(output[0, 0, 0, 0], 0)
        self.assertEqual(output[-1, -1, -1, 0], 0)

    def test_zero_pad3d_5d_input(self):
        """Test ZeroPad3d with 5D input."""
        pad = brainstate.nn.ZeroPad3d([1, 1, 2])
        x = jnp.ones((1, 3, 3, 3, 2))
        output = pad(x)
        self.assertEqual(output.shape, (1, 5, 5, 7, 2))


class TestConstantPad1d(unittest.TestCase):
    """Test ConstantPad1d class."""

    def test_constant_pad1d_default_value(self):
        """Test ConstantPad1d with default value (0)."""
        pad = brainstate.nn.ConstantPad1d(2)
        x = jnp.ones((3, 2))
        output = pad(x)
        self.assertEqual(output.shape, (7, 2))
        np.testing.assert_array_equal(output[:2, :], jnp.zeros((2, 2)))

    def test_constant_pad1d_custom_value(self):
        """Test ConstantPad1d with custom value."""
        pad = brainstate.nn.ConstantPad1d(1, value=3.14)
        x = jnp.ones((2, 4, 3))
        output = pad(x)
        self.assertEqual(output.shape, (2, 6, 3))
        np.testing.assert_allclose(output[:, 0, :], 3.14)
        np.testing.assert_allclose(output[:, -1, :], 3.14)

    def test_constant_pad1d_asymmetric(self):
        """Test ConstantPad1d with asymmetric padding."""
        pad = brainstate.nn.ConstantPad1d([2, 3], value=-1)
        x = jnp.zeros((1, 5, 2))
        output = pad(x)
        self.assertEqual(output.shape, (1, 10, 2))
        np.testing.assert_array_equal(output[0, :2, :], -jnp.ones((2, 2)))
        np.testing.assert_array_equal(output[0, -3:, :], -jnp.ones((3, 2)))

    def test_constant_pad1d_with_in_size(self):
        """Test ConstantPad1d with in_size parameter."""
        pad = brainstate.nn.ConstantPad1d(4, value=2.5, in_size=(15, 8))
        self.assertEqual(pad.out_size, (23, 8))


class TestConstantPad2d(unittest.TestCase):
    """Test ConstantPad2d class."""

    def test_constant_pad2d_default_value(self):
        """Test ConstantPad2d with default value."""
        pad = brainstate.nn.ConstantPad2d(1)
        x = jnp.ones((3, 3, 1))
        output = pad(x)
        self.assertEqual(output.shape, (5, 5, 1))
        # Check borders are zero
        self.assertEqual(output[0, 0, 0], 0)

    def test_constant_pad2d_custom_value(self):
        """Test ConstantPad2d with custom value."""
        pad = brainstate.nn.ConstantPad2d([1, 2], value=5.0)
        x = jnp.zeros((2, 4, 4, 3))
        output = pad(x)
        self.assertEqual(output.shape, (2, 6, 8, 3))
        np.testing.assert_array_equal(output[0, 0, :, 0], 5 * jnp.ones(8))
        np.testing.assert_array_equal(output[0, :, 0, 0], 5 * jnp.ones(6))

    def test_constant_pad2d_negative_value(self):
        """Test ConstantPad2d with negative padding value."""
        pad = brainstate.nn.ConstantPad2d(2, value=-2.5)
        x = jnp.ones((1, 2, 2, 1))
        output = pad(x)
        self.assertEqual(output.shape, (1, 6, 6, 1))
        self.assertEqual(output[0, 0, 0, 0], -2.5)


class TestConstantPad3d(unittest.TestCase):
    """Test ConstantPad3d class."""

    def test_constant_pad3d_default_value(self):
        """Test ConstantPad3d with default value."""
        pad = brainstate.nn.ConstantPad3d(1)
        x = jnp.ones((2, 2, 2, 1))
        output = pad(x)
        self.assertEqual(output.shape, (4, 4, 4, 1))

    def test_constant_pad3d_custom_value(self):
        """Test ConstantPad3d with custom value."""
        pad = brainstate.nn.ConstantPad3d([1, 2, 3], value=10)
        x = jnp.zeros((1, 3, 3, 3, 2))
        output = pad(x)
        self.assertEqual(output.shape, (1, 5, 7, 9, 2))
        # Check that padding has correct value
        self.assertEqual(output[0, 0, 0, 0, 0], 10)
        self.assertEqual(output[0, -1, -1, -1, 0], 10)

    def test_constant_pad3d_asymmetric(self):
        """Test ConstantPad3d with asymmetric padding."""
        pad = brainstate.nn.ConstantPad3d([1, 2, 3, 4, 5, 6], value=7)
        x = jnp.ones((1, 2, 2, 2, 1))
        output = pad(x)
        self.assertEqual(output.shape, (1, 5, 9, 13, 1))


class TestCircularPad1d(unittest.TestCase):
    """Test CircularPad1d class."""

    def test_circular_pad1d_2d_input(self):
        """Test CircularPad1d with 2D input."""
        pad = brainstate.nn.CircularPad1d(2)
        x = jnp.array([[1], [2], [3], [4], [5]])
        output = pad(x)
        self.assertEqual(output.shape, (9, 1))
        # Check circular pattern
        expected = jnp.array([[4], [5], [1], [2], [3], [4], [5], [1], [2]])
        np.testing.assert_array_equal(output, expected)

    def test_circular_pad1d_3d_input(self):
        """Test CircularPad1d with 3D input."""
        pad = brainstate.nn.CircularPad1d(1)
        x = jnp.arange(12).reshape(2, 3, 2)
        output = pad(x)
        self.assertEqual(output.shape, (2, 5, 2))
        # Check wrapping for first batch
        np.testing.assert_array_equal(output[0, 0, :], x[0, -1, :])
        np.testing.assert_array_equal(output[0, -1, :], x[0, 0, :])

    def test_circular_pad1d_asymmetric(self):
        """Test CircularPad1d with asymmetric padding."""
        pad = brainstate.nn.CircularPad1d([1, 2])
        x = jnp.array([[[1], [2], [3]]])
        output = pad(x)
        self.assertEqual(output.shape, (1, 6, 1))
        expected = jnp.array([[[3], [1], [2], [3], [1], [2]]])
        np.testing.assert_array_equal(output, expected)

    def test_circular_pad1d_with_in_size(self):
        """Test CircularPad1d with in_size parameter."""
        pad = brainstate.nn.CircularPad1d(3, in_size=(2, 10, 4))
        self.assertEqual(pad.out_size, (2, 16, 4))


class TestCircularPad2d(unittest.TestCase):
    """Test CircularPad2d class."""

    def test_circular_pad2d_3d_input(self):
        """Test CircularPad2d with 3D input."""
        pad = brainstate.nn.CircularPad2d(1)
        x = jnp.arange(9).reshape(3, 3, 1)
        output = pad(x)
        self.assertEqual(output.shape, (5, 5, 1))
        # Check corners wrap correctly
        self.assertEqual(output[0, 0, 0], x[-1, -1, 0])
        self.assertEqual(output[0, -1, 0], x[-1, 0, 0])
        self.assertEqual(output[-1, 0, 0], x[0, -1, 0])
        self.assertEqual(output[-1, -1, 0], x[0, 0, 0])

    def test_circular_pad2d_4d_input(self):
        """Test CircularPad2d with 4D input."""
        pad = brainstate.nn.CircularPad2d([1, 2])
        x = jnp.ones((2, 3, 3, 2))
        output = pad(x)
        self.assertEqual(output.shape, (2, 5, 7, 2))

    def test_circular_pad2d_different_padding(self):
        """Test CircularPad2d with different padding for height and width."""
        pad = brainstate.nn.CircularPad2d([2, 1])
        x = jnp.arange(16).reshape(1, 4, 4, 1)
        output = pad(x)
        self.assertEqual(output.shape, (1, 8, 6, 1))


class TestCircularPad3d(unittest.TestCase):
    """Test CircularPad3d class."""

    def test_circular_pad3d_4d_input(self):
        """Test CircularPad3d with 4D input."""
        pad = brainstate.nn.CircularPad3d(1)
        x = jnp.arange(8).reshape(2, 2, 2, 1)
        output = pad(x)
        self.assertEqual(output.shape, (4, 4, 4, 1))
        # Check that values wrap around
        self.assertEqual(output[0, 0, 0, 0], x[-1, -1, -1, 0])
        self.assertEqual(output[-1, -1, -1, 0], x[0, 0, 0, 0])

    def test_circular_pad3d_5d_input(self):
        """Test CircularPad3d with 5D input."""
        pad = brainstate.nn.CircularPad3d([1, 1, 2])
        x = jnp.ones((1, 2, 2, 2, 3))
        output = pad(x)
        self.assertEqual(output.shape, (1, 4, 4, 6, 3))

    def test_circular_pad3d_asymmetric(self):
        """Test CircularPad3d with asymmetric padding."""
        pad = brainstate.nn.CircularPad3d([1, 0, 0, 1, 1, 2])
        x = jnp.ones((1, 3, 3, 3, 1))
        output = pad(x)
        self.assertEqual(output.shape, (1, 4, 4, 6, 1))


class TestPaddingIntegration(unittest.TestCase):
    """Integration tests for padding layers."""

    def test_all_1d_paddings_same_interface(self):
        """Test that all 1D padding classes have the same interface."""
        padding_classes = [
            brainstate.nn.ReflectionPad1d,
            brainstate.nn.ReplicationPad1d,
            brainstate.nn.ZeroPad1d,
            brainstate.nn.ConstantPad1d,
            brainstate.nn.CircularPad1d
        ]

        x = jnp.ones((2, 10, 3))
        for PadClass in padding_classes:
            if PadClass == brainstate.nn.ConstantPad1d:
                pad = PadClass(2, value=0)
            else:
                pad = PadClass(2)
            output = pad(x)
            self.assertEqual(output.shape, (2, 14, 3))

    def test_all_2d_paddings_same_interface(self):
        """Test that all 2D padding classes have the same interface."""
        padding_classes = [
            brainstate.nn.ReflectionPad2d,
            brainstate.nn.ReplicationPad2d,
            brainstate.nn.ZeroPad2d,
            brainstate.nn.ConstantPad2d,
            brainstate.nn.CircularPad2d
        ]

        x = jnp.ones((2, 8, 8, 3))
        for PadClass in padding_classes:
            if PadClass == brainstate.nn.ConstantPad2d:
                pad = PadClass([1, 2], value=0)
            else:
                pad = PadClass([1, 2])
            output = pad(x)
            self.assertEqual(output.shape, (2, 10, 12, 3))

    def test_all_3d_paddings_same_interface(self):
        """Test that all 3D padding classes have the same interface."""
        padding_classes = [
            brainstate.nn.ReflectionPad3d,
            brainstate.nn.ReplicationPad3d,
            brainstate.nn.ZeroPad3d,
            brainstate.nn.ConstantPad3d,
            brainstate.nn.CircularPad3d
        ]

        x = jnp.ones((1, 4, 4, 4, 2))
        for PadClass in padding_classes:
            if PadClass == brainstate.nn.ConstantPad3d:
                pad = PadClass([1, 2, 3], value=0)
            else:
                pad = PadClass([1, 2, 3])
            output = pad(x)
            self.assertEqual(output.shape, (1, 6, 8, 10, 2))

    def test_in_size_out_size_consistency(self):
        """Test that in_size and out_size are consistent with actual output."""
        test_cases = [
            (brainstate.nn.ReflectionPad1d(2), (10, 3)),
            (brainstate.nn.ReplicationPad2d([1, 2]), (8, 8, 3)),
            (brainstate.nn.ZeroPad3d(1), (4, 4, 4, 2)),
            (brainstate.nn.ConstantPad1d([1, 3], value=0), (2, 10, 5)),
            (brainstate.nn.CircularPad2d([2, 2]), (1, 6, 6, 1)),
        ]

        for pad, in_size in test_cases:
            pad_with_size = type(pad)(
                pad.padding[0][0] if hasattr(pad, 'padding') else 1,
                in_size=in_size
            )
            x = jnp.ones(in_size)
            output = pad_with_size(x)
            self.assertEqual(output.shape, pad_with_size.out_size)

    def test_gradient_flow(self):
        """Test that gradients flow correctly through padding layers."""
        def loss_fn(x, pad_layer):
            output = pad_layer(x)
            return jnp.sum(output ** 2)

        x = jnp.ones((2, 4, 3))
        pad = brainstate.nn.ReflectionPad1d(2)

        grad_fn = jax.grad(loss_fn, argnums=0)
        grad = grad_fn(x, pad)

        # Gradient should be non-zero
        self.assertGreater(jnp.sum(jnp.abs(grad)), 0)
        # Gradient should have same shape as input
        self.assertEqual(grad.shape, x.shape)

    def test_jit_compilation(self):
        """Test that padding layers work with JIT compilation."""
        @jax.jit
        def apply_padding(x):
            pad = brainstate.nn.ZeroPad2d(2)
            return pad(x)

        x = jnp.ones((1, 4, 4, 3))
        output = apply_padding(x)
        self.assertEqual(output.shape, (1, 8, 8, 3))

    def test_vmap_compatibility(self):
        """Test that padding layers work with vmap."""
        pad = brainstate.nn.CircularPad1d(1)

        # Create batch of different inputs
        x = jnp.arange(30).reshape(5, 6, 1)

        # Apply padding with vmap
        vmapped_pad = jax.vmap(pad.update)
        output = vmapped_pad(x)

        self.assertEqual(output.shape, (5, 8, 1))

        # Check that each batch element is padded correctly
        for i in range(5):
            single_output = pad(x[i])
            np.testing.assert_array_equal(output[i], single_output)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""

    def test_zero_padding(self):
        """Test padding with size 0."""
        pad = brainstate.nn.ZeroPad1d(0)
        x = jnp.ones((3, 5, 2))
        output = pad(x)
        np.testing.assert_array_equal(output, x)

    def test_large_padding(self):
        """Test with very large padding."""
        pad = brainstate.nn.ConstantPad1d(100, value=3.14)
        x = jnp.ones((1, 2, 1))
        output = pad(x)
        self.assertEqual(output.shape, (1, 202, 1))

    def test_reflection_pad_boundary_smaller_than_padding(self):
        """Test reflection padding when padding is larger than input."""
        # This might fail for reflection padding if input is too small
        pad = brainstate.nn.ReflectionPad1d(3)
        x = jnp.ones((2, 2))  # Length 2, padding 3
        # Reflection requires input size > padding size
        # This should work fine with JAX's pad implementation
        output = pad(x)
        self.assertEqual(output.shape, (8, 2))

    def test_different_dtypes(self):
        """Test padding with different data types."""
        dtypes = [jnp.float32, jnp.int32]
        pad = brainstate.nn.ZeroPad1d(1)

        for dtype in dtypes:
            x = jnp.ones((2, 3, 1), dtype=dtype)
            output = pad(x)
            # JAX may truncate float64 to float32 depending on config
            if dtype == jnp.float64:
                # Check if it's either float64 or float32 (truncated)
                self.assertIn(output.dtype, [jnp.float32, jnp.float64])
            else:
                self.assertEqual(output.dtype, dtype)
            self.assertEqual(output.shape, (2, 5, 1))

        # Test float64 separately with proper handling
        try:
            x = jnp.ones((2, 3, 1), dtype=jnp.float64)
            output = pad(x)
            # Output dtype might be float32 if JAX_ENABLE_X64 is not set
            self.assertIn(str(output.dtype), ['float32', 'float64'])
            self.assertEqual(output.shape, (2, 5, 1))
        except:
            pass  # Float64 not available

    def test_empty_batch(self):
        """Test padding with empty batch dimension."""
        pad = brainstate.nn.CircularPad2d(1)
        x = jnp.ones((0, 4, 4, 3))
        output = pad(x)
        self.assertEqual(output.shape, (0, 6, 6, 3))


if __name__ == '__main__':
    unittest.main()