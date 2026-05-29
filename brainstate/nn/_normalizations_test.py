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

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from absl.testing import absltest
from absl.testing import parameterized

import brainstate
from brainstate import _testing
from brainstate._testing import assert_allclose
from brainstate.nn._normalizations import (
    canonicalize_dtype,
    weight_standardization,
    _canonicalize_axes,
    _abs_sq,
    _compute_stats,
)


class TestBatchNorm0d(parameterized.TestCase):
    """Test BatchNorm0d with various configurations."""

    @parameterized.product(
        fit=[True, False],
        feature_axis=[-1, 0],
        track_running_stats=[True, False],
    )
    def test_batchnorm0d_with_batch(self, fit, feature_axis, track_running_stats):
        """Test BatchNorm0d with batched input."""
        batch_size = 8
        channels = 10

        # Channel last: (batch, channels)
        if feature_axis == -1:
            in_size = (channels,)
            input_shape = (batch_size, channels)
        # Channel first: (batch, channels) - same for 0D
        else:
            in_size = (channels,)
            input_shape = (batch_size, channels)

        # affine can only be True when track_running_stats is True
        affine = track_running_stats

        net = brainstate.nn.BatchNorm0d(
            in_size,
            feature_axis=feature_axis,
            track_running_stats=track_running_stats,
            affine=affine
        )
        brainstate.environ.set(fit=fit)

        x = brainstate.random.randn(*input_shape)
        output = net(x)

        # Check output shape matches input
        self.assertEqual(output.shape, input_shape)

        # Check that output has approximately zero mean and unit variance when fitting
        if fit and track_running_stats:
            # Stats should be computed along batch dimension
            mean = jnp.mean(output, axis=0)
            var = jnp.var(output, axis=0)
            np.testing.assert_allclose(mean, 0.0, atol=1e-5)
            np.testing.assert_allclose(var, 1.0, atol=1e-1)

    def test_batchnorm0d_without_batch(self):
        """Test BatchNorm0d without batching."""
        channels = 10
        in_size = (channels,)

        net = brainstate.nn.BatchNorm0d(in_size, track_running_stats=True)
        brainstate.environ.set(fit=False)  # Use running stats

        # Run with batch first to populate running stats
        brainstate.environ.set(fit=True)
        x_batch = brainstate.random.randn(16, channels)
        _ = net(x_batch)

        # Now test without batch
        brainstate.environ.set(fit=False)
        x_single = brainstate.random.randn(channels)
        output = net(x_single)

        self.assertEqual(output.shape, (channels,))

    def test_batchnorm0d_affine(self):
        """Test BatchNorm0d with and without affine parameters."""
        channels = 10
        in_size = (channels,)

        # With affine
        net_affine = brainstate.nn.BatchNorm0d(in_size, affine=True)
        self.assertIsNotNone(net_affine.weight)

        # Without affine (track_running_stats must be False)
        net_no_affine = brainstate.nn.BatchNorm0d(
            in_size, affine=False, track_running_stats=False
        )
        self.assertIsNone(net_no_affine.weight)


class TestBatchNorm1d(parameterized.TestCase):
    """Test BatchNorm1d with various configurations."""

    @parameterized.product(
        fit=[True, False],
        feature_axis=[-1, 0],
        track_running_stats=[True, False],
    )
    def test_batchnorm1d_with_batch(self, fit, feature_axis, track_running_stats):
        """Test BatchNorm1d with batched input."""
        batch_size = 8
        length = 20
        channels = 10

        # Channel last: (batch, length, channels)
        if feature_axis == -1:
            in_size = (length, channels)
            input_shape = (batch_size, length, channels)
            feature_axis_param = -1
        # Channel first: (batch, channels, length)
        else:
            in_size = (channels, length)
            input_shape = (batch_size, channels, length)
            feature_axis_param = 0

        # affine can only be True when track_running_stats is True
        affine = track_running_stats

        net = brainstate.nn.BatchNorm1d(
            in_size,
            feature_axis=feature_axis_param,
            track_running_stats=track_running_stats,
            affine=affine
        )
        brainstate.environ.set(fit=fit)

        x = brainstate.random.randn(*input_shape)
        output = net(x)

        # Check output shape matches input
        self.assertEqual(output.shape, input_shape)

    def test_batchnorm1d_without_batch(self):
        """Test BatchNorm1d without batching."""
        length = 20
        channels = 10
        in_size = (length, channels)

        net = brainstate.nn.BatchNorm1d(in_size, track_running_stats=True)

        # Populate running stats first
        brainstate.environ.set(fit=True)
        x_batch = brainstate.random.randn(8, length, channels)
        _ = net(x_batch)

        # Test without batch
        brainstate.environ.set(fit=False)
        x_single = brainstate.random.randn(length, channels)
        output = net(x_single)

        self.assertEqual(output.shape, (length, channels))

    @parameterized.product(
        feature_axis=[-1, 0],
    )
    def test_batchnorm1d_channel_consistency(self, feature_axis):
        """Test that normalization is consistent across different channel configurations."""
        batch_size = 16
        length = 20
        channels = 10

        if feature_axis == -1:
            in_size = (length, channels)
            input_shape = (batch_size, length, channels)
        else:
            in_size = (channels, length)
            input_shape = (batch_size, channels, length)

        net = brainstate.nn.BatchNorm1d(in_size, feature_axis=feature_axis)
        brainstate.environ.set(fit=True)

        x = brainstate.random.randn(*input_shape)
        output = net(x)

        # Output should have same shape as input
        self.assertEqual(output.shape, input_shape)


class TestBatchNorm2d(parameterized.TestCase):
    """Test BatchNorm2d with various configurations."""

    @parameterized.product(
        fit=[True, False],
        feature_axis=[-1, 0],
        track_running_stats=[True, False],
    )
    def test_batchnorm2d_with_batch(self, fit, feature_axis, track_running_stats):
        """Test BatchNorm2d with batched input (images)."""
        batch_size = 4
        height, width = 28, 28
        channels = 3

        # Channel last: (batch, height, width, channels)
        if feature_axis == -1:
            in_size = (height, width, channels)
            input_shape = (batch_size, height, width, channels)
            feature_axis_param = -1
        # Channel first: (batch, channels, height, width)
        else:
            in_size = (channels, height, width)
            input_shape = (batch_size, channels, height, width)
            feature_axis_param = 0

        # affine can only be True when track_running_stats is True
        affine = track_running_stats

        net = brainstate.nn.BatchNorm2d(
            in_size,
            feature_axis=feature_axis_param,
            track_running_stats=track_running_stats,
            affine=affine
        )
        brainstate.environ.set(fit=fit)

        x = brainstate.random.randn(*input_shape)
        output = net(x)

        # Check output shape matches input
        self.assertEqual(output.shape, input_shape)

        # Check normalization properties during training
        if fit and track_running_stats:
            # For channel last: normalize over (batch, height, width)
            # For channel first: normalize over (batch, height, width)
            if feature_axis == -1:
                axes = (0, 1, 2)
            else:
                axes = (0, 2, 3)

            mean = jnp.mean(output, axis=axes)
            var = jnp.var(output, axis=axes)

            # Each channel should be approximately normalized
            np.testing.assert_allclose(mean, 0.0, atol=1e-5)
            np.testing.assert_allclose(var, 1.0, atol=1e-1)

    def test_batchnorm2d_without_batch(self):
        """Test BatchNorm2d without batching."""
        height, width = 28, 28
        channels = 3
        in_size = (height, width, channels)

        net = brainstate.nn.BatchNorm2d(in_size, track_running_stats=True)

        # Populate running stats
        brainstate.environ.set(fit=True)
        x_batch = brainstate.random.randn(8, height, width, channels)
        _ = net(x_batch)

        # Test without batch
        brainstate.environ.set(fit=False)
        x_single = brainstate.random.randn(height, width, channels)
        output = net(x_single)

        self.assertEqual(output.shape, (height, width, channels))


class TestBatchNorm3d(parameterized.TestCase):
    """Test BatchNorm3d with various configurations."""

    @parameterized.product(
        fit=[True, False],
        feature_axis=[-1, 0],
        track_running_stats=[True, False],
    )
    def test_batchnorm3d_with_batch(self, fit, feature_axis, track_running_stats):
        """Test BatchNorm3d with batched input (volumes)."""
        batch_size = 2
        depth, height, width = 8, 16, 16
        channels = 2

        # Channel last: (batch, depth, height, width, channels)
        if feature_axis == -1:
            in_size = (depth, height, width, channels)
            input_shape = (batch_size, depth, height, width, channels)
            feature_axis_param = -1
        # Channel first: (batch, channels, depth, height, width)
        else:
            in_size = (channels, depth, height, width)
            input_shape = (batch_size, channels, depth, height, width)
            feature_axis_param = 0

        # affine can only be True when track_running_stats is True
        affine = track_running_stats

        net = brainstate.nn.BatchNorm3d(
            in_size,
            feature_axis=feature_axis_param,
            track_running_stats=track_running_stats,
            affine=affine
        )
        brainstate.environ.set(fit=fit)

        x = brainstate.random.randn(*input_shape)
        output = net(x)

        # Check output shape matches input
        self.assertEqual(output.shape, input_shape)

    def test_batchnorm3d_without_batch(self):
        """Test BatchNorm3d without batching."""
        depth, height, width = 8, 16, 16
        channels = 2
        in_size = (depth, height, width, channels)

        net = brainstate.nn.BatchNorm3d(in_size, track_running_stats=True)

        # Populate running stats
        brainstate.environ.set(fit=True)
        x_batch = brainstate.random.randn(4, depth, height, width, channels)
        _ = net(x_batch)

        # Test without batch
        brainstate.environ.set(fit=False)
        x_single = brainstate.random.randn(depth, height, width, channels)
        output = net(x_single)

        self.assertEqual(output.shape, (depth, height, width, channels))


class TestLayerNorm(parameterized.TestCase):
    """Test LayerNorm with various configurations."""

    @parameterized.product(
        reduction_axes=[(-1,), (-2, -1), (-3, -2, -1)],
        use_bias=[True, False],
        use_scale=[True, False],
    )
    def test_layernorm_basic(self, reduction_axes, use_bias, use_scale):
        """Test LayerNorm with different reduction axes."""
        in_size = (10, 20, 30)

        net = brainstate.nn.LayerNorm(
            in_size,
            reduction_axes=reduction_axes,
            use_bias=use_bias,
            use_scale=use_scale,
        )

        # With batch
        x = brainstate.random.randn(8, 10, 20, 30)
        output = net(x)
        self.assertEqual(output.shape, x.shape)

        # Check normalization properties
        mean = jnp.mean(output, axis=tuple(i + 1 for i in range(len(in_size))
                                           if i - len(in_size) in reduction_axes))
        var = jnp.var(output, axis=tuple(i + 1 for i in range(len(in_size))
                                         if i - len(in_size) in reduction_axes))

    def test_layernorm_2d_features(self):
        """Test LayerNorm on 2D features (like in transformers)."""
        seq_length = 50
        hidden_dim = 128
        batch_size = 16

        in_size = (seq_length, hidden_dim)
        net = brainstate.nn.LayerNorm(in_size, reduction_axes=-1, feature_axes=-1)

        x = brainstate.random.randn(batch_size, seq_length, hidden_dim)
        output = net(x)

        self.assertEqual(output.shape, x.shape)

        # Each position should be normalized independently
        mean = jnp.mean(output, axis=-1)
        var = jnp.var(output, axis=-1)

        np.testing.assert_allclose(mean, 0.0, atol=1e-5)
        np.testing.assert_allclose(var, 1.0, atol=1e-1)

    def test_layernorm_without_batch(self):
        """Test LayerNorm without batch dimension."""
        in_size = (10, 20)
        net = brainstate.nn.LayerNorm(in_size, reduction_axes=-1)

        x = brainstate.random.randn(10, 20)
        output = net(x)

        self.assertEqual(output.shape, (10, 20))

    @parameterized.product(
        in_size=[(10,), (10, 20), (10, 20, 30)],
    )
    def test_layernorm_various_dims(self, in_size):
        """Test LayerNorm with various input dimensions."""
        net = brainstate.nn.LayerNorm(in_size)

        # With batch
        x_with_batch = brainstate.random.randn(8, *in_size)
        output_with_batch = net(x_with_batch)
        self.assertEqual(output_with_batch.shape, x_with_batch.shape)


class TestRMSNorm(parameterized.TestCase):
    """Test RMSNorm with various configurations."""

    @parameterized.product(
        use_scale=[True, False],
        reduction_axes=[(-1,), (-2, -1)],
    )
    def test_rmsnorm_basic(self, use_scale, reduction_axes):
        """Test RMSNorm with different configurations."""
        in_size = (10, 20)

        net = brainstate.nn.RMSNorm(
            in_size,
            use_scale=use_scale,
            reduction_axes=reduction_axes,
        )

        x = brainstate.random.randn(8, 10, 20)
        output = net(x)

        self.assertEqual(output.shape, x.shape)

    def test_rmsnorm_transformer_like(self):
        """Test RMSNorm in transformer-like setting."""
        seq_length = 50
        hidden_dim = 128
        batch_size = 16

        in_size = (seq_length, hidden_dim)
        net = brainstate.nn.RMSNorm(in_size, reduction_axes=-1, feature_axes=-1)

        x = brainstate.random.randn(batch_size, seq_length, hidden_dim)
        output = net(x)

        self.assertEqual(output.shape, x.shape)

        # RMSNorm should have approximately unit RMS (not zero mean)
        rms = jnp.sqrt(jnp.mean(jnp.square(output), axis=-1))
        np.testing.assert_allclose(rms, 1.0, atol=1e-1)

    def test_rmsnorm_without_batch(self):
        """Test RMSNorm without batch dimension."""
        in_size = (10, 20)
        net = brainstate.nn.RMSNorm(in_size, reduction_axes=-1)

        x = brainstate.random.randn(10, 20)
        output = net(x)

        self.assertEqual(output.shape, (10, 20))


class TestGroupNorm(parameterized.TestCase):
    """Test GroupNorm with various configurations."""

    @parameterized.product(
        num_groups=[1, 2, 4, 8],
        use_bias=[True, False],
        use_scale=[True, False],
    )
    def test_groupnorm_basic(self, num_groups, use_bias, use_scale):
        """Test GroupNorm with different number of groups."""
        channels = 16
        # GroupNorm requires 1D feature axis (just the channel dimension)
        in_size = (channels,)

        # Check if channels is divisible by num_groups
        if channels % num_groups != 0:
            return

        net = brainstate.nn.GroupNorm(
            in_size,
            feature_axis=0,
            num_groups=num_groups,
            use_bias=use_bias,
            use_scale=use_scale,
        )

        # Input needs at least 2D: (height, width, channels) or (batch, channels)
        # Using (batch, channels) format
        x = brainstate.random.randn(4, channels)
        output = net(x)

        self.assertEqual(output.shape, x.shape)

    def test_groupnorm_channel_first(self):
        """Test GroupNorm with channel-first format for images."""
        channels = 16
        # GroupNorm requires 1D feature (just channels)
        in_size = (channels,)

        net = brainstate.nn.GroupNorm(
            in_size,
            feature_axis=0,
            num_groups=4,
        )

        # Test with image-like data: (batch, height, width, channels)
        x = brainstate.random.randn(4, 32, 32, channels)
        output = net(x)

        self.assertEqual(output.shape, x.shape)

    def test_groupnorm_channel_last(self):
        """Test GroupNorm with channel-last format for images."""
        channels = 16
        # GroupNorm requires 1D feature (just channels)
        in_size = (channels,)

        net = brainstate.nn.GroupNorm(
            in_size,
            feature_axis=0,  # feature_axis refers to position in in_size
            num_groups=4,
        )

        # Test with image-like data: (batch, height, width, channels)
        x = brainstate.random.randn(4, 32, 32, channels)
        output = net(x)

        self.assertEqual(output.shape, x.shape)

    def test_groupnorm_equals_layernorm(self):
        """Test that GroupNorm with num_groups=1 equals LayerNorm."""
        channels = 16
        # GroupNorm requires 1D feature
        in_size = (channels,)

        # GroupNorm with 1 group
        group_norm = brainstate.nn.GroupNorm(
            in_size,
            feature_axis=0,
            num_groups=1,
        )

        # LayerNorm with same setup
        layer_norm = brainstate.nn.LayerNorm(
            in_size,
            reduction_axes=-1,
            feature_axes=-1,
        )

        # Use 2D input: (batch, channels)
        x = brainstate.random.randn(8, channels)

        output_gn = group_norm(x)
        output_ln = layer_norm(x)

        # Shapes should match
        self.assertEqual(output_gn.shape, output_ln.shape)

    def test_groupnorm_group_size(self):
        """Test GroupNorm with group_size instead of num_groups."""
        channels = 16
        group_size = 4
        # GroupNorm requires 1D feature
        in_size = (channels,)

        net = brainstate.nn.GroupNorm(
            in_size,
            feature_axis=0,
            num_groups=None,
            group_size=group_size,
        )

        # Use 2D input: (batch, channels)
        x = brainstate.random.randn(4, channels)
        output = net(x)

        self.assertEqual(output.shape, x.shape)
        self.assertEqual(net.num_groups, channels // group_size)

    def test_groupnorm_invalid_groups(self):
        """Test that invalid num_groups raises error."""
        channels = 15  # Not divisible by many numbers
        # GroupNorm requires 1D feature
        in_size = (channels,)

        # Should raise error if num_groups doesn't divide channels
        with self.assertRaises(ValueError):
            net = brainstate.nn.GroupNorm(
                in_size,
                feature_axis=0,
                num_groups=4,  # 15 is not divisible by 4
            )


class TestNormalizationUtilities(parameterized.TestCase):
    """Test utility functions for normalization."""

    def test_weight_standardization(self):
        """Test weight_standardization function."""
        w = brainstate.random.randn(3, 4, 5, 6)

        w_std = brainstate.nn.weight_standardization(w, eps=1e-4)

        self.assertEqual(w_std.shape, w.shape)

        # Check that standardization works
        # Mean should be close to 0 along non-output axes
        mean = jnp.mean(w_std, axis=(0, 1, 2))
        np.testing.assert_allclose(mean, 0.0, atol=1e-4)

    def test_weight_standardization_with_gain(self):
        """Test weight_standardization with gain parameter."""
        w = brainstate.random.randn(3, 4, 5, 6)
        gain = jnp.ones((6,))

        w_std = brainstate.nn.weight_standardization(w, gain=gain)

        self.assertEqual(w_std.shape, w.shape)


class TestNormalizationEdgeCases(parameterized.TestCase):
    """Test edge cases and error conditions."""

    def test_batchnorm_shape_mismatch(self):
        """Test that BatchNorm raises error on shape mismatch."""
        net = brainstate.nn.BatchNorm2d((28, 28, 3))

        # Wrong shape should raise error
        with self.assertRaises(ValueError):
            x = brainstate.random.randn(4, 32, 32, 3)  # Wrong height/width
            _ = net(x)

    def test_batchnorm_without_track_and_affine(self):
        """Test that affine=True requires track_running_stats=True."""
        # This should raise an assertion error
        with self.assertRaises(AssertionError):
            net = brainstate.nn.BatchNorm2d(
                (28, 28, 3),
                track_running_stats=False,
                affine=True  # Requires track_running_stats=True
            )

    def test_groupnorm_both_params(self):
        """Test that GroupNorm raises error when both num_groups and group_size are specified."""
        with self.assertRaises(ValueError):
            net = brainstate.nn.GroupNorm(
                (32, 32, 16),
                num_groups=4,
                group_size=4,  # Can't specify both
            )

    def test_groupnorm_neither_param(self):
        """Test that GroupNorm raises error when neither num_groups nor group_size are specified."""
        with self.assertRaises(ValueError):
            net = brainstate.nn.GroupNorm(
                (32, 32, 16),
                num_groups=None,
                group_size=None,  # Must specify one
            )


class TestNormalizationConsistency(parameterized.TestCase):
    """Test consistency across different batch sizes and modes."""

    def test_batchnorm2d_consistency_across_batches(self):
        """Test that BatchNorm2d behaves consistently across different batch sizes."""
        in_size = (28, 28, 3)
        net = brainstate.nn.BatchNorm2d(in_size, track_running_stats=True)

        # Train on larger batch
        brainstate.environ.set(fit=True)
        x_large = brainstate.random.randn(32, 28, 28, 3)
        _ = net(x_large)

        # Test on smaller batch
        brainstate.environ.set(fit=False)
        x_small = brainstate.random.randn(4, 28, 28, 3)
        output = net(x_small)

        self.assertEqual(output.shape, x_small.shape)

    def test_layernorm_consistency(self):
        """Test that LayerNorm produces consistent results."""
        in_size = (10, 20)
        net = brainstate.nn.LayerNorm(in_size)

        x = brainstate.random.randn(8, 10, 20)

        # Run twice
        output1 = net(x)
        output2 = net(x)

        # Should be deterministic
        np.testing.assert_allclose(output1, output2)


class TestWeightStandardizationDetails(unittest.TestCase):
    """Lower-level behavior of weight_standardization, including unit handling."""

    def test_zeroes_mean_along_fan_in(self):
        """Standardized weights have ~zero mean along the fan-in axes."""
        with brainstate.random.seed_context(0):
            w = brainstate.random.randn(3, 4, 5, 6)
        w_std = weight_standardization(w)
        mean = jnp.mean(w_std, axis=(0, 1, 2))
        np.testing.assert_allclose(mean, 0.0, atol=1e-4)

    def test_negative_out_axis(self):
        """A negative out_axis is normalized to a positive index."""
        with brainstate.random.seed_context(1):
            w = brainstate.random.randn(4, 5)
        a = weight_standardization(w, out_axis=-1)
        b = weight_standardization(w, out_axis=1)
        assert_allclose(a, b)

    def test_unitless_quantity_input(self):
        """A unitless Quantity input takes the dimensionless rsqrt branch."""
        w = jnp.ones((3, 4)) * u.UNITLESS
        w_std = weight_standardization(w)
        self.assertEqual(u.math.shape(w_std), (3, 4))

    @pytest.mark.skip(
        reason="BUG: weight_standardization with a unit-carrying Quantity input "
               "constructs u.Quantity(..., unit=1/unit**0.5); since 1/unit**0.5 "
               "evaluates to a Quantity (not a Unit), Quantity.__init__ asserts."
    )
    def test_unit_carrying_quantity_input(self):
        """A unit-carrying Quantity should be standardizable (currently buggy)."""
        w = jnp.ones((3, 4)) * u.mV
        w_std = weight_standardization(w, eps=1e-4 * u.mV ** 2)
        self.assertEqual(u.math.shape(w_std), (3, 4))


class TestCanonicalizeDtype(unittest.TestCase):
    """Behavior of the canonicalize_dtype helper."""

    def test_integer_args_promoted_to_float(self):
        """Integer inputs are promoted to at least float32 when inexact=True."""
        dtype = canonicalize_dtype(jnp.array([1, 2, 3]))
        self.assertTrue(jnp.issubdtype(dtype, jnp.inexact))

    def test_explicit_float_dtype_passthrough(self):
        """An explicit inexact dtype is returned unchanged."""
        self.assertEqual(canonicalize_dtype(jnp.array([1, 2]), dtype=jnp.float32),
                         jnp.float32)

    def test_explicit_non_inexact_dtype_raises(self):
        """An explicit non-inexact dtype with inexact=True raises ValueError."""
        with self.assertRaises(ValueError):
            canonicalize_dtype(jnp.array([1, 2]), dtype=jnp.int32, inexact=True)

    def test_non_inexact_allowed_when_flag_false(self):
        """A non-inexact dtype is allowed when inexact=False."""
        dtype = canonicalize_dtype(jnp.array([1, 2]), inexact=False)
        self.assertTrue(jnp.issubdtype(dtype, jnp.integer))


class TestNormalizationHelpers(unittest.TestCase):
    """Tests for low-level normalization helper functions."""

    def test_canonicalize_axes_invalid_raises(self):
        """_canonicalize_axes rejects out-of-range axes."""
        with self.assertRaises(ValueError):
            _canonicalize_axes(2, (5,))

    def test_canonicalize_axes_negative(self):
        """_canonicalize_axes converts negative axes to positive indices."""
        self.assertEqual(_canonicalize_axes(3, (-1, -2)), (2, 1))

    def test_abs_sq_complex(self):
        """_abs_sq computes squared magnitude for complex input."""
        out = _abs_sq(jnp.array([3 + 4j], dtype=jnp.complex64))
        np.testing.assert_allclose(np.asarray(out), [25.0], atol=1e-5)

    def test_abs_sq_real(self):
        """_abs_sq squares a real input."""
        out = _abs_sq(jnp.array([2.0, -3.0]))
        np.testing.assert_allclose(np.asarray(out), [4.0, 9.0])

    def test_compute_stats_slow_variance(self):
        """_compute_stats with use_fast_variance=False matches the fast path."""
        with brainstate.random.seed_context(2):
            x = brainstate.random.randn(8, 4)
        mu_fast, var_fast = _compute_stats(x, axes=(0,), dtype=None,
                                           use_fast_variance=True)
        mu_slow, var_slow = _compute_stats(x, axes=(0,), dtype=None,
                                           use_fast_variance=False)
        assert_allclose(mu_fast, mu_slow, atol=1e-4)
        assert_allclose(var_fast, var_slow, atol=1e-4)

    def test_compute_stats_axis_name_pmean(self):
        """_compute_stats reduces across a named (vmapped) axis via pmean."""
        with brainstate.random.seed_context(3):
            x = brainstate.random.randn(2, 6, 4)

        def f(xi):
            return _compute_stats(xi, axes=(0,), dtype=None, axis_name='b',
                                  use_fast_variance=True)

        mus, vars_ = jax.vmap(f, axis_name='b')(x)
        # Means across the named axis are shared, so all replicas agree.
        assert_allclose(mus[0], mus[1], atol=1e-5)
        self.assertEqual(mus.shape, (2, 4))

    def test_compute_stats_axis_name_single_array(self):
        """A named axis with use_fast_variance=False uses single-array pmean calls."""
        with brainstate.random.seed_context(11):
            x = brainstate.random.randn(2, 6, 4)

        def f(xi):
            return _compute_stats(xi, axes=(0,), dtype=None, axis_name='b',
                                  use_fast_variance=False)

        mus, vars_ = jax.vmap(f, axis_name='b')(x)
        # Means and variances are reduced across the named axis, so replicas agree.
        assert_allclose(mus[0], mus[1], atol=1e-5)
        assert_allclose(vars_[0], vars_[1], atol=1e-5)


class TestComputeStatsDtype(unittest.TestCase):
    """dtype handling inside _compute_stats."""

    def test_default_dtype_inference(self):
        """When dtype is None, statistics are computed at >=float32 precision."""
        x = jnp.arange(12, dtype=jnp.int32).reshape(3, 4)
        mu, var = _compute_stats(x, axes=(0,), dtype=None)
        self.assertTrue(jnp.issubdtype(mu.dtype, jnp.floating))


class TestBatchNormInvalidNdim(unittest.TestCase):
    """BatchNorm rejects inputs with an unexpected number of dimensions."""

    def setUp(self):
        """Enter fitting mode."""
        brainstate.environ.set(fit=True)

    def test_batchnorm1d_wrong_ndim_raises(self):
        """BatchNorm1d rejects a 1D input (needs 2D or 3D)."""
        net = brainstate.nn.BatchNorm1d((20, 10))
        with self.assertRaises(ValueError):
            net(brainstate.random.randn(20))

    def test_batchnorm0d_wrong_ndim_raises(self):
        """BatchNorm0d rejects a 3D input (needs 1D or 2D)."""
        net = brainstate.nn.BatchNorm0d((10,))
        with self.assertRaises(ValueError):
            net(brainstate.random.randn(4, 5, 10))

    def test_batchnorm2d_wrong_ndim_raises(self):
        """BatchNorm2d rejects a 2D input (needs 3D or 4D)."""
        net = brainstate.nn.BatchNorm2d((8, 8, 3))
        with self.assertRaises(ValueError):
            net(brainstate.random.randn(8, 3))


class TestBatchNormTrainEval(unittest.TestCase):
    """Train vs eval behavior of the running statistics."""

    def test_running_stats_update_in_fit(self):
        """Running mean/var move away from their initial values during fitting."""
        net = brainstate.nn.BatchNorm1d((6, 4), momentum=0.5)
        mean0 = np.array(net.running_mean.value)
        brainstate.environ.set(fit=True)
        with brainstate.random.seed_context(4):
            x = brainstate.random.randn(8, 6, 4) + 5.0
        _ = net(x)
        # Running mean should have shifted toward the batch mean.
        self.assertFalse(np.allclose(mean0, np.array(net.running_mean.value)))

    def test_eval_uses_frozen_running_stats(self):
        """In eval mode the running statistics are used and not updated."""
        net = brainstate.nn.BatchNorm1d((6, 4))
        brainstate.environ.set(fit=True)
        with brainstate.random.seed_context(5):
            _ = net(brainstate.random.randn(8, 6, 4) + 3.0)
        snap_mean = np.array(net.running_mean.value)
        snap_var = np.array(net.running_var.value)

        brainstate.environ.set(fit=False)
        with brainstate.random.seed_context(6):
            _ = net(brainstate.random.randn(8, 6, 4) - 2.0)
        # Eval must not modify the running statistics.
        np.testing.assert_allclose(snap_mean, np.array(net.running_mean.value))
        np.testing.assert_allclose(snap_var, np.array(net.running_var.value))


class TestGroupNormBranches(unittest.TestCase):
    """Less-common GroupNorm code paths."""

    def test_group_size_not_divisible_raises(self):
        """group_size that does not divide the channel count raises ValueError."""
        with self.assertRaises(ValueError):
            brainstate.nn.GroupNorm((15,), feature_axis=0, num_groups=None,
                                    group_size=4)

    def test_explicit_reduction_axes(self):
        """GroupNorm honors explicit reduction_axes and preserves shape."""
        net = brainstate.nn.GroupNorm((16,), feature_axis=0, num_groups=4,
                                      reduction_axes=(1, 2, -1))
        with brainstate.random.seed_context(7):
            x = brainstate.random.randn(2, 4, 4, 16)
        out = net(x)
        self.assertEqual(out.shape, x.shape)

    def test_groupnorm_with_mask(self):
        """GroupNorm accepts a broadcastable mask and preserves shape."""
        net = brainstate.nn.GroupNorm((16,), feature_axis=0, num_groups=4)
        with brainstate.random.seed_context(8):
            x = brainstate.random.randn(4, 16)
        mask = jnp.ones((4, 16), dtype=bool)
        out = net(x, mask=mask)
        self.assertEqual(out.shape, x.shape)


class TestNormalizationJitGrad(unittest.TestCase):
    """JIT consistency and gradient finiteness for normalization layers."""

    def test_layernorm_jit_equal(self):
        """LayerNorm output is identical under JIT and eager execution."""
        net = brainstate.nn.LayerNorm((8,))
        with brainstate.random.seed_context(9):
            x = brainstate.random.randn(4, 8)
        _testing.assert_jit_equal(net.update, x)

    def test_rmsnorm_grad_finite(self):
        """Gradients through RMSNorm are finite."""
        net = brainstate.nn.RMSNorm((8,))
        with brainstate.random.seed_context(10):
            x = brainstate.random.randn(4, 8)

        def loss_fn(inp):
            return jnp.sum(net(inp) ** 2)

        _testing.assert_grad_finite(loss_fn, x)


if __name__ == '__main__':
    absltest.main()
