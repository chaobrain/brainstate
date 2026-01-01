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

"""Tests for regularization classes."""

import unittest

import brainstate
import jax.numpy as jnp
import numpy as np

from brainstate.nn import (
    ChainedReg,
    GaussianReg,
    L1Reg,
    L2Reg,
    ElasticNetReg,
    HuberReg,
    GroupLassoReg,
    TotalVariationReg,
    MaxNormReg,
    EntropyReg,
    OrthogonalReg,
    SpectralNormReg,
    # Prior distribution-based regularizations
    StudentTReg,
    CauchyReg,
    UniformReg,
    LogNormalReg,
    ExponentialReg,
    GammaReg,
    BetaReg,
    HorseshoeReg,
    InverseGammaReg,
    LogUniformReg,
    SpikeAndSlabReg,
    DirichletReg,
)


class TestL2Reg(unittest.TestCase):
    """Tests for L2 (Ridge) regularization."""

    def test_basic_loss(self):
        """Test L2 loss computation."""
        reg = L2Reg(weight=1.0)
        value = jnp.array([1.0, 2.0, 3.0])
        loss = reg.loss(value)
        # L2 loss = 1.0 * (1^2 + 2^2 + 3^2) = 14.0
        np.testing.assert_allclose(loss, 14.0, rtol=1e-5)

    def test_weighted_loss(self):
        """Test L2 loss with different weight."""
        reg = L2Reg(weight=0.5)
        value = jnp.array([2.0, 2.0])
        loss = reg.loss(value)
        # L2 loss = 0.5 * (4 + 4) = 4.0
        np.testing.assert_allclose(loss, 4.0, rtol=1e-5)

    def test_sample_init_shape(self):
        """Test sample_init returns correct shape."""
        reg = L2Reg(weight=1.0)
        sample = reg.sample_init((3, 4))
        self.assertEqual(sample.shape, (3, 4))

    def test_sample_init_int_shape(self):
        """Test sample_init with int shape."""
        reg = L2Reg(weight=1.0)
        sample = reg.sample_init(5)
        self.assertEqual(sample.shape, (5,))

    def test_reset_value(self):
        """Test reset_value returns zero."""
        reg = L2Reg(weight=1.0)
        reset = reg.reset_value()
        self.assertEqual(reset, 0.0)

    def test_fit_hyper_false(self):
        """Test that weight is not trainable by default."""
        reg = L2Reg(weight=1.0, fit_hyper=False)
        self.assertFalse(reg.fit_hyper)
        self.assertNotIsInstance(reg.weight, brainstate.State)

    def test_fit_hyper_true(self):
        """Test that fit_hyper=True is stored correctly."""
        reg = L2Reg(weight=1.0, fit_hyper=True)
        self.assertTrue(reg.fit_hyper)


class TestL1Reg(unittest.TestCase):
    """Tests for L1 (Lasso) regularization."""

    def test_basic_loss(self):
        """Test L1 loss computation."""
        reg = L1Reg(weight=1.0)
        value = jnp.array([1.0, -2.0, 3.0])
        loss = reg.loss(value)
        # L1 loss = 1.0 * (|1| + |-2| + |3|) = 6.0
        np.testing.assert_allclose(loss, 6.0, rtol=1e-5)

    def test_weighted_loss(self):
        """Test L1 loss with different weight."""
        reg = L1Reg(weight=0.5)
        value = jnp.array([2.0, -2.0])
        loss = reg.loss(value)
        # L1 loss = 0.5 * (2 + 2) = 2.0
        np.testing.assert_allclose(loss, 2.0, rtol=1e-5)

    def test_sample_init_shape(self):
        """Test sample_init returns correct shape."""
        reg = L1Reg(weight=1.0)
        sample = reg.sample_init((3, 4))
        self.assertEqual(sample.shape, (3, 4))

    def test_sample_init_int_shape(self):
        """Test sample_init with int shape."""
        reg = L1Reg(weight=1.0)
        sample = reg.sample_init(5)
        self.assertEqual(sample.shape, (5,))

    def test_reset_value(self):
        """Test reset_value returns zero."""
        reg = L1Reg(weight=1.0)
        reset = reg.reset_value()
        self.assertEqual(reset, 0.0)

    def test_fit_hyper_false(self):
        """Test that weight is not trainable by default."""
        reg = L1Reg(weight=1.0, fit_hyper=False)
        self.assertFalse(reg.fit_hyper)
        self.assertNotIsInstance(reg.weight, brainstate.State)

    def test_fit_hyper_true(self):
        """Test that fit_hyper=True is stored correctly."""
        reg = L1Reg(weight=1.0, fit_hyper=True)
        self.assertTrue(reg.fit_hyper)


class TestGaussianReg(unittest.TestCase):
    """Tests for Gaussian prior regularization."""

    def test_loss_at_mean(self):
        """Test that loss is minimized at mean."""
        reg = GaussianReg(mean=0.0, std=1.0)
        loss_at_mean = reg.loss(jnp.array([0.0]))
        loss_away = reg.loss(jnp.array([1.0]))
        self.assertLess(float(loss_at_mean), float(loss_away))

    def test_loss_increases_with_distance(self):
        """Test that loss increases with distance from mean."""
        reg = GaussianReg(mean=0.0, std=1.0)
        loss_1 = reg.loss(jnp.array([1.0]))
        loss_2 = reg.loss(jnp.array([2.0]))
        self.assertLess(float(loss_1), float(loss_2))

    def test_sample_init_shape(self):
        """Test sample_init returns correct shape."""
        reg = GaussianReg(mean=0.0, std=1.0)
        sample = reg.sample_init((3, 4))
        self.assertEqual(sample.shape, (3, 4))

    def test_sample_init_int_shape(self):
        """Test sample_init with int shape."""
        reg = GaussianReg(mean=0.0, std=1.0)
        sample = reg.sample_init(5)
        self.assertEqual(sample.shape, (5,))

    def test_reset_value(self):
        """Test reset_value returns mean."""
        reg = GaussianReg(mean=2.5, std=1.0)
        reset = reg.reset_value()
        np.testing.assert_allclose(reset, 2.5)

    def test_fit_hyper_false(self):
        """Test that hyperparams are not trainable by default."""
        reg = GaussianReg(mean=0.0, std=1.0, fit_hyper=False)
        self.assertFalse(reg.fit_hyper)
        self.assertNotIsInstance(reg.mean, brainstate.State)
        self.assertNotIsInstance(reg.precision, brainstate.State)
        self.assertNotIsInstance(reg.weight, brainstate.State)

    def test_fit_hyper_true(self):
        """Test that fit_hyper=True is stored correctly."""
        reg = GaussianReg(mean=0.0, std=1.0, fit_hyper=True)
        self.assertTrue(reg.fit_hyper)

    def test_weight_scales_loss(self):
        """Test that weight parameter scales the loss."""
        reg1 = GaussianReg(mean=0.0, std=1.0, weight=1.0)
        reg2 = GaussianReg(mean=0.0, std=1.0, weight=2.0)
        value = jnp.array([1.0])
        loss1 = reg1.loss(value)
        loss2 = reg2.loss(value)
        np.testing.assert_allclose(float(loss2), float(loss1) * 2.0, rtol=1e-5)

    def test_array_mean_std(self):
        """Test with array-valued mean and std."""
        reg = GaussianReg(mean=jnp.array([0.0, 1.0]), std=jnp.array([1.0, 2.0]))
        reset = reg.reset_value()
        np.testing.assert_allclose(reset, jnp.array([0.0, 1.0]))


class TestRegularizationInheritance(unittest.TestCase):
    """Tests for regularization inheritance."""

    def test_l2_inherits_from_module(self):
        """Test L2Reg inherits from brainstate.nn.Module."""
        reg = L2Reg(weight=1.0)
        self.assertIsInstance(reg, brainstate.nn.Module)

    def test_l1_inherits_from_module(self):
        """Test L1Reg inherits from brainstate.nn.Module."""
        reg = L1Reg(weight=1.0)
        self.assertIsInstance(reg, brainstate.nn.Module)

    def test_gaussian_inherits_from_module(self):
        """Test GaussianReg inherits from brainstate.nn.Module."""
        reg = GaussianReg(mean=0.0, std=1.0)
        self.assertIsInstance(reg, brainstate.nn.Module)


class TestElasticNetReg(unittest.TestCase):
    """Tests for Elastic Net regularization."""

    def test_basic_loss(self):
        """Test Elastic Net loss computation."""
        reg = ElasticNetReg(l1_weight=1.0, l2_weight=1.0, alpha=0.5)
        value = jnp.array([1.0, -2.0])
        loss = reg.loss(value)
        # alpha * L1 + (1-alpha) * L2
        # 0.5 * 1.0 * (1 + 2) + 0.5 * 1.0 * (1 + 4) = 0.5 * 3 + 0.5 * 5 = 4.0
        np.testing.assert_allclose(loss, 4.0, rtol=1e-5)

    def test_alpha_zero_equals_l2(self):
        """Test that alpha=0 gives pure L2."""
        reg = ElasticNetReg(l1_weight=1.0, l2_weight=1.0, alpha=0.0)
        value = jnp.array([1.0, 2.0])
        loss = reg.loss(value)
        # Pure L2: 1.0 * (1 + 4) = 5.0
        np.testing.assert_allclose(loss, 5.0, rtol=1e-5)

    def test_alpha_one_equals_l1(self):
        """Test that alpha=1 gives pure L1."""
        reg = ElasticNetReg(l1_weight=1.0, l2_weight=1.0, alpha=1.0)
        value = jnp.array([1.0, -2.0])
        loss = reg.loss(value)
        # Pure L1: 1.0 * (1 + 2) = 3.0
        np.testing.assert_allclose(loss, 3.0, rtol=1e-5)

    def test_sample_init_shape(self):
        """Test sample_init returns correct shape."""
        reg = ElasticNetReg(l1_weight=1.0, l2_weight=1.0, alpha=0.5)
        sample = reg.sample_init((3, 4))
        self.assertEqual(sample.shape, (3, 4))

    def test_reset_value(self):
        """Test reset_value returns zero."""
        reg = ElasticNetReg(l1_weight=1.0, l2_weight=1.0, alpha=0.5)
        reset = reg.reset_value()
        self.assertEqual(reset, 0.0)

    def test_fit_hyper_true(self):
        """Test that fit_hyper=True is stored correctly."""
        reg = ElasticNetReg(l1_weight=1.0, l2_weight=1.0, alpha=0.5, fit_hyper=True)
        self.assertTrue(reg.fit_hyper)


class TestHuberReg(unittest.TestCase):
    """Tests for Huber regularization."""

    def test_small_values_quadratic(self):
        """Test that small values give quadratic (L2-like) loss."""
        reg = HuberReg(weight=1.0, delta=1.0)
        value = jnp.array([0.5])
        loss = reg.loss(value)
        # For |x| <= delta: 0.5 * x^2 = 0.5 * 0.25 = 0.125
        np.testing.assert_allclose(loss, 0.125, rtol=1e-5)

    def test_large_values_linear(self):
        """Test that large values give linear (L1-like) loss."""
        reg = HuberReg(weight=1.0, delta=1.0)
        value = jnp.array([2.0])
        loss = reg.loss(value)
        # For |x| > delta: delta * (|x| - 0.5*delta) = 1.0 * (2.0 - 0.5) = 1.5
        np.testing.assert_allclose(loss, 1.5, rtol=1e-5)

    def test_weighted_loss(self):
        """Test Huber loss with different weight."""
        reg = HuberReg(weight=2.0, delta=1.0)
        value = jnp.array([0.5])
        loss = reg.loss(value)
        # weight * 0.5 * x^2 = 2.0 * 0.125 = 0.25
        np.testing.assert_allclose(loss, 0.25, rtol=1e-5)

    def test_sample_init_shape(self):
        """Test sample_init returns correct shape."""
        reg = HuberReg(weight=1.0, delta=1.0)
        sample = reg.sample_init((3, 4))
        self.assertEqual(sample.shape, (3, 4))

    def test_reset_value(self):
        """Test reset_value returns zero."""
        reg = HuberReg(weight=1.0, delta=1.0)
        reset = reg.reset_value()
        self.assertEqual(reset, 0.0)

    def test_fit_hyper_true(self):
        """Test that fit_hyper=True is stored correctly."""
        reg = HuberReg(weight=1.0, delta=1.0, fit_hyper=True)
        self.assertTrue(reg.fit_hyper)


class TestGroupLassoReg(unittest.TestCase):
    """Tests for Group Lasso regularization."""

    def test_basic_loss(self):
        """Test Group Lasso loss computation."""
        reg = GroupLassoReg(weight=1.0, group_size=2)
        value = jnp.array([3.0, 4.0, 0.0, 0.0])
        loss = reg.loss(value)
        # Group 1: sqrt(9 + 16) = 5, Group 2: sqrt(0 + 0) ~ 0
        # Total: 5 + epsilon
        np.testing.assert_allclose(loss, 5.0, rtol=1e-3)

    def test_group_size_one_like_l1(self):
        """Test that group_size=1 behaves like L1 (approximately)."""
        reg = GroupLassoReg(weight=1.0, group_size=1)
        value = jnp.array([1.0, 2.0, 3.0])
        loss = reg.loss(value)
        # With group_size=1, each element is its own group
        # sqrt(1) + sqrt(4) + sqrt(9) = 1 + 2 + 3 = 6 (approximately, with epsilon)
        np.testing.assert_allclose(loss, 6.0, rtol=1e-3)

    def test_sample_init_shape(self):
        """Test sample_init returns correct shape."""
        reg = GroupLassoReg(weight=1.0, group_size=2)
        sample = reg.sample_init((4, 4))
        self.assertEqual(sample.shape, (4, 4))

    def test_reset_value(self):
        """Test reset_value returns zero."""
        reg = GroupLassoReg(weight=1.0, group_size=2)
        reset = reg.reset_value()
        self.assertEqual(reset, 0.0)

    def test_fit_hyper_true(self):
        """Test that fit_hyper=True is stored correctly."""
        reg = GroupLassoReg(weight=1.0, group_size=2, fit_hyper=True)
        self.assertTrue(reg.fit_hyper)


class TestTotalVariationReg(unittest.TestCase):
    """Tests for Total Variation regularization."""

    def test_constant_signal_zero_loss(self):
        """Test that constant signal has zero TV loss."""
        reg = TotalVariationReg(weight=1.0, order=1)
        value = jnp.array([1.0, 1.0, 1.0, 1.0])
        loss = reg.loss(value)
        np.testing.assert_allclose(loss, 0.0, atol=1e-6)

    def test_first_order_loss(self):
        """Test first-order TV loss computation."""
        reg = TotalVariationReg(weight=1.0, order=1)
        value = jnp.array([0.0, 1.0, 0.0])
        loss = reg.loss(value)
        # |1-0| + |0-1| = 1 + 1 = 2
        np.testing.assert_allclose(loss, 2.0, rtol=1e-5)

    def test_second_order_loss(self):
        """Test second-order TV loss computation."""
        reg = TotalVariationReg(weight=1.0, order=2)
        value = jnp.array([0.0, 1.0, 0.0])
        loss = reg.loss(value)
        # |0 - 2*1 + 0| = 2
        np.testing.assert_allclose(loss, 2.0, rtol=1e-5)

    def test_sample_init_shape(self):
        """Test sample_init returns correct shape."""
        reg = TotalVariationReg(weight=1.0, order=1)
        sample = reg.sample_init((10,))
        self.assertEqual(sample.shape, (10,))

    def test_reset_value(self):
        """Test reset_value returns zero."""
        reg = TotalVariationReg(weight=1.0, order=1)
        reset = reg.reset_value()
        self.assertEqual(reset, 0.0)

    def test_fit_hyper_true(self):
        """Test that fit_hyper=True is stored correctly."""
        reg = TotalVariationReg(weight=1.0, order=1, fit_hyper=True)
        self.assertTrue(reg.fit_hyper)


class TestMaxNormReg(unittest.TestCase):
    """Tests for Max Norm regularization."""

    def test_within_norm_zero_loss(self):
        """Test that values within max_value have zero loss."""
        reg = MaxNormReg(weight=1.0, max_value=5.0)
        value = jnp.array([1.0, 2.0, 2.0])  # norm = 3 < 5
        loss = reg.loss(value)
        np.testing.assert_allclose(loss, 0.0, atol=1e-5)

    def test_exceeds_norm_has_loss(self):
        """Test that values exceeding max_value have positive loss."""
        reg = MaxNormReg(weight=1.0, max_value=3.0)
        value = jnp.array([3.0, 4.0])  # norm = 5 > 3
        loss = reg.loss(value)
        # (5 - 3)^2 = 4
        np.testing.assert_allclose(loss, 4.0, rtol=1e-3)

    def test_sample_init_shape(self):
        """Test sample_init returns correct shape."""
        reg = MaxNormReg(weight=1.0, max_value=1.0)
        sample = reg.sample_init((3, 4))
        self.assertEqual(sample.shape, (3, 4))

    def test_sample_init_bounded(self):
        """Test that sample_init produces bounded samples."""
        reg = MaxNormReg(weight=1.0, max_value=2.0)
        sample = reg.sample_init((100,))
        norm = jnp.sqrt(jnp.sum(sample ** 2))
        self.assertLessEqual(float(norm), 2.0 + 1e-5)

    def test_reset_value(self):
        """Test reset_value returns zero."""
        reg = MaxNormReg(weight=1.0, max_value=1.0)
        reset = reg.reset_value()
        self.assertEqual(reset, 0.0)

    def test_fit_hyper_true(self):
        """Test that fit_hyper=True is stored correctly."""
        reg = MaxNormReg(weight=1.0, max_value=1.0, fit_hyper=True)
        self.assertTrue(reg.fit_hyper)


class TestEntropyReg(unittest.TestCase):
    """Tests for Entropy regularization."""

    def test_uniform_high_entropy(self):
        """Test that uniform distribution has high entropy."""
        reg = EntropyReg(weight=1.0, maximize=True)
        uniform = jnp.array([1.0, 1.0, 1.0, 1.0])
        concentrated = jnp.array([10.0, 0.0, 0.0, 0.0])
        loss_uniform = reg.loss(uniform)
        loss_concentrated = reg.loss(concentrated)
        # When maximizing entropy, uniform should have lower loss
        self.assertLess(float(loss_uniform), float(loss_concentrated))

    def test_minimize_entropy(self):
        """Test entropy minimization mode."""
        reg = EntropyReg(weight=1.0, maximize=False)
        uniform = jnp.array([1.0, 1.0, 1.0, 1.0])
        concentrated = jnp.array([10.0, 0.0, 0.0, 0.0])
        loss_uniform = reg.loss(uniform)
        loss_concentrated = reg.loss(concentrated)
        # When minimizing entropy, concentrated should have lower loss
        self.assertLess(float(loss_concentrated), float(loss_uniform))

    def test_sample_init_shape(self):
        """Test sample_init returns correct shape."""
        reg = EntropyReg(weight=1.0, maximize=True)
        sample = reg.sample_init((3, 4))
        self.assertEqual(sample.shape, (3, 4))

    def test_reset_value(self):
        """Test reset_value returns zero."""
        reg = EntropyReg(weight=1.0, maximize=True)
        reset = reg.reset_value()
        self.assertEqual(reset, 0.0)

    def test_fit_hyper_true(self):
        """Test that fit_hyper=True is stored correctly."""
        reg = EntropyReg(weight=1.0, maximize=True, fit_hyper=True)
        self.assertTrue(reg.fit_hyper)


class TestOrthogonalReg(unittest.TestCase):
    """Tests for Orthogonal regularization."""

    def test_orthogonal_matrix_low_loss(self):
        """Test that orthogonal matrix has low loss."""
        reg = OrthogonalReg(weight=1.0)
        # Create orthogonal matrix
        Q = jnp.array([[1.0, 0.0], [0.0, 1.0]])  # Identity is orthogonal
        loss = reg.loss(Q)
        np.testing.assert_allclose(loss, 0.0, atol=1e-5)

    def test_non_orthogonal_has_loss(self):
        """Test that non-orthogonal matrix has positive loss."""
        reg = OrthogonalReg(weight=1.0)
        W = jnp.array([[2.0, 0.0], [0.0, 2.0]])  # W^T W = 4*I, not I
        loss = reg.loss(W)
        # ||W^T W - I||_F^2 = ||(4I - I)||_F^2 = ||3I||_F^2 = 18
        np.testing.assert_allclose(loss, 18.0, rtol=1e-3)

    def test_sample_init_2d_shape(self):
        """Test sample_init returns correct shape for 2D."""
        reg = OrthogonalReg(weight=1.0)
        sample = reg.sample_init((4, 3))
        self.assertEqual(sample.shape, (4, 3))

    def test_reset_value(self):
        """Test reset_value returns zero."""
        reg = OrthogonalReg(weight=1.0)
        reset = reg.reset_value()
        self.assertEqual(reset, 0.0)

    def test_fit_hyper_true(self):
        """Test that fit_hyper=True is stored correctly."""
        reg = OrthogonalReg(weight=1.0, fit_hyper=True)
        self.assertTrue(reg.fit_hyper)


class TestSpectralNormReg(unittest.TestCase):
    """Tests for Spectral Norm regularization."""

    def test_within_spectral_norm_zero_loss(self):
        """Test that matrix within spectral norm bound has zero loss."""
        reg = SpectralNormReg(weight=1.0, max_value=2.0, n_power_iterations=10)
        W = jnp.array([[1.0, 0.0], [0.0, 1.0]])  # spectral norm = 1 < 2
        loss = reg.loss(W)
        np.testing.assert_allclose(loss, 0.0, atol=1e-3)

    def test_exceeds_spectral_norm_has_loss(self):
        """Test that matrix exceeding spectral norm has positive loss."""
        reg = SpectralNormReg(weight=1.0, max_value=1.0, n_power_iterations=10)
        W = jnp.array([[3.0, 0.0], [0.0, 1.0]])  # spectral norm = 3 > 1
        loss = reg.loss(W)
        # (3 - 1)^2 = 4
        np.testing.assert_allclose(loss, 4.0, rtol=0.1)

    def test_sample_init_shape(self):
        """Test sample_init returns correct shape."""
        reg = SpectralNormReg(weight=1.0, max_value=1.0)
        sample = reg.sample_init((4, 3))
        self.assertEqual(sample.shape, (4, 3))

    def test_reset_value(self):
        """Test reset_value returns zero."""
        reg = SpectralNormReg(weight=1.0, max_value=1.0)
        reset = reg.reset_value()
        self.assertEqual(reset, 0.0)

    def test_fit_hyper_true(self):
        """Test that fit_hyper=True is stored correctly."""
        reg = SpectralNormReg(weight=1.0, max_value=1.0, fit_hyper=True)
        self.assertTrue(reg.fit_hyper)


class TestNewRegularizationsInheritance(unittest.TestCase):
    """Tests for new regularization classes inheritance."""

    def test_elastic_net_inherits_from_module(self):
        """Test ElasticNetReg inherits from brainstate.nn.Module."""
        reg = ElasticNetReg(l1_weight=1.0, l2_weight=1.0, alpha=0.5)
        self.assertIsInstance(reg, brainstate.nn.Module)

    def test_huber_inherits_from_module(self):
        """Test HuberReg inherits from brainstate.nn.Module."""
        reg = HuberReg(weight=1.0, delta=1.0)
        self.assertIsInstance(reg, brainstate.nn.Module)

    def test_group_lasso_inherits_from_module(self):
        """Test GroupLassoReg inherits from brainstate.nn.Module."""
        reg = GroupLassoReg(weight=1.0, group_size=2)
        self.assertIsInstance(reg, brainstate.nn.Module)

    def test_total_variation_inherits_from_module(self):
        """Test TotalVariationReg inherits from brainstate.nn.Module."""
        reg = TotalVariationReg(weight=1.0, order=1)
        self.assertIsInstance(reg, brainstate.nn.Module)

    def test_max_norm_inherits_from_module(self):
        """Test MaxNormReg inherits from brainstate.nn.Module."""
        reg = MaxNormReg(weight=1.0, max_value=1.0)
        self.assertIsInstance(reg, brainstate.nn.Module)

    def test_entropy_inherits_from_module(self):
        """Test EntropyReg inherits from brainstate.nn.Module."""
        reg = EntropyReg(weight=1.0, maximize=True)
        self.assertIsInstance(reg, brainstate.nn.Module)

    def test_orthogonal_inherits_from_module(self):
        """Test OrthogonalReg inherits from brainstate.nn.Module."""
        reg = OrthogonalReg(weight=1.0)
        self.assertIsInstance(reg, brainstate.nn.Module)

    def test_spectral_norm_inherits_from_module(self):
        """Test SpectralNormReg inherits from brainstate.nn.Module."""
        reg = SpectralNormReg(weight=1.0, max_value=1.0)
        self.assertIsInstance(reg, brainstate.nn.Module)


# =============================================================================
# Tests for Prior Distribution-Based Regularizations
# =============================================================================


class TestStudentTReg(unittest.TestCase):
    """Tests for Student's t-distribution regularization."""

    def test_basic_loss(self):
        """Test Student-t loss computation."""
        reg = StudentTReg(weight=1.0, df=3.0, scale=1.0)
        value = jnp.array([0.0])
        loss = reg.loss(value)
        # At x=0, loss = log(1 + 0) = 0
        np.testing.assert_allclose(loss, 0.0, atol=1e-5)

    def test_loss_increases_with_distance(self):
        """Test that loss increases with distance from zero."""
        reg = StudentTReg(weight=1.0, df=3.0, scale=1.0)
        loss_0 = reg.loss(jnp.array([0.0]))
        loss_1 = reg.loss(jnp.array([1.0]))
        loss_2 = reg.loss(jnp.array([2.0]))
        self.assertLess(float(loss_0), float(loss_1))
        self.assertLess(float(loss_1), float(loss_2))

    def test_heavy_tails(self):
        """Test that Student-t has heavier tails than Gaussian."""
        # Student-t with low df penalizes outliers less than Gaussian
        student_t = StudentTReg(weight=1.0, df=1.0, scale=1.0)
        gaussian = GaussianReg(mean=0.0, std=1.0)
        outlier = jnp.array([5.0])
        # Relative penalty should be lower for Student-t
        t_loss = student_t.loss(outlier)
        g_loss = gaussian.loss(outlier)
        # Student-t loss grows slower than quadratic
        self.assertLess(float(t_loss), float(g_loss))

    def test_sample_init_shape(self):
        """Test sample_init returns correct shape."""
        reg = StudentTReg(weight=1.0, df=3.0, scale=1.0)
        sample = reg.sample_init((3, 4))
        self.assertEqual(sample.shape, (3, 4))

    def test_reset_value(self):
        """Test reset_value returns zero."""
        reg = StudentTReg(weight=1.0, df=3.0, scale=1.0)
        reset = reg.reset_value()
        self.assertEqual(reset, 0.0)

    def test_fit_hyper_true(self):
        """Test that fit_hyper=True is stored correctly."""
        reg = StudentTReg(weight=1.0, df=3.0, scale=1.0, fit_hyper=True)
        self.assertTrue(reg.fit_hyper)

    def test_inherits_from_module(self):
        """Test StudentTReg inherits from brainstate.nn.Module."""
        reg = StudentTReg(weight=1.0, df=3.0, scale=1.0)
        self.assertIsInstance(reg, brainstate.nn.Module)


class TestCauchyReg(unittest.TestCase):
    """Tests for Cauchy distribution regularization."""

    def test_basic_loss(self):
        """Test Cauchy loss computation."""
        reg = CauchyReg(weight=1.0, scale=1.0)
        value = jnp.array([0.0])
        loss = reg.loss(value)
        # At x=0, loss = log(1 + 0) = 0
        np.testing.assert_allclose(loss, 0.0, atol=1e-5)

    def test_loss_at_scale(self):
        """Test Cauchy loss at x=scale."""
        reg = CauchyReg(weight=1.0, scale=1.0)
        value = jnp.array([1.0])
        loss = reg.loss(value)
        # At x=scale, loss = log(1 + 1) = log(2)
        np.testing.assert_allclose(loss, jnp.log(2.0), rtol=1e-5)

    def test_sample_init_shape(self):
        """Test sample_init returns correct shape."""
        reg = CauchyReg(weight=1.0, scale=1.0)
        sample = reg.sample_init((3, 4))
        self.assertEqual(sample.shape, (3, 4))

    def test_reset_value(self):
        """Test reset_value returns zero."""
        reg = CauchyReg(weight=1.0, scale=1.0)
        reset = reg.reset_value()
        self.assertEqual(reset, 0.0)

    def test_fit_hyper_true(self):
        """Test that fit_hyper=True is stored correctly."""
        reg = CauchyReg(weight=1.0, scale=1.0, fit_hyper=True)
        self.assertTrue(reg.fit_hyper)

    def test_inherits_from_module(self):
        """Test CauchyReg inherits from brainstate.nn.Module."""
        reg = CauchyReg(weight=1.0, scale=1.0)
        self.assertIsInstance(reg, brainstate.nn.Module)


class TestUniformReg(unittest.TestCase):
    """Tests for Uniform prior regularization."""

    def test_within_bounds_zero_loss(self):
        """Test that values within bounds have zero loss."""
        reg = UniformReg(weight=1.0, lower=-1.0, upper=1.0)
        value = jnp.array([0.0, 0.5, -0.5])
        loss = reg.loss(value)
        np.testing.assert_allclose(loss, 0.0, atol=1e-5)

    def test_outside_bounds_has_loss(self):
        """Test that values outside bounds have positive loss."""
        reg = UniformReg(weight=1.0, lower=-1.0, upper=1.0)
        value = jnp.array([2.0])  # exceeds upper by 1
        loss = reg.loss(value)
        # (2 - 1)^2 = 1
        np.testing.assert_allclose(loss, 1.0, rtol=1e-5)

    def test_both_bounds_exceeded(self):
        """Test with values exceeding both bounds."""
        reg = UniformReg(weight=1.0, lower=-1.0, upper=1.0)
        value = jnp.array([2.0, -2.0])  # both exceed by 1
        loss = reg.loss(value)
        # (1)^2 + (1)^2 = 2
        np.testing.assert_allclose(loss, 2.0, rtol=1e-5)

    def test_sample_init_shape(self):
        """Test sample_init returns correct shape."""
        reg = UniformReg(weight=1.0, lower=-1.0, upper=1.0)
        sample = reg.sample_init((3, 4))
        self.assertEqual(sample.shape, (3, 4))

    def test_sample_init_bounded(self):
        """Test that samples are within bounds."""
        reg = UniformReg(weight=1.0, lower=-1.0, upper=1.0)
        sample = reg.sample_init((100,))
        self.assertTrue(jnp.all(sample >= -1.0))
        self.assertTrue(jnp.all(sample <= 1.0))

    def test_reset_value(self):
        """Test reset_value returns midpoint."""
        reg = UniformReg(weight=1.0, lower=-2.0, upper=4.0)
        reset = reg.reset_value()
        np.testing.assert_allclose(reset, 1.0)  # (-2 + 4) / 2 = 1

    def test_fit_hyper_true(self):
        """Test that fit_hyper=True is stored correctly."""
        reg = UniformReg(weight=1.0, lower=-1.0, upper=1.0, fit_hyper=True)
        self.assertTrue(reg.fit_hyper)

    def test_inherits_from_module(self):
        """Test UniformReg inherits from brainstate.nn.Module."""
        reg = UniformReg(weight=1.0, lower=-1.0, upper=1.0)
        self.assertIsInstance(reg, brainstate.nn.Module)


class TestLogNormalReg(unittest.TestCase):
    """Tests for Log-normal prior regularization."""

    def test_loss_at_exp_mu(self):
        """Test that loss is minimized near exp(mu)."""
        reg = LogNormalReg(weight=1.0, mu=0.0, sigma=1.0)
        # Mode is at exp(mu - sigma^2) = exp(-1)
        loss_at_mode = reg.loss(jnp.array([jnp.exp(-1.0)]))
        loss_away = reg.loss(jnp.array([jnp.exp(2.0)]))
        self.assertLess(float(loss_at_mode), float(loss_away))

    def test_sample_init_shape(self):
        """Test sample_init returns correct shape."""
        reg = LogNormalReg(weight=1.0, mu=0.0, sigma=1.0)
        sample = reg.sample_init((3, 4))
        self.assertEqual(sample.shape, (3, 4))

    def test_sample_init_positive(self):
        """Test that samples are positive."""
        reg = LogNormalReg(weight=1.0, mu=0.0, sigma=1.0)
        sample = reg.sample_init((100,))
        self.assertTrue(jnp.all(sample > 0))

    def test_reset_value(self):
        """Test reset_value returns exp(mu)."""
        reg = LogNormalReg(weight=1.0, mu=1.0, sigma=1.0)
        reset = reg.reset_value()
        np.testing.assert_allclose(reset, jnp.exp(1.0), rtol=1e-5)

    def test_fit_hyper_true(self):
        """Test that fit_hyper=True is stored correctly."""
        reg = LogNormalReg(weight=1.0, mu=0.0, sigma=1.0, fit_hyper=True)
        self.assertTrue(reg.fit_hyper)

    def test_inherits_from_module(self):
        """Test LogNormalReg inherits from brainstate.nn.Module."""
        reg = LogNormalReg(weight=1.0, mu=0.0, sigma=1.0)
        self.assertIsInstance(reg, brainstate.nn.Module)


class TestExponentialReg(unittest.TestCase):
    """Tests for Exponential prior regularization."""

    def test_basic_loss(self):
        """Test Exponential loss computation."""
        reg = ExponentialReg(weight=1.0, rate=1.0)
        value = jnp.array([1.0, 2.0, 3.0])
        loss = reg.loss(value)
        # Loss = rate * sum(x) = 1 * 6 = 6
        np.testing.assert_allclose(loss, 6.0, rtol=1e-5)

    def test_weighted_loss(self):
        """Test Exponential loss with different weight and rate."""
        reg = ExponentialReg(weight=2.0, rate=0.5)
        value = jnp.array([2.0, 2.0])
        loss = reg.loss(value)
        # Loss = 2 * 0.5 * 4 = 4
        np.testing.assert_allclose(loss, 4.0, rtol=1e-5)

    def test_sample_init_shape(self):
        """Test sample_init returns correct shape."""
        reg = ExponentialReg(weight=1.0, rate=1.0)
        sample = reg.sample_init((3, 4))
        self.assertEqual(sample.shape, (3, 4))

    def test_sample_init_positive(self):
        """Test that samples are positive."""
        reg = ExponentialReg(weight=1.0, rate=1.0)
        sample = reg.sample_init((100,))
        self.assertTrue(jnp.all(sample > 0))

    def test_reset_value(self):
        """Test reset_value returns zero."""
        reg = ExponentialReg(weight=1.0, rate=1.0)
        reset = reg.reset_value()
        self.assertEqual(reset, 0.0)

    def test_fit_hyper_true(self):
        """Test that fit_hyper=True is stored correctly."""
        reg = ExponentialReg(weight=1.0, rate=1.0, fit_hyper=True)
        self.assertTrue(reg.fit_hyper)

    def test_inherits_from_module(self):
        """Test ExponentialReg inherits from brainstate.nn.Module."""
        reg = ExponentialReg(weight=1.0, rate=1.0)
        self.assertIsInstance(reg, brainstate.nn.Module)


class TestGammaReg(unittest.TestCase):
    """Tests for Gamma prior regularization."""

    def test_loss_increases_from_mode(self):
        """Test that loss increases away from mode."""
        reg = GammaReg(weight=1.0, alpha=3.0, beta=1.0)
        # Mode is at (alpha - 1) / beta = 2
        loss_at_mode = reg.loss(jnp.array([2.0]))
        loss_away = reg.loss(jnp.array([5.0]))
        self.assertLess(float(loss_at_mode), float(loss_away))

    def test_sample_init_shape(self):
        """Test sample_init returns correct shape."""
        reg = GammaReg(weight=1.0, alpha=2.0, beta=1.0)
        sample = reg.sample_init((3, 4))
        self.assertEqual(sample.shape, (3, 4))

    def test_sample_init_positive(self):
        """Test that samples are positive."""
        reg = GammaReg(weight=1.0, alpha=2.0, beta=1.0)
        sample = reg.sample_init((100,))
        self.assertTrue(jnp.all(sample > 0))

    def test_reset_value(self):
        """Test reset_value returns mode."""
        reg = GammaReg(weight=1.0, alpha=3.0, beta=2.0)
        reset = reg.reset_value()
        # Mode = (alpha - 1) / beta = 2 / 2 = 1
        np.testing.assert_allclose(reset, 1.0, rtol=1e-3)

    def test_fit_hyper_true(self):
        """Test that fit_hyper=True is stored correctly."""
        reg = GammaReg(weight=1.0, alpha=2.0, beta=1.0, fit_hyper=True)
        self.assertTrue(reg.fit_hyper)

    def test_inherits_from_module(self):
        """Test GammaReg inherits from brainstate.nn.Module."""
        reg = GammaReg(weight=1.0, alpha=2.0, beta=1.0)
        self.assertIsInstance(reg, brainstate.nn.Module)


class TestBetaReg(unittest.TestCase):
    """Tests for Beta prior regularization."""

    def test_symmetric_beta_favors_half(self):
        """Test that symmetric Beta favors 0.5."""
        reg = BetaReg(weight=1.0, a=2.0, b=2.0)
        loss_half = reg.loss(jnp.array([0.5]))
        loss_edge = reg.loss(jnp.array([0.1]))
        self.assertLess(float(loss_half), float(loss_edge))

    def test_sample_init_shape(self):
        """Test sample_init returns correct shape."""
        reg = BetaReg(weight=1.0, a=2.0, b=2.0)
        sample = reg.sample_init((3, 4))
        self.assertEqual(sample.shape, (3, 4))

    def test_sample_init_bounded(self):
        """Test that samples are in [0, 1]."""
        reg = BetaReg(weight=1.0, a=2.0, b=2.0)
        sample = reg.sample_init((100,))
        self.assertTrue(jnp.all(sample >= 0))
        self.assertTrue(jnp.all(sample <= 1))

    def test_reset_value(self):
        """Test reset_value returns mode."""
        reg = BetaReg(weight=1.0, a=3.0, b=2.0)
        reset = reg.reset_value()
        # Mode = (a - 1) / (a + b - 2) = 2 / 3
        np.testing.assert_allclose(reset, 2.0 / 3.0, rtol=1e-3)

    def test_fit_hyper_true(self):
        """Test that fit_hyper=True is stored correctly."""
        reg = BetaReg(weight=1.0, a=2.0, b=2.0, fit_hyper=True)
        self.assertTrue(reg.fit_hyper)

    def test_inherits_from_module(self):
        """Test BetaReg inherits from brainstate.nn.Module."""
        reg = BetaReg(weight=1.0, a=2.0, b=2.0)
        self.assertIsInstance(reg, brainstate.nn.Module)


class TestHorseshoeReg(unittest.TestCase):
    """Tests for Horseshoe prior regularization."""

    def test_basic_loss(self):
        """Test Horseshoe loss computation."""
        reg = HorseshoeReg(weight=1.0, tau=1.0)
        value = jnp.array([0.0])
        loss = reg.loss(value)
        # At x=0, loss = log(1 + 0) = 0
        np.testing.assert_allclose(loss, 0.0, atol=1e-5)

    def test_sparsity_encouraging(self):
        """Test that Horseshoe encourages sparsity."""
        reg = HorseshoeReg(weight=1.0, tau=0.1)
        small_val = jnp.array([0.01])
        large_val = jnp.array([1.0])
        loss_small = reg.loss(small_val)
        loss_large = reg.loss(large_val)
        self.assertLess(float(loss_small), float(loss_large))

    def test_sample_init_shape(self):
        """Test sample_init returns correct shape."""
        reg = HorseshoeReg(weight=1.0, tau=1.0)
        sample = reg.sample_init((3, 4))
        self.assertEqual(sample.shape, (3, 4))

    def test_reset_value(self):
        """Test reset_value returns zero."""
        reg = HorseshoeReg(weight=1.0, tau=1.0)
        reset = reg.reset_value()
        self.assertEqual(reset, 0.0)

    def test_fit_hyper_true(self):
        """Test that fit_hyper=True is stored correctly."""
        reg = HorseshoeReg(weight=1.0, tau=1.0, fit_hyper=True)
        self.assertTrue(reg.fit_hyper)

    def test_inherits_from_module(self):
        """Test HorseshoeReg inherits from brainstate.nn.Module."""
        reg = HorseshoeReg(weight=1.0, tau=1.0)
        self.assertIsInstance(reg, brainstate.nn.Module)


class TestInverseGammaReg(unittest.TestCase):
    """Tests for Inverse-Gamma prior regularization."""

    def test_loss_increases_from_mode(self):
        """Test that loss increases away from mode."""
        reg = InverseGammaReg(weight=1.0, alpha=3.0, beta=2.0)
        # Mode is beta / (alpha + 1) = 2 / 4 = 0.5
        loss_at_mode = reg.loss(jnp.array([0.5]))
        loss_away = reg.loss(jnp.array([2.0]))
        self.assertLess(float(loss_at_mode), float(loss_away))

    def test_sample_init_shape(self):
        """Test sample_init returns correct shape."""
        reg = InverseGammaReg(weight=1.0, alpha=2.0, beta=1.0)
        sample = reg.sample_init((3, 4))
        self.assertEqual(sample.shape, (3, 4))

    def test_sample_init_positive(self):
        """Test that samples are positive."""
        reg = InverseGammaReg(weight=1.0, alpha=2.0, beta=1.0)
        sample = reg.sample_init((100,))
        self.assertTrue(jnp.all(sample > 0))

    def test_reset_value(self):
        """Test reset_value returns mode."""
        reg = InverseGammaReg(weight=1.0, alpha=3.0, beta=2.0)
        reset = reg.reset_value()
        # Mode = beta / (alpha + 1) = 2 / 4 = 0.5
        np.testing.assert_allclose(reset, 0.5, rtol=1e-3)

    def test_fit_hyper_true(self):
        """Test that fit_hyper=True is stored correctly."""
        reg = InverseGammaReg(weight=1.0, alpha=2.0, beta=1.0, fit_hyper=True)
        self.assertTrue(reg.fit_hyper)

    def test_inherits_from_module(self):
        """Test InverseGammaReg inherits from brainstate.nn.Module."""
        reg = InverseGammaReg(weight=1.0, alpha=2.0, beta=1.0)
        self.assertIsInstance(reg, brainstate.nn.Module)


class TestLogUniformReg(unittest.TestCase):
    """Tests for Log-uniform (Jeffreys) prior regularization."""

    def test_loss_proportional_to_log(self):
        """Test that loss is proportional to log(x)."""
        reg = LogUniformReg(weight=1.0, lower=1e-3, upper=1e3)
        value1 = jnp.array([1.0])
        value10 = jnp.array([10.0])
        loss1 = reg.loss(value1)
        loss10 = reg.loss(value10)
        # log(10) > log(1), so loss10 > loss1
        self.assertGreater(float(loss10), float(loss1))

    def test_sample_init_shape(self):
        """Test sample_init returns correct shape."""
        reg = LogUniformReg(weight=1.0, lower=1e-3, upper=1e3)
        sample = reg.sample_init((3, 4))
        self.assertEqual(sample.shape, (3, 4))

    def test_sample_init_bounded(self):
        """Test that samples are within bounds."""
        reg = LogUniformReg(weight=1.0, lower=0.1, upper=10.0)
        sample = reg.sample_init((100,))
        self.assertTrue(jnp.all(sample >= 0.1))
        self.assertTrue(jnp.all(sample <= 10.0))

    def test_reset_value(self):
        """Test reset_value returns geometric mean."""
        reg = LogUniformReg(weight=1.0, lower=1.0, upper=100.0)
        reset = reg.reset_value()
        # Geometric mean = sqrt(1 * 100) = 10
        np.testing.assert_allclose(reset, 10.0, rtol=1e-3)

    def test_fit_hyper_true(self):
        """Test that fit_hyper=True is stored correctly."""
        reg = LogUniformReg(weight=1.0, lower=1e-3, upper=1e3, fit_hyper=True)
        self.assertTrue(reg.fit_hyper)

    def test_inherits_from_module(self):
        """Test LogUniformReg inherits from brainstate.nn.Module."""
        reg = LogUniformReg(weight=1.0, lower=1e-3, upper=1e3)
        self.assertIsInstance(reg, brainstate.nn.Module)


class TestSpikeAndSlabReg(unittest.TestCase):
    """Tests for Spike-and-slab prior regularization."""

    def test_spike_favors_zero(self):
        """Test that high pi favors values near zero."""
        reg = SpikeAndSlabReg(weight=1.0, spike_scale=0.01, slab_scale=1.0, pi=0.9)
        loss_zero = reg.loss(jnp.array([0.0]))
        loss_large = reg.loss(jnp.array([1.0]))
        self.assertLess(float(loss_zero), float(loss_large))

    def test_slab_allows_large_values(self):
        """Test that low pi (more slab) allows larger values."""
        reg_spike = SpikeAndSlabReg(weight=1.0, spike_scale=0.01, slab_scale=1.0, pi=0.9)
        reg_slab = SpikeAndSlabReg(weight=1.0, spike_scale=0.01, slab_scale=1.0, pi=0.1)
        value = jnp.array([0.5])
        loss_spike = reg_spike.loss(value)
        loss_slab = reg_slab.loss(value)
        # Slab-heavy should penalize moderate values less
        self.assertLess(float(loss_slab), float(loss_spike))

    def test_sample_init_shape(self):
        """Test sample_init returns correct shape."""
        reg = SpikeAndSlabReg(weight=1.0, spike_scale=0.01, slab_scale=1.0, pi=0.5)
        sample = reg.sample_init((3, 4))
        self.assertEqual(sample.shape, (3, 4))

    def test_reset_value(self):
        """Test reset_value returns zero."""
        reg = SpikeAndSlabReg(weight=1.0, spike_scale=0.01, slab_scale=1.0, pi=0.5)
        reset = reg.reset_value()
        self.assertEqual(reset, 0.0)

    def test_fit_hyper_true(self):
        """Test that fit_hyper=True is stored correctly."""
        reg = SpikeAndSlabReg(weight=1.0, spike_scale=0.01, slab_scale=1.0, pi=0.5, fit_hyper=True)
        self.assertTrue(reg.fit_hyper)

    def test_inherits_from_module(self):
        """Test SpikeAndSlabReg inherits from brainstate.nn.Module."""
        reg = SpikeAndSlabReg(weight=1.0, spike_scale=0.01, slab_scale=1.0, pi=0.5)
        self.assertIsInstance(reg, brainstate.nn.Module)


class TestDirichletReg(unittest.TestCase):
    """Tests for Dirichlet prior regularization."""

    def test_uniform_alpha_favors_uniform_probs(self):
        """Test that alpha=1 (uniform Dirichlet) doesn't favor any distribution."""
        reg = DirichletReg(weight=1.0, alpha=1.0)
        uniform_logits = jnp.array([0.0, 0.0, 0.0])
        concentrated_logits = jnp.array([10.0, 0.0, 0.0])
        loss_uniform = reg.loss(uniform_logits)
        loss_concentrated = reg.loss(concentrated_logits)
        # With alpha=1, both should have similar loss (close to 0)
        np.testing.assert_allclose(loss_uniform, 0.0, atol=1e-5)
        np.testing.assert_allclose(loss_concentrated, 0.0, atol=1e-5)

    def test_high_alpha_favors_uniform(self):
        """Test that alpha>1 favors uniform distributions."""
        reg = DirichletReg(weight=1.0, alpha=5.0)
        uniform_logits = jnp.array([0.0, 0.0, 0.0])
        concentrated_logits = jnp.array([10.0, 0.0, 0.0])
        loss_uniform = reg.loss(uniform_logits)
        loss_concentrated = reg.loss(concentrated_logits)
        # High alpha should favor uniform
        self.assertLess(float(loss_uniform), float(loss_concentrated))

    def test_sample_init_shape(self):
        """Test sample_init returns correct shape."""
        reg = DirichletReg(weight=1.0, alpha=1.0)
        sample = reg.sample_init((5,))
        self.assertEqual(sample.shape, (5,))

    def test_reset_value(self):
        """Test reset_value returns zero."""
        reg = DirichletReg(weight=1.0, alpha=1.0)
        reset = reg.reset_value()
        self.assertEqual(reset, 0.0)

    def test_fit_hyper_true(self):
        """Test that fit_hyper=True is stored correctly."""
        reg = DirichletReg(weight=1.0, alpha=1.0, fit_hyper=True)
        self.assertTrue(reg.fit_hyper)

    def test_inherits_from_module(self):
        """Test DirichletReg inherits from brainstate.nn.Module."""
        reg = DirichletReg(weight=1.0, alpha=1.0)
        self.assertIsInstance(reg, brainstate.nn.Module)


class TestChainedReg(unittest.TestCase):
    """Tests for ChainedReg (composite regularization)."""

    def test_combines_multiple_regularizations(self):
        """Test that ChainedReg combines multiple regularization losses."""
        reg1 = L1Reg(weight=1.0)
        reg2 = L2Reg(weight=1.0)
        chained = ChainedReg(reg1, reg2)
        value = jnp.array([1.0, 2.0])

        # Individual losses
        l1_loss = reg1.loss(value)  # |1| + |2| = 3
        l2_loss = reg2.loss(value)  # 1^2 + 2^2 = 5

        # Chained should sum them
        chained_loss = chained.loss(value)
        np.testing.assert_allclose(chained_loss, l1_loss + l2_loss, rtol=1e-5)

    def test_weight_scales_combined_loss(self):
        """Test that weight parameter scales the combined loss."""
        reg1 = L1Reg(weight=1.0)
        reg2 = L2Reg(weight=1.0)
        chained1 = ChainedReg(reg1, reg2, weight=1.0)
        chained2 = ChainedReg(reg1, reg2, weight=2.0)
        value = jnp.array([1.0, 2.0])

        loss1 = chained1.loss(value)
        loss2 = chained2.loss(value)
        np.testing.assert_allclose(float(loss2), float(loss1) * 2.0, rtol=1e-5)

    def test_single_regularization(self):
        """Test ChainedReg with single regularization."""
        reg = L2Reg(weight=1.0)
        chained = ChainedReg(reg)
        value = jnp.array([1.0, 2.0, 3.0])

        np.testing.assert_allclose(chained.loss(value), reg.loss(value), rtol=1e-5)

    def test_empty_chain_zero_loss(self):
        """Test that empty chain returns zero loss."""
        chained = ChainedReg()
        value = jnp.array([1.0, 2.0])
        loss = chained.loss(value)
        self.assertEqual(loss, 0.0)

    def test_sample_init_uses_first_regularization(self):
        """Test that sample_init uses the first regularization."""
        reg1 = GaussianReg(mean=5.0, std=0.001)  # Very narrow around 5
        reg2 = L2Reg(weight=1.0)
        chained = ChainedReg(reg1, reg2)

        sample = chained.sample_init((100,))
        # Should be around 5.0 (from Gaussian prior)
        np.testing.assert_allclose(jnp.mean(sample), 5.0, atol=0.1)

    def test_sample_init_empty_chain(self):
        """Test that empty chain returns zeros for sample_init."""
        chained = ChainedReg()
        sample = chained.sample_init((3, 4))
        self.assertEqual(sample.shape, (3, 4))
        np.testing.assert_allclose(sample, jnp.zeros((3, 4)))

    def test_reset_value_uses_first_regularization(self):
        """Test that reset_value uses the first regularization."""
        reg1 = GaussianReg(mean=3.5, std=1.0)
        reg2 = L2Reg(weight=1.0)
        chained = ChainedReg(reg1, reg2)

        reset = chained.reset_value()
        np.testing.assert_allclose(reset, 3.5)

    def test_reset_value_empty_chain(self):
        """Test that empty chain returns zero for reset_value."""
        chained = ChainedReg()
        reset = chained.reset_value()
        self.assertEqual(reset, 0.0)

    def test_fit_hyper_false(self):
        """Test that weight is not trainable by default."""
        chained = ChainedReg(L1Reg(weight=1.0), L2Reg(weight=1.0), fit_hyper=False)
        self.assertFalse(chained.fit_hyper)
        self.assertNotIsInstance(chained.weight, brainstate.State)

    def test_fit_hyper_true(self):
        """Test that fit_hyper=True is stored correctly."""
        chained = ChainedReg(L1Reg(weight=1.0), L2Reg(weight=1.0), fit_hyper=True)
        self.assertTrue(chained.fit_hyper)

    def test_inherits_from_module(self):
        """Test ChainedReg inherits from brainstate.nn.Module."""
        chained = ChainedReg(L1Reg(weight=1.0))
        self.assertIsInstance(chained, brainstate.nn.Module)

    def test_three_regularizations(self):
        """Test ChainedReg with three regularizations."""
        reg1 = L1Reg(weight=1.0)
        reg2 = L2Reg(weight=1.0)
        reg3 = UniformReg(weight=1.0, lower=-1.0, upper=1.0)
        chained = ChainedReg(reg1, reg2, reg3)

        value = jnp.array([0.5, -0.5])
        expected = reg1.loss(value) + reg2.loss(value) + reg3.loss(value)
        np.testing.assert_allclose(chained.loss(value), expected, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
