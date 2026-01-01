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

"""Tests for Param and Const parameter modules."""

import unittest

import brainstate
import jax.numpy as jnp
import numpy as np

from braintools.param import (
    Param,
    Const,
    IdentityT,
    SigmoidT,
    SoftplusT,
    L1Reg,
    L2Reg,
    GaussianReg,
)


class TestParamBasic(unittest.TestCase):
    """Tests for basic Param functionality."""

    def test_basic_instantiation(self):
        """Test basic instantiation with default parameters."""
        param = Param(jnp.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(param.value(), jnp.array([1.0, 2.0, 3.0]))

    def test_trainable_by_default(self):
        """Test that parameters are trainable by default."""
        param = Param(jnp.array([1.0]))
        self.assertTrue(param.fit_par)
        self.assertIsInstance(param.val, brainstate.ParamState)

    def test_non_trainable(self):
        """Test creating non-trainable parameter."""
        param = Param(jnp.array([1.0]), fit_par=False)
        self.assertFalse(param.fit_par)
        self.assertNotIsInstance(param.val, brainstate.State)

    def test_value_method(self):
        """Test value() method returns correct value."""
        value = jnp.array([1.0, 2.0])
        param = Param(value)
        np.testing.assert_allclose(param.value(), value)

    def test_set_value_method(self):
        """Test set_value() method."""
        param = Param(jnp.array([1.0, 2.0]))
        new_value = jnp.array([3.0, 4.0])
        param.set_value(new_value)
        np.testing.assert_allclose(param.value(), new_value)

    def test_inherits_from_module(self):
        """Test that Param inherits from brainstate.nn.Module."""
        param = Param(jnp.array([1.0]))
        self.assertIsInstance(param, brainstate.nn.Module)


class TestParamWithTransform(unittest.TestCase):
    """Tests for Param with transforms."""

    def test_with_identity_transform(self):
        """Test with explicit identity transform."""
        param = Param(jnp.array([1.0, 2.0]), t=IdentityT())
        np.testing.assert_allclose(param.value(), jnp.array([1.0, 2.0]))

    def test_with_softplus_transform(self):
        """Test with softplus transform for positive constraint."""
        value = jnp.array([1.0, 2.0, 3.0])
        param = Param(value, t=SoftplusT(0.0))
        # Value should match original
        np.testing.assert_allclose(param.value(), value, rtol=1e-5)

    def test_with_sigmoid_transform(self):
        """Test with sigmoid transform for bounded constraint."""
        value = jnp.array([0.3, 0.5, 0.7])
        param = Param(value, t=SigmoidT(0.0, 1.0))
        # Value should approximately match original
        np.testing.assert_allclose(param.value(), value, rtol=1e-4)

    def test_set_value_with_transform(self):
        """Test set_value with transform."""
        param = Param(jnp.array([1.0]), t=SoftplusT(0.0))
        new_value = jnp.array([5.0])
        param.set_value(new_value)
        np.testing.assert_allclose(param.value(), new_value, rtol=1e-5)

    def test_invalid_transform_raises(self):
        """Test that invalid transform raises TypeError."""
        with self.assertRaises(TypeError):
            Param(jnp.array([1.0]), t="not a transform")


class TestParamWithRegularization(unittest.TestCase):
    """Tests for Param with regularization."""

    def test_with_l2_reg(self):
        """Test with L2 regularization."""
        param = Param(jnp.array([1.0, 2.0]), reg=L2Reg(weight=0.1))
        loss = param.reg_loss()
        # L2 loss = 0.1 * (1.0^2 + 2.0^2) = 0.1 * 5.0 = 0.5
        np.testing.assert_allclose(loss, 0.5, rtol=1e-5)

    def test_with_l1_reg(self):
        """Test with L1 regularization."""
        param = Param(jnp.array([1.0, -2.0]), reg=L1Reg(weight=0.1))
        loss = param.reg_loss()
        # L1 loss = 0.1 * (|1.0| + |-2.0|) = 0.1 * 3.0 = 0.3
        np.testing.assert_allclose(loss, 0.3, rtol=1e-5)

    def test_with_gaussian_reg(self):
        """Test with Gaussian regularization."""
        param = Param(jnp.array([0.5]), reg=GaussianReg(mean=0.0, std=1.0))
        loss = param.reg_loss()
        # Loss should be positive
        self.assertGreater(float(loss), 0.0)

    def test_no_reg_returns_zero(self):
        """Test that no regularization returns zero loss."""
        param = Param(jnp.array([1.0, 2.0]))
        loss = param.reg_loss()
        self.assertEqual(loss, 0.0)

    def test_non_trainable_returns_zero_loss(self):
        """Test that non-trainable param returns zero reg loss."""
        param = Param(jnp.array([1.0]), reg=L2Reg(weight=0.1), fit_par=False)
        loss = param.reg_loss()
        self.assertEqual(loss, 0.0)

    def test_invalid_reg_raises(self):
        """Test that invalid regularization raises ValueError."""
        with self.assertRaises(ValueError):
            Param(jnp.array([1.0]), reg="not a regularization")


class TestParamResetToPrior(unittest.TestCase):
    """Tests for reset_to_prior functionality."""

    def test_reset_to_prior_gaussian(self):
        """Test reset_to_prior with Gaussian reg."""
        param = Param(jnp.array([5.0]), reg=GaussianReg(mean=0.0, std=1.0))
        param.reset_to_prior()
        np.testing.assert_allclose(param.value(), jnp.array([0.0]))

    def test_reset_to_prior_l2(self):
        """Test reset_to_prior with L2 reg (resets to zero)."""
        param = Param(jnp.array([5.0]), reg=L2Reg(weight=0.1))
        param.reset_to_prior()
        np.testing.assert_allclose(param.value(), jnp.array([0.0]))

    def test_reset_to_prior_no_reg(self):
        """Test reset_to_prior with no reg does nothing."""
        original = jnp.array([5.0])
        param = Param(original)
        param.reset_to_prior()
        np.testing.assert_allclose(param.value(), original)


class TestParamClip(unittest.TestCase):
    """Tests for clip functionality."""

    def test_clip_upper(self):
        """Test clipping upper bound."""
        param = Param(jnp.array([5.0, 10.0]))
        param.clip(max_val=7.0)
        np.testing.assert_allclose(param.value(), jnp.array([5.0, 7.0]))

    def test_clip_lower(self):
        """Test clipping lower bound."""
        param = Param(jnp.array([1.0, 5.0]))
        param.clip(min_val=3.0)
        np.testing.assert_allclose(param.value(), jnp.array([3.0, 5.0]))

    def test_clip_both(self):
        """Test clipping both bounds."""
        param = Param(jnp.array([1.0, 5.0, 10.0]))
        param.clip(min_val=2.0, max_val=8.0)
        np.testing.assert_allclose(param.value(), jnp.array([2.0, 5.0, 8.0]))


class TestConst(unittest.TestCase):
    """Tests for Const (non-trainable parameter)."""

    def test_basic_instantiation(self):
        """Test basic instantiation."""
        const = Const(jnp.array([1.0, 2.0]))
        np.testing.assert_allclose(const.value(), jnp.array([1.0, 2.0]))

    def test_not_trainable(self):
        """Test that Const is not trainable."""
        const = Const(jnp.array([1.0]))
        self.assertFalse(const.fit_par)

    def test_reg_loss_zero(self):
        """Test that Const returns zero reg loss even with reg."""
        # Const doesn't take reg parameter, so this tests the fit_par=False behavior
        const = Const(jnp.array([1.0]))
        self.assertEqual(const.reg_loss(), 0.0)

    def test_inherits_from_param(self):
        """Test that Const inherits from Param."""
        const = Const(jnp.array([1.0]))
        self.assertIsInstance(const, Param)


if __name__ == '__main__':
    unittest.main()
