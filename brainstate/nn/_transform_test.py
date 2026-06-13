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
import unittest

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from brainstate.nn import (
    IdentityT,
    SigmoidT,
    SoftplusT,
    NegSoftplusT,
    AffineT,
    ChainT,
    MaskedT,
    LogT,
    ExpT,
    TanhT,
    SoftsignT,
    PositiveT,
    NegativeT,
    ScaledSigmoidT,
    PowerT,
    OrderedT,
    SimplexT,
    UnitVectorT,
    ReluT,
    ClipT,
)
from brainstate.nn._transform import save_exp


class TestSaveExp(unittest.TestCase):
    def test_save_exp_clipping(self):
        large = 1000.0
        out = save_exp(large)
        np.testing.assert_allclose(out, np.exp(20.0), rtol=1e-6)

    def test_save_exp_regular(self):
        x = jnp.array([-2.0, 0.0, 2.0])
        out = save_exp(x)
        np.testing.assert_allclose(out, np.exp(np.array(x)), rtol=1e-6)


class TestIdentityTTransform(unittest.TestCase):
    def test_roundtrip(self):
        t = IdentityT()
        x = jnp.array([-3.0, 0.0, 4.0])
        y = t.forward(x)
        xr = t.inverse(y)
        np.testing.assert_allclose(x, xr)


class TestSigmoidTransform(unittest.TestCase):
    def test_forward_inverse_numeric(self):
        t = SigmoidT(0.0, 1.0)
        x = jnp.array([-5.0, 0.0, 5.0])
        y = t.forward(x)
        xr = t.inverse(y)
        np.testing.assert_allclose(x, xr, rtol=1e-5, atol=1e-6)

    def test_unit_roundtrip(self):
        unit = u.mV
        t = SigmoidT(0.0 * unit, 1.0 * unit)
        x = jnp.array([-2.0, 0.0, 2.0])
        y = t.forward(x)
        self.assertTrue(isinstance(y, u.Quantity))
        xr = t.inverse(y)
        np.testing.assert_allclose(x, xr, rtol=1e-5, atol=1e-6)

    def test_range(self):
        t = SigmoidT(-2.0, 3.0)
        y = t.forward(jnp.array([-100.0, 0.0, 100.0]))
        self.assertTrue(np.all(y >= -2.0))
        self.assertTrue(np.all(y <= 3.0))


class TestSoftplusTransforms(unittest.TestCase):
    def test_softplus_roundtrip(self):
        t = SoftplusT(0.0)
        x = jnp.array([-5.0, -1.0, 0.0, 2.0, 5.0])
        y = t.forward(x)
        xr = t.inverse(y)
        np.testing.assert_allclose(x, xr, rtol=1e-5, atol=1e-6)

    def test_negsoftplus_roundtrip(self):
        t = NegSoftplusT(0.0)
        x = jnp.array([-5.0, -1.0, 0.0, 2.0, 5.0])
        y = t.forward(x)
        xr = t.inverse(y)
        np.testing.assert_allclose(x, xr, rtol=1e-5, atol=1e-6)


class TestAffineTransform(unittest.TestCase):
    def test_forward_inverse(self):
        t = AffineT(2.5, -3.0)
        x = jnp.array([-2.0, 0.0, 1.2])
        y = t.forward(x)
        xr = t.inverse(y)
        np.testing.assert_allclose(x, xr)

    def test_invalid_scale_raises(self):
        with self.assertRaises(ValueError):
            _ = AffineT(0.0, 1.0)


class TestLogExpTransform(unittest.TestCase):
    def test_log_transform_roundtrip_units(self):
        lower = 1.0 * u.mV
        t = LogT(lower)
        x = jnp.array([-3.0, 0.0, 3.0])
        y = t.forward(x)
        self.assertTrue(isinstance(y, u.Quantity))
        xr = t.inverse(y)
        np.testing.assert_allclose(x, xr, rtol=1e-5, atol=1e-6)

    def test_exp_transform_equivalent(self):
        lower = 0.5 * u.mV
        t1 = LogT(lower)
        t2 = ExpT(lower)
        x = jnp.array([-2.0, 0.5, 2.0])
        y1 = t1.forward(x)
        y2 = t2.forward(x)
        assert u.math.allclose(y1, y2)
        xr1 = t1.inverse(y1)
        xr2 = t2.inverse(y2)
        np.testing.assert_allclose(xr1, xr2)


class TestTanhSoftsignTransform(unittest.TestCase):
    def test_tanh_roundtrip_and_range(self):
        t = TanhT(-2.0, 5.0)
        x = jnp.array([-4.0, 0.0, 4.0])
        y = t.forward(x)
        self.assertTrue(np.all(y > -2.0))
        self.assertTrue(np.all(y < 5.0))
        xr = t.inverse(y)
        np.testing.assert_allclose(x, xr, rtol=1e-2, atol=1e-2)

    def test_softsign_roundtrip_and_range(self):
        t = SoftsignT(-1.0, 2.0)
        x = jnp.array([-4.0, 0.0, 4.0])
        y = t.forward(x)
        self.assertTrue(np.all(y > -1.0))
        self.assertTrue(np.all(y < 2.0))
        xr = t.inverse(y)
        np.testing.assert_allclose(x, xr, rtol=1e-5, atol=1e-6)


class TestChainTransform(unittest.TestCase):
    def test_chain_roundtrip(self):
        # Map R -> (0,1) then affine to (-1,1)
        sigmoid = SigmoidT(0.0, 1.0)
        affine = AffineT(2.0, -1.0)
        chain = ChainT(sigmoid, affine)
        x = jnp.array([-3.0, 0.0, 3.0])
        y = chain.forward(x)
        self.assertTrue(np.all(y > -1.0))
        self.assertTrue(np.all(y < 1.0))
        xr = chain.inverse(y)
        np.testing.assert_allclose(x, xr, rtol=1e-5, atol=1e-6)


class TestMaskedTransform(unittest.TestCase):
    def test_masked_forward_inverse(self):
        mask = jnp.array([False, True, False, True])
        base = SoftplusT(0.0)
        t = MaskedT(mask, base)
        x = jnp.array([-1.0, -1.0, 2.0, 2.0])
        y = t.forward(x)
        # Unmasked indices unchanged
        np.testing.assert_allclose(y[0], x[0])
        np.testing.assert_allclose(y[2], x[2])
        # Masked indices transformed (softplus(x) >= 0)
        self.assertTrue(y[1] >= 0.0 and y[3] >= 0.0)
        xr = t.inverse(y)
        np.testing.assert_allclose(x, xr, rtol=1e-5, atol=1e-6)


class TestPositiveTransform(unittest.TestCase):
    def test_positive_roundtrip(self):
        t = PositiveT()
        x = jnp.array([-3.0, -1.0, 0.0, 1.0, 3.0])
        y = t.forward(x)
        # Check all outputs are positive
        self.assertTrue(np.all(y > 0))
        xr = t.inverse(y)
        np.testing.assert_allclose(x, xr, rtol=1e-5, atol=1e-6)

    def test_positive_repr(self):
        t = PositiveT()
        self.assertEqual(repr(t), "PositiveT()")


class TestNegativeTransform(unittest.TestCase):
    def test_negative_roundtrip(self):
        t = NegativeT()
        x = jnp.array([-3.0, -1.0, 0.0, 1.0, 3.0])
        y = t.forward(x)
        # Check all outputs are negative
        self.assertTrue(np.all(y < 0))
        xr = t.inverse(y)
        np.testing.assert_allclose(x, xr, rtol=1e-5, atol=1e-6)

    def test_negative_repr(self):
        t = NegativeT()
        self.assertEqual(repr(t), "NegativeT()")


class TestScaledSigmoidTransform(unittest.TestCase):
    def test_scaled_sigmoid_roundtrip(self):
        t = ScaledSigmoidT(0.0, 1.0, beta=2.0)
        x = jnp.array([-3.0, 0.0, 3.0])
        y = t.forward(x)
        # Check range
        self.assertTrue(np.all(y > 0.0))
        self.assertTrue(np.all(y < 1.0))
        xr = t.inverse(y)
        np.testing.assert_allclose(x, xr, rtol=1e-5, atol=1e-6)

    def test_scaled_sigmoid_beta_effect(self):
        t_normal = ScaledSigmoidT(0.0, 1.0, beta=1.0)
        t_sharp = ScaledSigmoidT(0.0, 1.0, beta=5.0)
        x = jnp.array([0.5])
        y_normal = t_normal.forward(x)
        y_sharp = t_sharp.forward(x)
        # Sharper sigmoid should be closer to 1 for positive x
        self.assertTrue(y_sharp[0] > y_normal[0])

    def test_scaled_sigmoid_repr(self):
        t = ScaledSigmoidT(0.0, 1.0, beta=2.0)
        self.assertIn("ScaledSigmoidT", repr(t))
        self.assertIn("beta=2.0", repr(t))


class TestPowerTransform(unittest.TestCase):
    def test_power_roundtrip(self):
        t = PowerT(lmbda=0.5)
        x = jnp.array([0.1, 1.0, 4.0, 9.0])
        y = t.forward(x)
        xr = t.inverse(y)
        np.testing.assert_allclose(x, xr, rtol=1e-5, atol=1e-6)

    def test_power_lambda_one(self):
        # Lambda = 1 should be close to identity (shifted)
        t = PowerT(lmbda=1.0)
        x = jnp.array([1.0, 2.0, 3.0])
        y = t.forward(x)
        # y = (x^1 - 1) / 1 = x - 1
        np.testing.assert_allclose(y, x - 1, rtol=1e-5)

    def test_power_repr(self):
        t = PowerT(lmbda=0.5)
        self.assertEqual(repr(t), "PowerT(lmbda=0.5)")


class TestOrderedTransform(unittest.TestCase):
    def test_ordered_roundtrip(self):
        t = OrderedT()
        x = jnp.array([0.0, 1.0, -0.5, 2.0])
        y = t.forward(x)
        xr = t.inverse(y)
        np.testing.assert_allclose(x, xr, rtol=1e-5, atol=1e-6)

    def test_ordered_monotonic(self):
        t = OrderedT()
        x = jnp.array([0.0, 1.0, -0.5, 2.0])
        y = t.forward(x)
        # Check monotonically increasing
        diffs = jnp.diff(y)
        self.assertTrue(np.all(diffs > 0))

    def test_ordered_repr(self):
        t = OrderedT()
        self.assertEqual(repr(t), "OrderedT()")


class TestSimplexTransform(unittest.TestCase):
    def test_simplex_roundtrip(self):
        t = SimplexT()
        x = jnp.array([0.0, 1.0, -1.0])  # 3D input -> 4D simplex
        y = t.forward(x)
        xr = t.inverse(y)
        np.testing.assert_allclose(x, xr, rtol=1e-4, atol=1e-5)

    def test_simplex_sums_to_one(self):
        t = SimplexT()
        x = jnp.array([0.5, -0.5, 1.0])
        y = t.forward(x)
        np.testing.assert_allclose(jnp.sum(y), 1.0, rtol=1e-6)

    def test_simplex_all_positive(self):
        t = SimplexT()
        x = jnp.array([2.0, -2.0, 0.0])
        y = t.forward(x)
        self.assertTrue(np.all(y > 0))

    def test_simplex_repr(self):
        t = SimplexT()
        self.assertEqual(repr(t), "SimplexT()")


class TestUnitVectorTransform(unittest.TestCase):
    def test_unit_vector_norm(self):
        t = UnitVectorT()
        x = jnp.array([3.0, 4.0])
        y = t.forward(x)
        np.testing.assert_allclose(jnp.linalg.norm(y), 1.0, rtol=1e-6)

    def test_unit_vector_direction(self):
        t = UnitVectorT()
        x = jnp.array([3.0, 4.0])
        y = t.forward(x)
        # Direction should be preserved
        np.testing.assert_allclose(y, jnp.array([0.6, 0.8]), rtol=1e-6)

    def test_unit_vector_repr(self):
        t = UnitVectorT()
        self.assertEqual(repr(t), "UnitVectorT()")


class TestLogAbsDetJacobian(unittest.TestCase):
    def test_identity_jacobian(self):
        t = IdentityT()
        x = jnp.array([1.0, 2.0, 3.0])
        y = t.forward(x)
        ladj = t.log_abs_det_jacobian(x, y)
        np.testing.assert_allclose(ladj, 0.0, rtol=1e-6)

    def test_exp_jacobian(self):
        t = ExpT(0.0)
        x = jnp.array([0.0, 1.0, 2.0])
        y = t.forward(x)
        ladj = t.log_abs_det_jacobian(x, y)
        # d/dx exp(x) = exp(x), log det = sum(x)
        np.testing.assert_allclose(ladj, jnp.sum(x), rtol=1e-5)

    def test_positive_jacobian(self):
        t = PositiveT()
        x = jnp.array([0.0, 1.0, 2.0])
        y = t.forward(x)
        ladj = t.log_abs_det_jacobian(x, y)
        np.testing.assert_allclose(ladj, jnp.sum(x), rtol=1e-5)

    def test_affine_jacobian(self):
        t = AffineT(2.0, 1.0)
        x = jnp.array([1.0, 2.0, 3.0])
        y = t.forward(x)
        ladj = t.log_abs_det_jacobian(x, y)
        # d/dx (2x + 1) = 2, log det = 3 * log(2)
        np.testing.assert_allclose(ladj, 3 * jnp.log(2.0), rtol=1e-5)


class TestTransformRepr(unittest.TestCase):
    def test_sigmoid_repr(self):
        t = SigmoidT(0.0, 1.0)
        self.assertIn("Sigmoid", repr(t))

    def test_softplus_repr(self):
        t = SoftplusT(0.0)
        self.assertEqual(repr(t), "SoftplusT(lower=0.0)")

    def test_chain_repr(self):
        t = ChainT(SigmoidT(0.0, 1.0), AffineT(2.0, 0.0))
        r = repr(t)
        self.assertIn("ChainT", r)
        self.assertIn("SigmoidT", r)
        self.assertIn("AffineT", r)

    def test_masked_repr(self):
        mask = jnp.array([True, False])
        t = MaskedT(mask, SoftplusT(0.0))
        r = repr(t)
        self.assertIn("MaskedT", r)
        self.assertIn("SoftplusT", r)


class TestReluTransform(unittest.TestCase):
    def test_relu_forward(self):
        t = ReluT(lower_bound=0.0)
        x = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        y = t.forward(x)
        expected = jnp.array([0.0, 0.0, 0.0, 1.0, 2.0])
        np.testing.assert_allclose(y, expected, rtol=1e-6)

    def test_relu_with_lower_bound(self):
        t = ReluT(lower_bound=1.0)
        x = jnp.array([-2.0, 0.0, 2.0])
        y = t.forward(x)
        # relu(x) + lower_bound
        expected = jnp.array([1.0, 1.0, 3.0])
        np.testing.assert_allclose(y, expected, rtol=1e-6)

    def test_relu_inverse(self):
        t = ReluT(lower_bound=0.0)
        x = jnp.array([0.0, 1.0, 2.0])
        y = t.forward(x)
        xr = t.inverse(y)
        # Inverse only recovers non-negative inputs correctly
        np.testing.assert_allclose(xr, jnp.array([0.0, 1.0, 2.0]), rtol=1e-6)

    def test_relu_output_positive(self):
        t = ReluT(lower_bound=0.0)
        x = jnp.array([-10.0, -5.0, 0.0, 5.0, 10.0])
        y = t.forward(x)
        self.assertTrue(np.all(y >= 0.0))


class TestClipTransform(unittest.TestCase):
    def test_clip_forward(self):
        t = ClipT(lower=0.0, upper=1.0)
        x = jnp.array([-0.5, 0.5, 1.5])
        y = t.forward(x)
        expected = jnp.array([0.0, 0.5, 1.0])
        np.testing.assert_allclose(y, expected, rtol=1e-6)

    def test_clip_custom_bounds(self):
        t = ClipT(lower=-2.0, upper=3.0)
        x = jnp.array([-5.0, 0.0, 5.0])
        y = t.forward(x)
        expected = jnp.array([-2.0, 0.0, 3.0])
        np.testing.assert_allclose(y, expected, rtol=1e-6)

    def test_clip_inverse_passthrough(self):
        t = ClipT(lower=0.0, upper=1.0)
        y = jnp.array([0.5])
        xr = t.inverse(y)
        # Inverse returns clipped value (passthrough for in-range values)
        np.testing.assert_allclose(xr, y, rtol=1e-6)

    def test_clip_repr(self):
        t = ClipT(lower=0.0, upper=1.0)
        self.assertIn("ClipT", repr(t))
        self.assertIn("lower=0.0", repr(t))
        self.assertIn("upper=1.0", repr(t))


class TestTransformAuditRegressions(unittest.TestCase):
    """Regression tests for audit findings T2-T5 (transform numerics/units)."""

    def test_softplus_roundtrip_large_constrained_value(self):
        # T2/T3: the old log1p(save_exp(x)) forward clipped at exp(20), so large
        # unconstrained inputs saturated and inverse(forward(x)) failed to round-trip.
        t = SoftplusT(0.0)
        x = jnp.array([25.0, 30.0, 50.0])  # all beyond the old save_exp clip (20)
        y = t.forward(x)
        # forward should be ~identity for large x (softplus(x) ≈ x), not clipped.
        np.testing.assert_allclose(u.get_mantissa(y), np.asarray(x), rtol=1e-4)
        xr = t.inverse(y)
        np.testing.assert_allclose(np.asarray(xr), np.asarray(x), rtol=1e-4)

    def test_negsoftplus_roundtrip_large_value(self):
        # T2/T3: same saturation bug in the reflected transform.
        t = NegSoftplusT(0.0)
        x = jnp.array([25.0, 40.0])
        y = t.forward(x)
        xr = t.inverse(y)
        np.testing.assert_allclose(np.asarray(xr), np.asarray(x), rtol=1e-4)

    def test_softplus_roundtrip_with_units(self):
        # T4: dividing out the unit yields a dimensionless Quantity; inverse must
        # strip it so bare jnp.expm1/log do not choke on a Quantity.
        t = SoftplusT(0.0 * u.mV)
        x = jnp.array([-3.0, 0.0, 5.0, 25.0])
        y = t.forward(x)
        self.assertEqual(u.get_unit(y), u.mV)
        xr = t.inverse(y)
        np.testing.assert_allclose(np.asarray(xr), np.asarray(x), rtol=1e-4)

    def test_sigmoid_log_abs_det_jacobian_with_units(self):
        # T4: log_abs_det_jacobian crashed when bounds carried units because
        # jnp.log was applied to a Quantity width. It must now run and be finite.
        t = SigmoidT(0.0 * u.mV, 10.0 * u.mV)
        x = jnp.array([-2.0, 0.0, 3.0])
        ladj = t.log_abs_det_jacobian(x, t.forward(x))
        self.assertTrue(np.all(np.isfinite(np.asarray(ladj))))
        # large |x| must not produce -inf/nan (stable log_sigmoid form).
        ladj_big = t.log_abs_det_jacobian(jnp.array([100.0, -100.0]), None)
        self.assertTrue(np.all(np.isfinite(np.asarray(ladj_big))))

    def test_affine_log_abs_det_jacobian_batch_shape(self):
        # T5: batched input (B, n) must yield a (B,) log-det, not a single scalar.
        t = AffineT(2.0, 1.0)
        x = jnp.ones((4, 3))
        ladj = t.log_abs_det_jacobian(x, t.forward(x))
        self.assertEqual(np.shape(ladj), (4,))
        np.testing.assert_allclose(np.asarray(ladj), 3 * np.log(2.0), rtol=1e-6)

    def test_affine_log_abs_det_jacobian_array_scale(self):
        # T5: per-dimension scale must contract over the event axis as sum(log|a_i|).
        a = jnp.array([2.0, 3.0, 4.0])
        t = AffineT(a, 0.0)
        x = jnp.ones((5, 3))
        ladj = t.log_abs_det_jacobian(x, t.forward(x))
        self.assertEqual(np.shape(ladj), (5,))
        np.testing.assert_allclose(
            np.asarray(ladj), np.sum(np.log(np.abs(np.asarray(a)))), rtol=1e-6
        )

    def test_affine_log_abs_det_jacobian_with_units(self):
        # T4: unit-carrying scale must not crash jnp.log.
        t = AffineT(2.0 * u.mV, 0.0 * u.mV)
        x = jnp.array([1.0, 2.0])
        ladj = t.log_abs_det_jacobian(x, None)
        self.assertTrue(np.all(np.isfinite(np.asarray(ladj))))
        np.testing.assert_allclose(np.asarray(ladj), 2 * np.log(2.0), rtol=1e-6)

    def test_masked_forward_grad_finite_through_masked_positions(self):
        # H11: jnp.where evaluates the inner transform over the WHOLE array, so a
        # masked-out entry could feed an invalid value (here -1.0 into a sqrt) into
        # the transform and yield NaN gradients. The double-where fix substitutes a
        # safe value, keeping the gradient finite everywhere.
        def loss(x):
            return MaskedT(jnp.array([True, False]), PowerT(0.5)).forward(x).sum()

        g = jax.grad(loss)(jnp.array([4.0, -1.0]))
        self.assertTrue(np.all(np.isfinite(np.asarray(g))))

    def test_masked_inverse_grad_finite_through_masked_positions(self):
        # H11: analogous to forward, but for inverse. PowerT(2.0).inverse computes a
        # square root, so a masked-out -1.0 would produce a NaN gradient without the
        # double-where guard.
        def loss(y):
            return MaskedT(jnp.array([True, False]), PowerT(2.0)).inverse(y).sum()

        g = jax.grad(loss)(jnp.array([4.0, -1.0]))
        self.assertTrue(np.all(np.isfinite(np.asarray(g))))

    def test_log_abs_det_jacobian_scalar_input(self):
        # M22: jnp.sum(expr, axis=-1) raises on a 0-d (scalar) array. Each
        # element-wise transform's log_abs_det_jacobian must return a finite scalar
        # of shape () for a 0-d input.
        x = jnp.array(1.5)
        transforms = [
            SigmoidT(0.0, 1.0),
            SoftplusT(0.0),
            LogT(0.0),
            ExpT(0.0),
            PositiveT(),
        ]
        for t in transforms:
            ladj = t.log_abs_det_jacobian(x, t.forward(x))
            self.assertEqual(np.shape(ladj), (), msg=f"{t!r} should give scalar shape")
            self.assertTrue(np.isfinite(np.asarray(ladj)), msg=f"{t!r} should be finite")

    def test_chain_log_abs_det_jacobian_scalar_input(self):
        # M22: a chain of element-wise transforms must also work on a scalar input.
        t = ChainT(SigmoidT(0.0, 1.0), AffineT(2.0, 0.0))
        x = jnp.array(0.3)
        ladj = t.log_abs_det_jacobian(x, t.forward(x))
        self.assertEqual(np.shape(ladj), ())
        self.assertTrue(np.isfinite(np.asarray(ladj)))

    def test_log_abs_det_jacobian_batched_unchanged(self):
        # M22: the scalar guard must not alter batched (1-d) behaviour.
        x = jnp.array([0.0, 1.0, 2.0])
        np.testing.assert_allclose(
            np.asarray(ExpT(0.0).log_abs_det_jacobian(x, None)), np.sum(np.asarray(x)), rtol=1e-6
        )
        np.testing.assert_allclose(
            np.asarray(PositiveT().log_abs_det_jacobian(x, None)), np.sum(np.asarray(x)), rtol=1e-6
        )

    def test_simplex_inverse_at_vertex_finite(self):
        # M23: at a simplex vertex a stick-breaking ratio hits 0 or 1, and the old
        # one-sided "+ 1e-8" denominator nudge let the logit overflow to +/-inf. The
        # clamp into [eps, 1 - eps] must keep the inverse finite on the boundary.
        x = SimplexT().inverse(jnp.array([1.0, 0.0, 0.0, 0.0]))
        self.assertTrue(np.all(np.isfinite(np.asarray(x))))

    def test_simplex_roundtrip_interior(self):
        # M23: on the open simplex (interior) inverse(forward(x)) must still
        # round-trip for moderate x after the clamp change.
        t = SimplexT()
        x = jnp.array([0.5, -0.5, 1.0])
        xr = t.inverse(t.forward(x))
        np.testing.assert_allclose(np.asarray(xr), np.asarray(x), atol=1e-4)


if __name__ == '__main__':
    unittest.main()
