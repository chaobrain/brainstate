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

"""Comprehensive tests for physical-unit support in ``brainstate.random``.

These tests pin down the unit contract of the random distributions:

* **Location/scale distributions** (``normal``, ``laplace``, ``logistic``,
  ``gumbel``, ``wald``, ``truncated_normal``) carry the physical unit of their
  ``loc``/``scale`` (or ``mean``/bounds) parameters into the sampled output.
* **Scale-only distributions** (``exponential``, ``gamma``, ``rayleigh``,
  ``weibull_min``) carry the unit of their ``scale`` parameter.
* **Shape/rate/count/probability parameters are strictly dimensionless** — a
  dimensional :class:`~brainunit.Quantity` raises ``ValueError``.
* Mismatched location/scale units raise :class:`~brainunit.UnitMismatchError`.
* Without any units, every distribution returns a plain array.
"""

import unittest

import jax.numpy as jnp
import numpy as np

import brainstate
import brainunit as u
from brainstate.random._impl import (
    _remove_unit_param,
    _scale_unit,
    _loc_scale_unit,
    _validate_raw_key_data,
)
from brainstate.random._state import RandomState

UNIT = u.mV
SIZE = (200,)


def _rng():
    """A freshly seeded, isolated RandomState for deterministic draws."""
    return RandomState(0)


class TestLocScaleUnitPropagation(unittest.TestCase):
    """``normal``/``laplace``/``logistic``/``gumbel`` carry the loc/scale unit."""

    DISTS = ('normal', 'laplace', 'logistic', 'gumbel')

    def test_both_loc_and_scale_carry_unit(self):
        """When both loc and scale carry the unit, the output carries it too."""
        for name in self.DISTS:
            with self.subTest(distribution=name):
                fn = getattr(_rng(), name)
                out = fn(1.0 * UNIT, 2.0 * UNIT, size=SIZE)
                self.assertIsInstance(out, u.Quantity)
                self.assertEqual(out.unit, UNIT)
                self.assertEqual(out.mantissa.shape, SIZE)

    def test_loc_plain_scale_unit(self):
        """A plain loc is interpreted in the scale's unit."""
        for name in self.DISTS:
            with self.subTest(distribution=name):
                out = getattr(_rng(), name)(0.0, 2.0 * UNIT, size=SIZE)
                self.assertIsInstance(out, u.Quantity)
                self.assertEqual(out.unit, UNIT)

    def test_loc_unit_scale_plain(self):
        """A plain scale is interpreted in the loc's unit."""
        for name in self.DISTS:
            with self.subTest(distribution=name):
                out = getattr(_rng(), name)(1.0 * UNIT, 2.0, size=SIZE)
                self.assertIsInstance(out, u.Quantity)
                self.assertEqual(out.unit, UNIT)

    def test_mismatched_units_raise(self):
        """Incompatible loc/scale dimensions raise ``UnitMismatchError``."""
        for name in self.DISTS:
            with self.subTest(distribution=name):
                with self.assertRaises(u.UnitMismatchError):
                    getattr(_rng(), name)(1.0 * u.mV, 2.0 * u.mA, size=SIZE)

    def test_no_units_returns_plain(self):
        """Without units the output is a plain (non-Quantity) array."""
        for name in self.DISTS:
            with self.subTest(distribution=name):
                out = getattr(_rng(), name)(0.0, 1.0, size=SIZE)
                self.assertNotIsInstance(out, u.Quantity)

    def test_compatible_units_are_converted(self):
        """A scale given in a compatible unit (V) is converted into loc's unit (mV)."""
        out = _rng().normal(0.0 * u.mV, 1.0 * u.volt, size=SIZE)
        self.assertIsInstance(out, u.Quantity)
        self.assertEqual(out.unit, u.mV)


class TestScaleOnlyUnitPropagation(unittest.TestCase):
    """``exponential``/``gamma``/``rayleigh``/``weibull_min`` carry the scale unit."""

    def test_exponential_scale_unit(self):
        out = _rng().exponential(2.0 * UNIT, size=SIZE)
        self.assertIsInstance(out, u.Quantity)
        self.assertEqual(out.unit, UNIT)

    def test_exponential_no_unit(self):
        out = _rng().exponential(2.0, size=SIZE)
        self.assertNotIsInstance(out, u.Quantity)

    def test_exponential_default_scale_is_plain(self):
        """The default scale (None) yields a plain array."""
        out = _rng().exponential(size=SIZE)
        self.assertNotIsInstance(out, u.Quantity)

    def test_gamma_scale_unit_shape_dimensionless(self):
        out = _rng().gamma(2.0, 3.0 * UNIT, size=SIZE)
        self.assertIsInstance(out, u.Quantity)
        self.assertEqual(out.unit, UNIT)

    def test_rayleigh_scale_unit(self):
        out = _rng().rayleigh(2.0 * UNIT, size=SIZE)
        self.assertIsInstance(out, u.Quantity)
        self.assertEqual(out.unit, UNIT)

    def test_weibull_min_scale_unit(self):
        out = _rng().weibull_min(1.5, scale=2.0 * UNIT, size=SIZE)
        self.assertIsInstance(out, u.Quantity)
        self.assertEqual(out.unit, UNIT)

    def test_weibull_min_default_scale_plain(self):
        out = _rng().weibull_min(1.5, size=SIZE)
        self.assertNotIsInstance(out, u.Quantity)


class TestWaldUnitPropagation(unittest.TestCase):
    """``wald`` carries the shared unit of its mean and scale."""

    def test_mean_and_scale_unit(self):
        out = _rng().wald(1.0 * UNIT, 2.0 * UNIT, size=SIZE)
        self.assertIsInstance(out, u.Quantity)
        self.assertEqual(out.unit, UNIT)
        self.assertTrue(bool((out.mantissa > 0).all()))

    def test_mean_unit_scale_plain(self):
        out = _rng().wald(1.0 * UNIT, 2.0, size=SIZE)
        self.assertIsInstance(out, u.Quantity)
        self.assertEqual(out.unit, UNIT)

    def test_mismatched_units_raise(self):
        with self.assertRaises(u.UnitMismatchError):
            _rng().wald(1.0 * u.mV, 2.0 * u.mA, size=SIZE)

    def test_no_units_plain(self):
        out = _rng().wald(1.0, 2.0, size=SIZE)
        self.assertNotIsInstance(out, u.Quantity)


class TestTruncatedNormalUnitPropagation(unittest.TestCase):
    """``truncated_normal`` shares one unit across bounds, loc, and scale."""

    def test_bounds_carry_unit(self):
        out = _rng().truncated_normal(-1.0 * UNIT, 1.0 * UNIT, size=SIZE)
        self.assertIsInstance(out, u.Quantity)
        self.assertEqual(out.unit, UNIT)
        mag = out.mantissa
        self.assertTrue(bool((mag >= -1.0).all() and (mag <= 1.0).all()))

    def test_bounds_loc_scale_units(self):
        out = _rng().truncated_normal(
            -2.0 * UNIT, 2.0 * UNIT, loc=0.0 * UNIT, scale=1.0 * UNIT, size=SIZE
        )
        self.assertIsInstance(out, u.Quantity)
        self.assertEqual(out.unit, UNIT)

    def test_no_units_plain(self):
        out = _rng().truncated_normal(-1.0, 1.0, size=SIZE)
        self.assertNotIsInstance(out, u.Quantity)


class TestVonmisesUnits(unittest.TestCase):
    """``vonmises``: ``kappa`` is strictly dimensionless; ``mu`` defaults unitless."""

    def test_kappa_dimensional_raises(self):
        with self.assertRaises(ValueError):
            _rng().vonmises(0.0, 2.0 * u.mV, size=SIZE)

    def test_default_is_plain(self):
        out = _rng().vonmises(0.0, 2.0, size=SIZE)
        self.assertNotIsInstance(out, u.Quantity)

    def test_radian_mu_collapses_to_plain(self):
        """radian is dimensionless in brainunit, so a radian mu yields a plain array."""
        out = _rng().vonmises(0.0 * u.radian, 2.0, size=SIZE)
        self.assertNotIsInstance(out, u.Quantity)


class TestMultivariateNormalUnits(unittest.TestCase):
    """``multivariate_normal`` carries the unit of its mean (cov is unit squared)."""

    def test_mean_and_cov_units(self):
        mean = jnp.array([0.0, 1.0]) * u.mV
        cov = jnp.array([[1.0, 0.5], [0.5, 2.0]]) * (u.mV ** 2)
        out = _rng().multivariate_normal(mean, cov, size=(5,))
        self.assertIsInstance(out, u.Quantity)
        self.assertEqual(out.unit, u.mV)
        self.assertEqual(out.mantissa.shape, (5, 2))

    def test_no_units_plain(self):
        mean = jnp.array([0.0, 1.0])
        cov = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        out = _rng().multivariate_normal(mean, cov, size=(5,))
        self.assertNotIsInstance(out, u.Quantity)


class TestStrictDimensionlessParameters(unittest.TestCase):
    """Shape / rate / count / probability parameters reject physical units.

    Each entry passes a dimensional :class:`~brainunit.Quantity` to a parameter
    that must be dimensionless; the call must raise ``ValueError`` mentioning the
    parameter name.
    """

    # (distribution name, args, kwargs) where exactly one arg carries u.mV.
    CASES = [
        ('randint', (0, 10 * u.mV), {}),
        ('random_integers', (1, 5 * u.mV), {}),
        ('beta', (2.0 * u.mV, 3.0), {}),
        ('pareto', (2.0 * u.mV,), {}),
        ('poisson', (3.0 * u.mV,), {}),
        ('standard_gamma', (2.0 * u.mV,), {}),
        ('standard_t', (3.0 * u.mV,), {}),
        ('lognormal', (0.0 * u.mV, 1.0), {}),
        ('binomial', (10 * u.mV, 0.5), {}),
        ('chisquare', (3 * u.mV,), {}),
        ('dirichlet', (jnp.array([1.0, 2.0, 3.0]) * u.mV,), {}),
        ('geometric', (0.5 * u.mV,), {}),
        ('multinomial', (10 * u.mV, jnp.array([0.2, 0.3, 0.5])), {}),
        ('bernoulli', (0.5 * u.mV,), {}),
        ('weibull', (2.0 * u.mV,), {}),
        ('negative_binomial', (5 * u.mV, 0.5), {}),
        ('t', (3.0 * u.mV,), {}),
        ('noncentral_chisquare', (3.0 * u.mV, 1.0), {}),
        ('loggamma', (2.0 * u.mV,), {}),
        ('categorical', (jnp.array([0.1, 0.2, 0.7]) * u.mV,), {}),
        ('zipf', (2.0 * u.mV,), {}),
        ('power', (2.0 * u.mV,), {}),
        ('f', (3.0 * u.mV, 5.0), {}),
        ('hypergeometric', (5 * u.mV, 5, 4), {}),
        ('logseries', (0.5 * u.mV,), {}),
        ('noncentral_f', (3.0 * u.mV, 5.0, 1.0), {}),
        # gamma's *shape* is dimensionless (only its scale carries a unit).
        ('gamma', (2.0 * u.mV, 1.0), {}),
        # weibull_min's concentration *a* is dimensionless.
        ('weibull_min', (2.0 * u.mV,), {}),
        # vonmises kappa is dimensionless.
        ('vonmises', (0.0, 2.0 * u.mV), {}),
    ]

    def test_dimensional_parameter_raises(self):
        for name, args, kwargs in self.CASES:
            with self.subTest(distribution=name):
                with self.assertRaises(ValueError):
                    getattr(_rng(), name)(*args, **kwargs)

    def test_unitless_quantity_is_accepted(self):
        """A genuinely dimensionless ``Quantity`` (UNITLESS) is accepted."""
        rng = _rng()
        # poisson rate and bernoulli prob accept dimensionless quantities.
        self.assertEqual(rng.poisson(3.0 * u.UNITLESS, size=SIZE).shape, SIZE)
        self.assertEqual(rng.bernoulli(0.5 * u.UNITLESS, size=SIZE).shape, SIZE)


class TestDimensionlessDistributionsReturnPlain(unittest.TestCase):
    """Distributions with only dimensionless parameters return plain arrays."""

    def test_plain_outputs(self):
        rng = _rng()
        checks = {
            'beta': lambda: rng.beta(2.0, 3.0, size=SIZE),
            'pareto': lambda: rng.pareto(2.0, size=SIZE),
            'poisson': lambda: rng.poisson(3.0, size=SIZE),
            'standard_gamma': lambda: rng.standard_gamma(2.0, size=SIZE),
            'standard_t': lambda: rng.standard_t(3.0, size=SIZE),
            'lognormal': lambda: rng.lognormal(0.0, 1.0, size=SIZE),
            'bernoulli': lambda: rng.bernoulli(0.5, size=SIZE),
            'geometric': lambda: rng.geometric(0.5, size=SIZE),
            'weibull': lambda: rng.weibull(2.0, size=SIZE),
            'loggamma': lambda: rng.loggamma(2.0, size=SIZE),
            'zipf': lambda: rng.zipf(2.0, size=SIZE),
            'power': lambda: rng.power(2.0, size=SIZE),
        }
        for name, fn in checks.items():
            with self.subTest(distribution=name):
                self.assertNotIsInstance(fn(), u.Quantity)


class TestModuleLevelFunctionsCarryUnits(unittest.TestCase):
    """The module-level wrappers honour the same unit contract as the methods."""

    def setUp(self):
        brainstate.random.seed(0)

    def test_normal_carries_unit(self):
        out = brainstate.random.normal(0.0 * UNIT, 1.0 * UNIT, size=SIZE)
        self.assertIsInstance(out, u.Quantity)
        self.assertEqual(out.unit, UNIT)

    def test_exponential_carries_unit(self):
        out = brainstate.random.exponential(2.0 * UNIT, size=SIZE)
        self.assertIsInstance(out, u.Quantity)
        self.assertEqual(out.unit, UNIT)

    def test_poisson_unit_raises(self):
        with self.assertRaises(ValueError):
            brainstate.random.poisson(3.0 * UNIT, size=SIZE)


class TestUnitHelpers(unittest.TestCase):
    """Directly exercise the low-level unit helpers in ``_impl``."""

    # ------------------------------------------------------------------ #
    # _remove_unit_param                                                 #
    # ------------------------------------------------------------------ #

    def test_remove_unit_param_none(self):
        self.assertIsNone(_remove_unit_param('x', None))

    def test_remove_unit_param_plain_passthrough(self):
        arr = jnp.array([1.0, 2.0])
        self.assertIs(_remove_unit_param('x', arr), arr)

    def test_remove_unit_param_unitless_quantity(self):
        out = _remove_unit_param('x', 3.0 * u.UNITLESS)
        self.assertNotIsInstance(out, u.Quantity)
        self.assertEqual(float(out), 3.0)

    def test_remove_unit_param_dimensional_raises(self):
        with self.assertRaises(ValueError):
            _remove_unit_param('rate', 3.0 * u.mV)

    def test_remove_unit_param_dtype_cast(self):
        out = _remove_unit_param('x', [1, 2, 3], dtype=jnp.float32)
        self.assertEqual(out.dtype, jnp.float32)

    # ------------------------------------------------------------------ #
    # _scale_unit                                                        #
    # ------------------------------------------------------------------ #

    def test_scale_unit_none(self):
        mantissa, unit = _scale_unit(None)
        self.assertIsNone(mantissa)
        self.assertEqual(unit, u.UNITLESS)

    def test_scale_unit_plain(self):
        mantissa, unit = _scale_unit(2.0)
        self.assertEqual(unit, u.UNITLESS)

    def test_scale_unit_quantity(self):
        mantissa, unit = _scale_unit(2.0 * u.mV)
        self.assertEqual(unit, u.mV)
        self.assertEqual(float(mantissa), 2.0)

    # ------------------------------------------------------------------ #
    # _loc_scale_unit                                                    #
    # ------------------------------------------------------------------ #

    def test_loc_scale_unit_neither(self):
        loc, scale, unit = _loc_scale_unit(0.0, 1.0)
        self.assertEqual(unit, u.UNITLESS)

    def test_loc_scale_unit_both(self):
        loc, scale, unit = _loc_scale_unit(1.0 * u.mV, 2.0 * u.volt)
        self.assertEqual(unit, u.mV)
        # 2 V == 2000 mV after conversion to loc's unit.
        self.assertAlmostEqual(float(scale), 2000.0, places=3)

    def test_loc_scale_unit_loc_only(self):
        loc, scale, unit = _loc_scale_unit(1.0 * u.mV, 2.0)
        self.assertEqual(unit, u.mV)
        self.assertEqual(float(scale), 2.0)

    def test_loc_scale_unit_scale_only(self):
        loc, scale, unit = _loc_scale_unit(1.0, 2.0 * u.mV)
        self.assertEqual(unit, u.mV)
        self.assertEqual(float(loc), 1.0)

    def test_loc_scale_unit_mismatch_raises(self):
        with self.assertRaises(u.UnitMismatchError):
            _loc_scale_unit(1.0 * u.mV, 2.0 * u.mA)

    # ------------------------------------------------------------------ #
    # _validate_raw_key_data                                             #
    # ------------------------------------------------------------------ #

    def test_validate_raw_key_data_accepts_uint32_pair(self):
        # A valid uint32[2] passes silently.
        self.assertIsNone(_validate_raw_key_data(np.array([1, 2], dtype=np.uint32)))

    def test_validate_raw_key_data_rejects_non_array(self):
        with self.assertRaises(TypeError):
            _validate_raw_key_data(123)

    def test_validate_raw_key_data_rejects_wrong_dtype(self):
        with self.assertRaises(TypeError):
            _validate_raw_key_data(np.array([1, 2], dtype=np.int32))

    def test_validate_raw_key_data_rejects_wrong_size(self):
        with self.assertRaises(TypeError):
            _validate_raw_key_data(np.array([1, 2, 3], dtype=np.uint32))


if __name__ == '__main__':
    unittest.main()
