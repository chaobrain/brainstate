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

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import brainstate
from brainstate.random import _impl


class TestMultinomial(unittest.TestCase):
    """Validate the JAX-native ``multinomial`` helper in ``_impl.py``."""

    def setUp(self):
        """Seed the global generator and grab a reusable key."""
        brainstate.random.seed(0)
        self.key = brainstate.random.split_key()

    def test_counts_sum_to_n(self):
        """multinomial returns nonneg counts that sum to n."""
        p = jnp.array([0.2, 0.3, 0.5])
        counts = _impl.multinomial(self.key, p, 10, n_max=10)
        self.assertEqual(int(jnp.sum(counts)), 10)
        self.assertTrue(bool(jnp.all(counts >= 0)))

    def test_n_max_zero_returns_zeros(self):
        """multinomial short-circuits to all-zero counts when n_max == 0."""
        p = jnp.array([0.5, 0.5])
        counts = _impl.multinomial(self.key, p, 0, n_max=0)
        self.assertEqual(counts.shape, (2,))
        self.assertTrue(bool(jnp.all(counts == 0)))

    def test_heterogeneous_counts_per_batch(self):
        """multinomial respects a per-batch vector of trial counts n."""
        p = jnp.array([[0.2, 0.3, 0.5], [0.1, 0.1, 0.8]])
        n = jnp.array([10, 5])
        counts = _impl.multinomial(self.key, p, n, n_max=10)
        self.assertEqual(counts.shape, (2, 3))
        np.testing.assert_array_equal(np.asarray(jnp.sum(counts, axis=-1)), [10, 5])

    def test_scalar_n_broadcasts_against_batched_p(self):
        """multinomial broadcasts a scalar n across a batched probability matrix."""
        p = jnp.array([[0.2, 0.3, 0.5], [0.1, 0.1, 0.8]])
        counts = _impl.multinomial(self.key, p, 7, n_max=7)
        self.assertEqual(counts.shape, (2, 3))
        np.testing.assert_array_equal(np.asarray(jnp.sum(counts, axis=-1)), [7, 7])

    def test_explicit_shape_argument(self):
        """multinomial honours an explicit sample ``shape`` argument."""
        p = jnp.array([0.25, 0.25, 0.5])
        counts = _impl.multinomial(self.key, p, 4, n_max=4, shape=())
        self.assertEqual(counts.shape, (3,))
        self.assertEqual(int(jnp.sum(counts)), 4)


class TestVonMisesCentered(unittest.TestCase):
    """Validate the ``von_mises_centered`` rejection sampler in ``_impl.py``."""

    def setUp(self):
        """Seed the global generator and grab a reusable key."""
        brainstate.random.seed(1)
        self.key = brainstate.random.split_key()

    def test_float32_samples_within_pi(self):
        """von_mises_centered returns float32 samples bounded by +/-pi."""
        out = _impl.von_mises_centered(self.key, 2.0, (16,), dtype=jnp.float32)
        self.assertEqual(out.shape, (16,))
        self.assertTrue(bool(jnp.all(jnp.abs(out) <= jnp.pi + 1e-4)))

    def test_default_shape_from_concentration(self):
        """von_mises_centered derives its shape from the concentration when shape is empty."""
        concentration = jnp.array([0.5, 1.0, 4.0])
        out = _impl.von_mises_centered(self.key, concentration, (), dtype=jnp.float32)
        self.assertEqual(out.shape, (3,))

    def test_float64_dtype_branch(self):
        """von_mises_centered accepts the float64 dtype cutoff branch."""
        out = _impl.von_mises_centered(self.key, 2.0, (8,), dtype=jnp.float64)
        self.assertEqual(out.shape, (8,))
        self.assertTrue(bool(jnp.all(jnp.abs(out) <= jnp.pi + 1e-3)))

    def test_unsupported_dtype_raises(self):
        """von_mises_centered raises ValueError for an unsupported dtype."""
        with self.assertRaises(ValueError):
            _impl.von_mises_centered(self.key, 2.0, (4,), dtype=jnp.int32)

    def test_float16_dtype_branch(self):
        """von_mises_centered supports the float16 dtype cutoff branch."""
        out = _impl.von_mises_centered(self.key, 2.0, (4,), dtype=jnp.float16)
        self.assertEqual(out.shape, (4,))
        self.assertEqual(out.dtype, jnp.float16)

    def test_zero_concentration_is_finite_uniform(self):
        """concentration==0 yields finite samples on [-pi, pi] (the uniform
        limiting case), not all-NaN. Regression test for M24."""
        out = _impl.von_mises_centered(self.key, 0.0, (1000,), dtype=jnp.float32)
        self.assertEqual(out.shape, (1000,))
        self.assertFalse(bool(jnp.any(jnp.isnan(out))))
        self.assertTrue(bool(jnp.all(jnp.abs(out) <= jnp.pi + 1e-4)))
        # Empirically approximately uniform on [-pi, pi]: mean near 0.
        self.assertLess(float(jnp.abs(jnp.mean(out))), 0.2)

    def test_negative_concentration_is_finite_uniform(self):
        """concentration<0 also maps to the uniform limiting case (finite,
        bounded), rather than producing NaN. Regression test for M24."""
        out = _impl.von_mises_centered(self.key, -1.0, (256,), dtype=jnp.float32)
        self.assertFalse(bool(jnp.any(jnp.isnan(out))))
        self.assertTrue(bool(jnp.all(jnp.abs(out) <= jnp.pi + 1e-4)))

    def test_mixed_zero_and_positive_concentration(self):
        """A broadcast concentration mixing 0 with a positive value keeps the
        positive entry finite and the zero entry finite (NaN does not leak
        across the jnp.where mask). Regression test for M24."""
        concentration = jnp.array([0.0, 2.0])
        out = _impl.von_mises_centered(self.key, concentration, (2,), dtype=jnp.float32)
        self.assertFalse(bool(jnp.any(jnp.isnan(out))))
        self.assertTrue(bool(jnp.all(jnp.abs(out) <= jnp.pi + 1e-4)))

    def test_public_vonmises_zero_kappa_not_nan(self):
        """RandomState.vonmises(mu, 0.0) returns finite uniform angles instead
        of all-NaN. End-to-end regression test for M24."""
        rs = brainstate.random.RandomState(42)
        out = rs.vonmises(0.0, 0.0, size=(10,))
        out = jnp.asarray(out)
        self.assertFalse(bool(jnp.any(jnp.isnan(out))))
        self.assertTrue(bool(jnp.all(jnp.abs(out) <= jnp.pi + 1e-4)))


class TestReshapeAndPromote(unittest.TestCase):
    """Validate the ``_reshape`` and ``_promote_shapes`` array utilities."""

    def test_reshape_python_scalar_returns_numpy(self):
        """_reshape converts a plain python scalar through numpy."""
        out = _impl._reshape(5, (1,))
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.shape, (1,))

    def test_reshape_numpy_array_stays_numpy(self):
        """_reshape keeps a numpy array on the numpy path."""
        out = _impl._reshape(np.array([1, 2]), (2,))
        self.assertIsInstance(out, np.ndarray)

    def test_reshape_jax_array_uses_jnp(self):
        """_reshape routes a jax array through jnp.reshape."""
        out = _impl._reshape(jnp.array([1, 2]), (2,))
        self.assertEqual(out.shape, (2,))
        self.assertNotIsInstance(out, np.ndarray)

    def test_promote_shapes_single_arg_no_shape(self):
        """_promote_shapes returns inputs untouched for <2 args with no shape."""
        arg = jnp.array([1, 2])
        out = _impl._promote_shapes(arg)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].shape, (2,))

    def test_promote_shapes_multiple_args(self):
        """_promote_shapes left-pads ranks across multiple arguments."""
        out = _impl._promote_shapes(jnp.array([1, 2]), jnp.array([[1, 2], [3, 4]]))
        self.assertEqual(out[0].shape, (1, 2))
        self.assertEqual(out[1].shape, (2, 2))

    def test_promote_shapes_with_shape_keyword(self):
        """_promote_shapes left-pads a single argument when shape is supplied."""
        out = _impl._promote_shapes(jnp.array([1, 2]), shape=(3, 2))
        self.assertEqual(out[0].shape, (1, 2))


class TestDtypeHelpers(unittest.TestCase):
    """Validate the ``_dtype``, ``_is_python_scalar`` and ``const`` helpers."""

    def test_dtype_none_raises(self):
        """_dtype raises ValueError when given None."""
        with self.assertRaises(ValueError):
            _impl._dtype(None)

    def test_dtype_from_python_type(self):
        """_dtype maps a python scalar *type* to its numpy dtype."""
        self.assertEqual(_impl._dtype(int), np.dtype('int64'))

    def test_dtype_from_python_scalar_instance(self):
        """_dtype maps a python scalar *instance* to its numpy dtype."""
        self.assertEqual(_impl._dtype(3.0), np.dtype('float64'))

    def test_dtype_from_object_with_dtype_attr(self):
        """_dtype reads the dtype attribute of an array-like object."""
        self.assertEqual(_impl._dtype(jnp.array([1.0], dtype=jnp.float32)), jnp.float32)

    def test_dtype_via_result_type_fallback(self):
        """_dtype falls back to np.result_type for unknown inputs."""
        self.assertEqual(_impl._dtype('float32'), np.dtype('float32'))

    def test_dtype_canonicalize(self):
        """_dtype canonicalizes float64 to float32 when X64 is disabled."""
        self.assertEqual(_impl._dtype(3.0, canonicalize=True), np.dtype('float32'))

    def test_is_python_scalar_for_zero_dim(self):
        """_is_python_scalar is True for a zero-dimensional value."""
        self.assertTrue(_impl._is_python_scalar(np.array(3.0)))
        self.assertTrue(_impl._is_python_scalar(5))

    def test_is_python_scalar_for_complex(self):
        """_is_python_scalar is True for a python complex scalar."""
        self.assertTrue(_impl._is_python_scalar(complex(1, 2)))

    def test_is_python_scalar_false_for_array(self):
        """_is_python_scalar is False for a multi-element numpy array."""
        # A numpy array exposes no ``aval`` and has ndim>0, reaching the final branch.
        self.assertFalse(_impl._is_python_scalar(np.array([1, 2])))

    def test_const_from_python_float_scalar(self):
        """const builds a value matching a python float example."""
        out = _impl.const(3.0, 5)
        self.assertEqual(float(out), 5.0)

    def test_const_from_python_int_scalar(self):
        """const builds a value matching a python int example."""
        out = _impl.const(3, 5)
        self.assertEqual(int(out), 5)

    def test_const_from_array_example(self):
        """const builds a numpy value matching an array example's dtype."""
        out = _impl.const(jnp.array([1.0, 2.0], dtype=jnp.float32), 7)
        self.assertEqual(np.asarray(out).dtype, np.float32)


class TestFormalizeKey(unittest.TestCase):
    """Validate ``formalize_key`` key-normalisation across input kinds.

    Under the typed-key contract, ``formalize_key`` always returns a JAX
    *typed* PRNG key (scalar shape ``()`` with a ``prng_key`` dtype), never a
    raw ``uint32[2]`` array.
    """

    def setUp(self):
        """Seed the global generator and obtain a (typed) key."""
        brainstate.random.seed(0)
        self.typed_key = brainstate.random.split_key()
        self.raw_key = jax.random.key_data(self.typed_key)

    def _assert_typed(self, key):
        """Assert ``key`` is a scalar typed PRNG key."""
        self.assertEqual(key.shape, ())
        self.assertTrue(jnp.issubdtype(key.dtype, jax.dtypes.prng_key))

    def test_int_seed_becomes_typed_key(self):
        """formalize_key turns an int seed into a typed PRNG key."""
        self._assert_typed(_impl.formalize_key(7))

    def test_raw_uint32_key_wrapped(self):
        """formalize_key wraps a two-element uint32 key into a typed key."""
        out = _impl.formalize_key(self.raw_key)
        self._assert_typed(out)
        # Wrapping is the lossless inverse of key_data.
        np.testing.assert_array_equal(
            np.asarray(jax.random.key_data(out)), np.asarray(self.raw_key)
        )

    def test_typed_prng_key_passthrough(self):
        """formalize_key returns a typed prng key unchanged."""
        typed = _impl.formalize_key(0)
        out = _impl.formalize_key(typed)
        self._assert_typed(out)
        self.assertEqual(out.dtype, typed.dtype)

    def test_scalar_integer_array_becomes_typed_key(self):
        """formalize_key turns a zero-dim integer array into a typed key."""
        self._assert_typed(_impl.formalize_key(np.array(42, dtype=np.int32)))

    def test_wrong_dtype_array_raises(self):
        """formalize_key rejects a float array that is not a valid key."""
        with self.assertRaises(TypeError):
            _impl.formalize_key(np.array([1.0, 2.0], dtype=np.float32))

    def test_wrong_size_array_raises(self):
        """formalize_key rejects a uint32 array that is not length two."""
        with self.assertRaises(TypeError):
            _impl.formalize_key(np.array([1, 2, 3], dtype=np.uint32))

    def test_unsupported_type_raises(self):
        """formalize_key rejects an unsupported python type."""
        with self.assertRaises(TypeError):
            _impl.formalize_key("not-a-key")

    def test_length_one_integer_array(self):
        """formalize_key accepts a length-one integer array seed."""
        self._assert_typed(_impl.formalize_key(np.array([42], dtype=np.int32)))


class TestShapeAndScaleHelpers(unittest.TestCase):
    """Validate the ``_size2shape``, ``_check_shape`` and ``_loc_scale`` helpers."""

    def test_size2shape_none(self):
        """_size2shape maps None to an empty shape tuple."""
        self.assertEqual(_impl._size2shape(None), ())

    def test_size2shape_sequence(self):
        """_size2shape converts a list/tuple to a tuple shape."""
        self.assertEqual(_impl._size2shape([2, 3]), (2, 3))

    def test_size2shape_int(self):
        """_size2shape wraps a bare int in a one-tuple."""
        self.assertEqual(_impl._size2shape(4), (4,))

    def test_check_shape_no_param_shapes(self):
        """_check_shape is a no-op when no parameter shapes are given."""
        self.assertIsNone(_impl._check_shape('x', (3,)))

    def test_check_shape_compatible(self):
        """_check_shape accepts broadcast-compatible parameter shapes."""
        self.assertIsNone(_impl._check_shape('x', (3,), (3,)))

    def test_check_shape_incompatible_raises(self):
        """_check_shape raises ValueError when broadcasting is impossible."""
        with self.assertRaises(ValueError):
            _impl._check_shape('x', (3,), (4,))

    def test_check_shape_result_mismatch_raises(self):
        """_check_shape raises ValueError when the broadcast result exceeds the shape."""
        # (3,) and (1, 3) broadcast to (1, 3), which differs from the (3,) argument.
        with self.assertRaises(ValueError):
            _impl._check_shape('x', (3,), (1, 3))

    def test_loc_scale_no_loc_no_scale(self):
        """_loc_scale returns the raw value when loc and scale are None."""
        self.assertEqual(_impl._loc_scale(None, None, 5.0), 5.0)

    def test_loc_scale_scale_only(self):
        """_loc_scale multiplies by scale when loc is None."""
        self.assertEqual(_impl._loc_scale(None, 2.0, 5.0), 10.0)

    def test_loc_scale_loc_only(self):
        """_loc_scale adds loc when scale is None."""
        self.assertEqual(_impl._loc_scale(1.0, None, 5.0), 6.0)

    def test_loc_scale_loc_and_scale(self):
        """_loc_scale scales then shifts when both are provided."""
        self.assertEqual(_impl._loc_scale(1.0, 2.0, 5.0), 11.0)

    def test_check_py_seq_converts_sequence(self):
        """_check_py_seq converts a list to an array but leaves arrays alone."""
        arr = _impl._check_py_seq([1, 2, 3])
        self.assertEqual(np.asarray(arr).tolist(), [1, 2, 3])
        passthrough = jnp.array([4, 5])
        self.assertIs(_impl._check_py_seq(passthrough), passthrough)


class TestFDistribution(unittest.TestCase):
    """Validate the central F-distribution sampler ``f``."""

    def setUp(self):
        """Seed the global generator and grab a reusable key."""
        brainstate.random.seed(2)
        self.key = brainstate.random.split_key()

    def test_shape_none_broadcasts_params(self):
        """f infers a scalar shape from scalar parameters when shape is None."""
        out = _impl.f(self.key, 3.0, 5.0, shape=None)
        self.assertEqual(out.shape, ())

    def test_shape_int_is_wrapped(self):
        """f wraps an integer shape into a one-tuple."""
        out = _impl.f(self.key, 3.0, 5.0, shape=4)
        self.assertEqual(out.shape, (4,))

    def test_shape_tuple(self):
        """f honours an explicit tuple shape and returns positive samples."""
        out = _impl.f(self.key, 3.0, 5.0, shape=(2, 3))
        self.assertEqual(out.shape, (2, 3))
        self.assertTrue(bool(jnp.all(out > 0)))

    def test_empty_shape_returns_empty(self):
        """f returns an empty array for a zero-sized shape."""
        out = _impl.f(self.key, 3.0, 5.0, shape=(0,))
        self.assertEqual(out.shape, (0,))


class TestNoncentralFDistribution(unittest.TestCase):
    """Validate the noncentral F-distribution sampler ``noncentral_f``."""

    def setUp(self):
        """Seed the global generator and grab a reusable key."""
        brainstate.random.seed(3)
        self.key = brainstate.random.split_key()

    def test_basic_shape(self):
        """noncentral_f returns positive samples of the requested shape."""
        out = _impl.noncentral_f(self.key, 3.0, 5.0, 1.0, shape=(8,))
        self.assertEqual(out.shape, (8,))
        self.assertTrue(bool(jnp.all(out > 0)))

    def test_scalar_shape(self):
        """noncentral_f supports a scalar (empty) output shape."""
        out = _impl.noncentral_f(self.key, 3.0, 5.0, 1.0, shape=())
        self.assertEqual(out.shape, ())

    def test_small_dfnum_branch(self):
        """noncentral_f exercises the dfnum<=1 noncentral-chisquare branch."""
        out = _impl.noncentral_f(self.key, 0.5, 5.0, 1.0, shape=(8,))
        self.assertEqual(out.shape, (8,))
        self.assertTrue(bool(jnp.all(out >= 0)))


class TestLogseriesDistribution(unittest.TestCase):
    """Validate the logarithmic-series sampler ``logseries``."""

    def setUp(self):
        """Seed the global generator and grab a reusable key."""
        brainstate.random.seed(4)
        self.key = brainstate.random.split_key()

    def test_shape_none_from_param(self):
        """logseries infers shape from the parameter when shape is None."""
        out = _impl.logseries(self.key, 0.5, shape=None)
        self.assertEqual(out.shape, ())

    def test_shape_int(self):
        """logseries wraps an integer shape and yields counts >= 1."""
        out = _impl.logseries(self.key, 0.5, shape=3)
        self.assertEqual(out.shape, (3,))
        self.assertTrue(bool(jnp.all(out >= 1)))

    def test_shape_tuple(self):
        """logseries honours a tuple shape."""
        out = _impl.logseries(self.key, 0.5, shape=(2, 2))
        self.assertEqual(out.shape, (2, 2))

    def test_empty_shape_returns_empty(self):
        """logseries returns an empty array for a zero-sized shape."""
        out = _impl.logseries(self.key, 0.5, shape=(0,))
        self.assertEqual(out.shape, (0,))

    def test_nonpositive_probability_limit_case(self):
        """logseries returns 1 for the p<=0 limit case."""
        out = _impl.logseries(self.key, jnp.array([0.0, 0.5]), shape=(2,))
        self.assertEqual(int(out[0]), 1)


class TestZipfDistribution(unittest.TestCase):
    """Validate the Zipf (zeta) sampler ``zipf``."""

    def setUp(self):
        """Seed the global generator and grab a reusable key."""
        brainstate.random.seed(5)
        self.key = brainstate.random.split_key()

    def test_shape_none_from_param(self):
        """zipf infers shape from the parameter when shape is None."""
        out = _impl.zipf(self.key, 2.0, shape=None)
        self.assertEqual(out.shape, ())

    def test_shape_int(self):
        """zipf wraps an integer shape and yields values >= 1."""
        out = _impl.zipf(self.key, 2.0, shape=3)
        self.assertEqual(out.shape, (3,))
        self.assertTrue(bool(jnp.all(out >= 1)))

    def test_shape_tuple(self):
        """zipf honours a tuple shape."""
        out = _impl.zipf(self.key, 2.0, shape=(2,))
        self.assertEqual(out.shape, (2,))

    def test_empty_shape_returns_empty(self):
        """zipf returns an empty array for a zero-sized shape."""
        out = _impl.zipf(self.key, 2.0, shape=(0,))
        self.assertEqual(out.shape, (0,))


class TestPowerDistribution(unittest.TestCase):
    """Validate the power-distribution sampler ``power``."""

    def setUp(self):
        """Seed the global generator and grab a reusable key."""
        brainstate.random.seed(6)
        self.key = brainstate.random.split_key()

    def test_shape_none_from_param(self):
        """power infers shape from the parameter when shape is None."""
        out = _impl.power(self.key, 2.0, shape=None)
        self.assertEqual(out.shape, ())

    def test_shape_int(self):
        """power wraps an integer shape and samples within the unit interval."""
        out = _impl.power(self.key, 2.0, shape=3)
        self.assertEqual(out.shape, (3,))
        self.assertTrue(bool(jnp.all((out >= 0) & (out <= 1))))

    def test_shape_tuple(self):
        """power honours a tuple shape."""
        out = _impl.power(self.key, 2.0, shape=(2, 2))
        self.assertEqual(out.shape, (2, 2))

    def test_empty_shape_returns_empty(self):
        """power returns an empty array for a zero-sized shape."""
        out = _impl.power(self.key, 2.0, shape=(0,))
        self.assertEqual(out.shape, (0,))


class TestHypergeometricDistribution(unittest.TestCase):
    """Validate the hypergeometric sampler ``hypergeometric``."""

    def setUp(self):
        """Seed the global generator and grab a reusable key."""
        brainstate.random.seed(7)
        self.key = brainstate.random.split_key()

    def test_shape_none_from_params(self):
        """hypergeometric infers shape by broadcasting params when shape is None."""
        out = _impl.hypergeometric(self.key, 5, 5, 4, shape=None)
        self.assertEqual(out.shape, ())

    def test_shape_int(self):
        """hypergeometric wraps an integer shape and yields counts in range."""
        out = _impl.hypergeometric(self.key, 5, 5, 4, shape=3)
        self.assertEqual(out.shape, (3,))
        self.assertTrue(bool(jnp.all((out >= 0) & (out <= 4))))

    def test_shape_tuple(self):
        """hypergeometric honours a tuple shape."""
        out = _impl.hypergeometric(self.key, 5, 5, 4, shape=(4,))
        self.assertEqual(out.shape, (4,))

    def test_empty_shape_returns_empty(self):
        """hypergeometric returns an empty array for a zero-sized shape."""
        out = _impl.hypergeometric(self.key, 5, 5, 4, shape=(0,))
        self.assertEqual(out.shape, (0,))


if __name__ == '__main__':
    unittest.main()
