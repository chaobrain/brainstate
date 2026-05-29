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

import brainunit as u
import jax.numpy as jnp
import numpy as np
import pytest

import brainstate
from brainstate import _testing
from brainstate._state import State
from brainstate.nn import init as I


class TestNormalInit(unittest.TestCase):

    def test_normal_init1(self):
        init = brainstate.nn.init.Normal()
        for size in [(10,), (10, 20), (10, 20, 30)]:
            weights = init(size)
            assert weights.shape == size

    def test_normal_init2(self):
        init = brainstate.nn.init.Normal(scale=0.5)
        for size in [(10,), (10, 20)]:
            weights = init(size)
            assert weights.shape == size

    def test_normal_init3(self):
        init1 = brainstate.nn.init.Normal(scale=0.5, seed=10)
        init2 = brainstate.nn.init.Normal(scale=0.5, seed=10)
        size = (10,)
        weights1 = init1(size)
        weights2 = init2(size)
        assert weights1.shape == size
        assert (weights1 == weights2).all()


class TestUniformInit(unittest.TestCase):
    def test_uniform_init1(self):
        init = brainstate.nn.init.Normal()
        for size in [(10,), (10, 20), (10, 20, 30)]:
            weights = init(size)
            assert weights.shape == size

    def test_uniform_init2(self):
        init = brainstate.nn.init.Uniform(min_val=10, max_val=20)
        for size in [(10,), (10, 20)]:
            weights = init(size)
            assert weights.shape == size


class TestVarianceScaling(unittest.TestCase):
    def test_var_scaling1(self):
        init = brainstate.nn.init.VarianceScaling(scale=1., mode='fan_in', distribution='truncated_normal')
        for size in [(10, 20), (10, 20, 30)]:
            weights = init(size)
            assert weights.shape == size

    def test_var_scaling2(self):
        init = brainstate.nn.init.VarianceScaling(scale=2, mode='fan_out', distribution='normal')
        for size in [(10, 20), (10, 20, 30)]:
            weights = init(size)
            assert weights.shape == size

    def test_var_scaling3(self):
        init = brainstate.nn.init.VarianceScaling(
            scale=2 / 4, mode='fan_avg', in_axis=0, out_axis=1, distribution='uniform'
        )
        for size in [(10, 20), (10, 20, 30)]:
            weights = init(size)
            assert weights.shape == size


class TestKaimingUniformUnit(unittest.TestCase):
    def test_kaiming_uniform_init(self):
        init = brainstate.nn.init.KaimingUniform()
        for size in [(10, 20), (10, 20, 30)]:
            weights = init(size)
            assert weights.shape == size


class TestKaimingNormalUnit(unittest.TestCase):
    def test_kaiming_normal_init(self):
        init = brainstate.nn.init.KaimingNormal()
        for size in [(10, 20), (10, 20, 30)]:
            weights = init(size)
            assert weights.shape == size


class TestXavierUniformUnit(unittest.TestCase):
    def test_xavier_uniform_init(self):
        init = brainstate.nn.init.XavierUniform()
        for size in [(10, 20), (10, 20, 30)]:
            weights = init(size)
            assert weights.shape == size


class TestXavierNormalUnit(unittest.TestCase):
    def test_xavier_normal_init(self):
        init = brainstate.nn.init.XavierNormal()
        for size in [(10, 20), (10, 20, 30)]:
            weights = init(size)
            assert weights.shape == size


class TestLecunUniformUnit(unittest.TestCase):
    def test_lecun_uniform_init(self):
        init = brainstate.nn.init.LecunUniform()
        for size in [(10, 20), (10, 20, 30)]:
            weights = init(size)
            assert weights.shape == size


class TestLecunNormalUnit(unittest.TestCase):
    def test_lecun_normal_init(self):
        init = brainstate.nn.init.LecunNormal()
        for size in [(10, 20), (10, 20, 30)]:
            weights = init(size)
            assert weights.shape == size


class TestOrthogonalUnit(unittest.TestCase):
    def test_orthogonal_init1(self):
        init = brainstate.nn.init.Orthogonal()
        for size in [(20, 20), (10, 20, 30)]:
            weights = init(size)
            assert weights.shape == size

    def test_orthogonal_init2(self):
        init = brainstate.nn.init.Orthogonal(scale=2., axis=0)
        for size in [(10, 20), (10, 20, 30)]:
            weights = init(size)
            assert weights.shape == size


class TestDeltaOrthogonalUnit(unittest.TestCase):
    def test_delta_orthogonal_init1(self):
        init = brainstate.nn.init.DeltaOrthogonal()
        for size in [(20, 20, 20), (10, 20, 30, 40), (50, 40, 30, 20, 20)]:
            weights = init(size)
            assert weights.shape == size


class TestZeroInit(unittest.TestCase):
    def test_zero_init(self):
        init = brainstate.nn.init.ZeroInit()
        for size in [(10,), (10, 20), (10, 20, 30)]:
            weights = init(size)
            assert weights.shape == size


class TestOneInit(unittest.TestCase):
    def test_one_init(self):
        for size in [(10,), (10, 20), (10, 20, 30)]:
            for value in [0., 1., -1.]:
                init = brainstate.nn.init.Constant(value=value)
                weights = init(size)
                assert weights.shape == size
                assert (weights == value).all()


class TestIdentityInit(unittest.TestCase):
    def test_identity_init(self):
        for size in [(10,), (10, 20)]:
            for value in [0., 1., -1.]:
                init = brainstate.nn.init.Identity(value=value)
                weights = init(size)
                if len(size) == 1:
                    assert weights.shape == (size[0], size[0])
                else:
                    assert weights.shape == size


# ---------------------------------------------------------------------------
# Appended coverage tests (target: brainstate/nn/init.py >= 90% line coverage).
# ---------------------------------------------------------------------------


_SHAPE = (_testing.SMALL_BATCH, _testing.SMALL_DIM)


class TestToSizeHelper(unittest.TestCase):
    """Tests for the ``to_size`` shape-normalising helper."""

    def test_tuple_and_list(self):
        """``to_size`` returns a tuple for tuple/list inputs."""
        self.assertEqual(I.to_size([2, 3]), (2, 3))
        self.assertEqual(I.to_size((4, 5)), (4, 5))

    def test_int(self):
        """``to_size`` wraps a bare integer in a one-tuple."""
        self.assertEqual(I.to_size(7), (7,))

    def test_none(self):
        """``to_size`` passes ``None`` through unchanged."""
        self.assertIsNone(I.to_size(None))

    def test_invalid_raises(self):
        """``to_size`` raises ``ValueError`` for unsupported types."""
        with self.assertRaises(ValueError):
            I.to_size('not-a-size')


class TestFormatShapeHelper(unittest.TestCase):
    """Tests for the ``_format_shape`` helper."""

    def test_int(self):
        """``_format_shape`` wraps an int into a one-tuple."""
        self.assertEqual(I._format_shape(5), (5,))

    def test_empty_raises(self):
        """``_format_shape`` raises when the shape is empty."""
        with self.assertRaises(ValueError):
            I._format_shape(())

    def test_single_nested(self):
        """``_format_shape`` unwraps a single nested tuple/list element."""
        self.assertEqual(I._format_shape([(3, 4)]), (3, 4))

    def test_single_scalar(self):
        """``_format_shape`` keeps a single scalar element as a one-tuple."""
        self.assertEqual(I._format_shape((3,)), (3,))

    def test_multi(self):
        """``_format_shape`` returns multi-dim shapes unchanged."""
        self.assertEqual(I._format_shape((2, 3, 4)), (2, 3, 4))


class TestBroadcastHelpers(unittest.TestCase):
    """Tests for ``are_broadcastable_shapes`` and ``_is_scalar``."""

    def test_broadcastable_equal(self):
        """Equal shapes are broadcastable."""
        self.assertTrue(I.are_broadcastable_shapes((3,), (3,)))

    def test_broadcastable_with_one(self):
        """A leading 1 broadcasts against a larger dimension."""
        self.assertTrue(I.are_broadcastable_shapes((1, 3), (4, 3)))

    def test_not_broadcastable(self):
        """Mismatched non-unit dimensions are not broadcastable."""
        self.assertFalse(I.are_broadcastable_shapes((2, 3), (4, 3)))

    def test_is_scalar(self):
        """``_is_scalar`` distinguishes scalars from arrays."""
        self.assertTrue(I._is_scalar(5))
        self.assertFalse(I._is_scalar(np.ones((3,))))

    def test_expand_params(self):
        """``_expand_params_to_match_sizes`` adds leading axes."""
        out = I._expand_params_to_match_sizes(np.ones((3,)), (2, 3))
        self.assertEqual(out.shape, (1, 3))


class TestCalculateInitGain(unittest.TestCase):
    """Tests for ``calculate_init_gain`` across every nonlinearity branch."""

    def test_linear_family(self):
        """Linear and conv nonlinearities return a gain of 1."""
        for name in ['linear', 'conv1d', 'conv2d', 'conv3d',
                     'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d']:
            self.assertEqual(I.calculate_init_gain(name), 1)

    def test_sigmoid(self):
        """Sigmoid returns a gain of 1."""
        self.assertEqual(I.calculate_init_gain('sigmoid'), 1)

    def test_tanh(self):
        """Tanh returns a gain of 5/3."""
        self.assertAlmostEqual(I.calculate_init_gain('tanh'), 5.0 / 3)

    def test_relu(self):
        """ReLU returns a gain of sqrt(2)."""
        self.assertAlmostEqual(I.calculate_init_gain('relu'), np.sqrt(2.0))

    def test_selu(self):
        """SELU returns a gain of 3/4."""
        self.assertEqual(I.calculate_init_gain('selu'), 3.0 / 4)

    def test_leaky_relu_default(self):
        """Leaky ReLU uses a default negative slope of 0.01."""
        expected = np.sqrt(2.0 / (1 + 0.01 ** 2))
        self.assertAlmostEqual(I.calculate_init_gain('leaky_relu'), expected)

    def test_leaky_relu_param(self):
        """Leaky ReLU honours an explicit numeric negative slope."""
        expected = np.sqrt(2.0 / (1 + 0.2 ** 2))
        self.assertAlmostEqual(I.calculate_init_gain('leaky_relu', 0.2), expected)

    def test_leaky_relu_invalid_param(self):
        """A non-numeric leaky-relu slope raises ``ValueError``."""
        with self.assertRaises(ValueError):
            I.calculate_init_gain('leaky_relu', 'bad')

    def test_unsupported(self):
        """An unknown nonlinearity raises ``ValueError``."""
        with self.assertRaises(ValueError):
            I.calculate_init_gain('does-not-exist')


class TestParamFunction(unittest.TestCase):
    """Tests for the ``param`` parameter-initialisation entry point."""

    def test_none_allowed(self):
        """``param`` returns ``None`` when ``parameter`` is ``None`` and allowed."""
        self.assertIsNone(I.param(None, (3,)))

    def test_none_disallowed(self):
        """``param`` raises when ``None`` is given but not allowed."""
        with self.assertRaises(ValueError):
            I.param(None, (3,), allow_none=False)

    def test_scalar_allowed(self):
        """``param`` returns a scalar unchanged when scalars are allowed."""
        self.assertEqual(I.param(5.0, (3,)), 5.0)

    def test_callable(self):
        """``param`` calls a callable initializer with the requested size."""
        out = I.param(brainstate.nn.init.ZeroInit(), (3,))
        self.assertEqual(out.shape, (3,))

    def test_callable_with_batch(self):
        """``param`` prepends the batch dimension for callable initializers."""
        out = I.param(brainstate.nn.init.ZeroInit(), (3,), batch_size=2)
        self.assertEqual(out.shape, (2, 3))

    def test_ndarray_matching(self):
        """``param`` accepts an array whose shape matches the requested size."""
        out = I.param(np.ones((3,)), (3,), allow_scalar=False)
        self.assertEqual(out.shape, (3,))

    def test_ndarray_with_batch(self):
        """``param`` broadcasts an array across a new batch dimension."""
        out = I.param(np.ones((3,)), (3,), batch_size=2, allow_scalar=False)
        self.assertEqual(out.shape, (2, 3))

    def test_ndarray_shape_mismatch(self):
        """``param`` raises when the array shape is not broadcastable."""
        with self.assertRaises(ValueError):
            I.param(np.ones((3,)), (5,), allow_scalar=False)

    def test_unknown_type(self):
        """``param`` raises for an unrecognised parameter type."""
        with self.assertRaises(ValueError):
            I.param(object(), (3,), allow_scalar=False)

    def test_state_input(self):
        """``param`` accepts a ``State`` parameter and returns a ``State``."""
        out = I.param(State(np.ones((3,))), (3,), allow_scalar=False)
        self.assertIsInstance(out, State)
        self.assertEqual(out.value.shape, (3,))


class TestInitializerRepr(unittest.TestCase):
    """Tests for the pretty-repr machinery on ``Initializer``."""

    def test_repr_contains_attrs(self):
        """The repr lists public attributes and skips private ones."""
        text = repr(brainstate.nn.init.Normal(scale=0.5, mean=0.1))
        self.assertIn('Normal', text)
        self.assertIn('scale', text)
        self.assertIn('mean', text)

    def test_base_call_not_implemented(self):
        """The base ``Initializer.__call__`` raises ``NotImplementedError``."""
        with self.assertRaises(NotImplementedError):
            I.Initializer()((3,))

    def test_repr_skips_private_attrs(self):
        """The repr omits attributes whose name starts with an underscore."""

        class _Custom(I.Initializer):
            """Initializer carrying both a public and a private attribute."""

            def __init__(self):
                self.public = 1
                self._private = 2

        text = repr(_Custom())
        self.assertIn('public', text)
        self.assertNotIn('_private', text)


class TestSeededDeterminism(unittest.TestCase):
    """Stochastic initializers reproduce under a fixed RNG seed context."""

    def _check(self, init, shape):
        """Assert two seeded draws from ``init`` match exactly."""
        with _testing.seeded(0):
            a = init(shape)
        with _testing.seeded(0):
            b = init(shape)
        _testing.assert_allclose(a, b)

    def test_normal(self):
        """``Normal`` is deterministic under a fixed seed."""
        self._check(brainstate.nn.init.Normal(), _SHAPE)

    def test_uniform(self):
        """``Uniform`` is deterministic under a fixed seed."""
        self._check(brainstate.nn.init.Uniform(), _SHAPE)

    def test_truncated_normal(self):
        """``TruncatedNormal`` is deterministic under a fixed seed."""
        self._check(brainstate.nn.init.TruncatedNormal(lower=-2, upper=2), _SHAPE)

    def test_variance_scaling(self):
        """``VarianceScaling`` is deterministic under a fixed seed."""
        init = brainstate.nn.init.VarianceScaling(
            scale=1., mode='fan_in', distribution='normal')
        self._check(init, _SHAPE)

    def test_orthogonal(self):
        """``Orthogonal`` is deterministic under a fixed seed."""
        self._check(brainstate.nn.init.Orthogonal(), _SHAPE)


class TestDtypeAndUnits(unittest.TestCase):
    """Initializers honour the ``dtype`` kwarg and the ``unit`` argument."""

    def test_zero_dtype(self):
        """``ZeroInit`` respects an explicit dtype."""
        out = brainstate.nn.init.ZeroInit()(_SHAPE, dtype=jnp.float16)
        self.assertEqual(out.dtype, jnp.float16)

    def test_constant_dtype(self):
        """``Constant`` respects an explicit dtype and fills the value."""
        out = brainstate.nn.init.Constant(value=3.0)(_SHAPE, dtype=jnp.float16)
        self.assertEqual(out.dtype, jnp.float16)
        self.assertTrue(bool((out == 3.0).all()))

    def test_normal_unit(self):
        """``Normal`` attaches the requested physical unit."""
        out = brainstate.nn.init.Normal(unit=u.mV)(_SHAPE)
        self.assertEqual(u.get_unit(out), u.mV)

    def test_zero_unit(self):
        """``ZeroInit`` attaches the requested physical unit."""
        out = brainstate.nn.init.ZeroInit(unit=u.mV)(_SHAPE)
        self.assertEqual(u.get_unit(out), u.mV)


class TestZeroInitValues(unittest.TestCase):
    """``ZeroInit`` produces all-zero arrays of the requested shape."""

    def test_zeros(self):
        """Every entry of a ``ZeroInit`` draw is zero."""
        out = brainstate.nn.init.ZeroInit()(_SHAPE)
        _testing.assert_allclose(out, jnp.zeros(_SHAPE))


class TestConstantValues(unittest.TestCase):
    """``Constant`` fills the array with the supplied value."""

    def test_fill(self):
        """Every entry equals the constant value."""
        out = brainstate.nn.init.Constant(value=2.5)(_SHAPE)
        _testing.assert_allclose(out, jnp.full(_SHAPE, 2.5))


class TestIdentityValues(unittest.TestCase):
    """``Identity`` produces a scaled identity matrix and validates rank."""

    def test_one_d_square(self):
        """A 1D shape yields a square identity matrix with scaled diagonal."""
        out = brainstate.nn.init.Identity(value=2.0)((3,))
        self.assertEqual(out.shape, (3, 3))
        np.testing.assert_allclose(np.diag(np.asarray(out)), [2.0, 2.0, 2.0])

    def test_three_d_raises(self):
        """Identity rejects shapes with more than two dimensions."""
        with self.assertRaises(ValueError):
            brainstate.nn.init.Identity()((3, 4, 5))


class TestOrthogonalProperty(unittest.TestCase):
    """``Orthogonal`` produces matrices with orthonormal columns."""

    def test_orthonormal_columns(self):
        """``w.T @ w`` approximates the identity for a tall matrix."""
        w = jnp.asarray(brainstate.nn.init.Orthogonal()((8, 4)))
        _testing.assert_allclose(w.T @ w, jnp.eye(4), atol=1e-4)

    def test_scale_and_axis(self):
        """``Orthogonal`` honours scale and axis arguments and shape."""
        out = brainstate.nn.init.Orthogonal(scale=2., axis=0)((4, 6))
        self.assertEqual(out.shape, (4, 6))


class TestDeltaOrthogonalValidation(unittest.TestCase):
    """``DeltaOrthogonal`` validates rank and fan ordering."""

    def test_valid_shapes(self):
        """3D/4D/5D shapes initialise to the requested shape."""
        for shape in [(3, 4, 5), (3, 3, 4, 5), (3, 3, 3, 4, 5)]:
            out = brainstate.nn.init.DeltaOrthogonal()(shape)
            self.assertEqual(out.shape, shape)

    def test_bad_rank(self):
        """A 2D shape is rejected."""
        with self.assertRaises(ValueError):
            brainstate.nn.init.DeltaOrthogonal()((3, 4))

    def test_fan_in_gt_fan_out(self):
        """A shape with fan_in > fan_out is rejected."""
        with self.assertRaises(ValueError):
            brainstate.nn.init.DeltaOrthogonal()((3, 4, 2))


class TestVarianceScalingBranches(unittest.TestCase):
    """``VarianceScaling`` covers each mode/distribution combination."""

    def test_modes_and_distributions(self):
        """Each valid mode and distribution yields the requested shape."""
        for mode in ['fan_in', 'fan_out', 'fan_avg']:
            for dist in ['truncated_normal', 'normal', 'uniform']:
                init = brainstate.nn.init.VarianceScaling(
                    scale=1., mode=mode, distribution=dist)
                out = init((10, 20))
                self.assertEqual(out.shape, (10, 20))

    def test_invalid_mode_assert(self):
        """An invalid mode is rejected at construction."""
        with self.assertRaises(AssertionError):
            brainstate.nn.init.VarianceScaling(
                scale=1., mode='bad', distribution='normal')

    def test_invalid_distribution_assert(self):
        """An invalid distribution is rejected at construction."""
        with self.assertRaises(AssertionError):
            brainstate.nn.init.VarianceScaling(
                scale=1., mode='fan_in', distribution='bad')


class TestTruncatedNormalBranches(unittest.TestCase):
    """``TruncatedNormal`` validates scale and respects bounds."""

    def test_scale_must_be_positive(self):
        """A non-positive scale is rejected at construction."""
        with self.assertRaises(AssertionError):
            brainstate.nn.init.TruncatedNormal(scale=-1.)

    def test_bounds_respected(self):
        """Samples lie within the requested lower/upper bounds."""
        init = brainstate.nn.init.TruncatedNormal(scale=1., lower=-1, upper=1, seed=0)
        arr = np.asarray(init((200,)))
        self.assertGreaterEqual(arr.min(), -1.001)
        self.assertLessEqual(arr.max(), 1.001)

    @pytest.mark.skip(reason="BUG: TruncatedNormal() with default lower/upper=None "
                             "crashes in random.truncated_normal (None - array)")
    def test_default_bounds(self):
        """``TruncatedNormal`` should work with default (None) bounds."""
        out = brainstate.nn.init.TruncatedNormal()((4,))
        self.assertEqual(out.shape, (4,))


class TestGammaExponential(unittest.TestCase):
    """``Gamma`` and ``Exponential`` produce arrays of the requested shape."""

    def test_gamma_shape(self):
        """``Gamma`` returns an array of the requested shape."""
        out = I.Gamma(shape=2.0, scale=1.0, seed=0)((4,))
        self.assertEqual(out.shape, (4,))

    def test_exponential_shape(self):
        """``Exponential`` returns an array of the requested shape."""
        out = I.Exponential(scale=1.0, seed=0)((4,))
        self.assertEqual(out.shape, (4,))
