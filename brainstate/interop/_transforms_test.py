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

"""Unit tests for the invertible weight-layout transforms."""

import brainunit as u
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

import brainstate
from brainstate.interop._transforms import (AddScalar, Chain, Identity,
                                            PermuteAxes, ReorderBlocks, Reshape,
                                            SplitConcat, Transpose)


def _arr(*shape):
    return brainstate.random.randn(*shape)


class TransformInvertibilityTest(absltest.TestCase):
    """Every transform must satisfy ``inverse(forward(x)) == x``."""

    def assertRoundTrip(self, tf, x):
        np.testing.assert_allclose(np.asarray(tf.inverse(tf.forward(x))),
                                   np.asarray(x), rtol=1e-6, atol=1e-6)

    def test_identity(self):
        x = _arr(3, 4)
        self.assertIs(Identity().forward(x), x)
        self.assertIs(Identity().inverse(x), x)
        self.assertRoundTrip(Identity(), x)

    def test_transpose_forward_and_roundtrip(self):
        x = _arr(2, 3, 4)
        tf = Transpose((2, 0, 1))
        y = tf.forward(x)
        self.assertEqual(y.shape, (4, 2, 3))
        np.testing.assert_array_equal(np.asarray(y), np.transpose(np.asarray(x), (2, 0, 1)))
        self.assertRoundTrip(tf, x)

    def test_transpose_inverse_perm_is_argsort(self):
        tf = Transpose((2, 0, 1))
        self.assertEqual(tf.inv_perm, (1, 2, 0))

    def test_permute_axes_is_transpose(self):
        x = _arr(2, 3)
        tf = PermuteAxes((1, 0))
        self.assertEqual(tf.forward(x).shape, (3, 2))
        self.assertRoundTrip(tf, x)

    def test_reshape_roundtrip(self):
        x = _arr(6)
        tf = Reshape(forward_shape=lambda s: (1, 1, s[0]), inverse_shape=lambda s: (s[-1],))
        y = tf.forward(x)
        self.assertEqual(y.shape, (1, 1, 6))
        self.assertRoundTrip(tf, x)

    def test_add_scalar(self):
        x = _arr(5)
        tf = AddScalar(1.0)
        np.testing.assert_allclose(np.asarray(tf.forward(x)), np.asarray(x) + 1.0, rtol=1e-6)
        np.testing.assert_allclose(np.asarray(tf.inverse(x)), np.asarray(x) - 1.0, rtol=1e-6)
        self.assertRoundTrip(tf, x)

    def test_reorder_blocks_specific(self):
        # four blocks [a, b, c, d]; order (0, 2, 1, 3) swaps the middle pair (self-inverse).
        x = jnp.concatenate([jnp.full((2,), float(i)) for i in range(4)])
        tf = ReorderBlocks(axis=0, order=(0, 2, 1, 3))
        y = tf.forward(x)
        np.testing.assert_array_equal(np.asarray(y[:2]), 0.0)
        np.testing.assert_array_equal(np.asarray(y[2:4]), 2.0)
        np.testing.assert_array_equal(np.asarray(y[4:6]), 1.0)
        np.testing.assert_array_equal(np.asarray(y[6:]), 3.0)
        self.assertRoundTrip(tf, x)

    def test_reorder_blocks_nontrivial_order(self):
        x = _arr(12, 5)
        tf = ReorderBlocks(axis=0, order=(2, 0, 3, 1))
        self.assertEqual(tf.inv_order, tuple(int(i) for i in np.argsort((2, 0, 3, 1))))
        self.assertRoundTrip(tf, x)

    def test_split_concat(self):
        x = _arr(10, 4)
        tf = SplitConcat(axis=0, sizes=(3, 7))
        parts = tf.forward(x)
        self.assertEqual(len(parts), 2)
        self.assertEqual(parts[0].shape, (3, 4))
        self.assertEqual(parts[1].shape, (7, 4))
        back = tf.inverse(parts)
        np.testing.assert_array_equal(np.asarray(back), np.asarray(x))

    def test_chain_order_and_roundtrip(self):
        x = _arr(2, 3)
        tf = Chain(Transpose((1, 0)), AddScalar(2.0))
        # forward = transpose then +2
        y = tf.forward(x)
        self.assertEqual(y.shape, (3, 2))
        np.testing.assert_allclose(np.asarray(y),
                                   np.transpose(np.asarray(x), (1, 0)) + 2.0, rtol=1e-6)
        self.assertRoundTrip(tf, x)

    def test_repr_smoke(self):
        for tf in (Identity(), Transpose((1, 0)), AddScalar(1.0),
                   ReorderBlocks(0, (1, 0)), SplitConcat(0, (1, 1)),
                   Chain(Identity())):
            self.assertIsInstance(repr(tf), str)


class TransformQuantityTest(absltest.TestCase):
    """Transforms must preserve ``brainunit.Quantity`` units."""

    def test_transpose_preserves_units(self):
        x = brainstate.random.randn(2, 3) * u.mV
        y = Transpose((1, 0)).forward(x)
        self.assertTrue(u.get_unit(y) == u.get_unit(x))
        self.assertEqual(y.shape, (3, 2))

    def test_reshape_preserves_units(self):
        x = brainstate.random.randn(6) * u.mV
        tf = Reshape(forward_shape=lambda s: (2, 3), inverse_shape=lambda s: (6,))
        y = tf.forward(x)
        self.assertTrue(u.get_unit(y) == u.get_unit(x))

    def test_add_scalar_preserves_units(self):
        # The constant must adopt x's unit so a united bias (e.g. the LSTM
        # forget-gate +1) folds in correctly instead of raising UnitMismatchError.
        x = brainstate.random.randn(5) * u.mV
        tf = AddScalar(1.0)
        y = tf.forward(x)
        self.assertTrue(u.get_unit(y) == u.get_unit(x))
        # forward adds the constant in x's unit ...
        np.testing.assert_allclose(np.asarray(u.get_mantissa(y)),
                                   np.asarray(u.get_mantissa(x)) + 1.0, rtol=1e-6)
        # ... and inverse(forward(x)) round-trips exactly (unit and value).
        z = tf.inverse(y)
        self.assertTrue(u.get_unit(z) == u.get_unit(x))
        np.testing.assert_allclose(np.asarray(u.get_mantissa(z)),
                                   np.asarray(u.get_mantissa(x)), rtol=1e-6)


if __name__ == '__main__':
    absltest.main()
