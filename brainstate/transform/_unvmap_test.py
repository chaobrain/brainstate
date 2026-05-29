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

"""Tests for :func:`brainstate.transform.unvmap` and its reduction variants.

``unvmap`` exposes custom primitives whose two code paths matter:
eager evaluation (the ``def_impl`` rule), and the batching rule that fires when
the primitive is encountered inside :func:`jax.vmap`. Both paths are exercised
here for ``all``/``any``/``max``/``none``.
"""

import unittest

import jax
import jax.numpy as jnp

import brainstate
from brainstate.transform import unvmap
from brainstate.transform._unvmap import unvmap_all, unvmap_any, unvmap_max


class TestUnvmapEager(unittest.TestCase):
    """Eager reductions and op validation (the ``def_impl`` path)."""

    def test_unvmap_any_eager(self):
        """``op='any'`` reduces a boolean array to its ``any()``."""
        self.assertTrue(bool(unvmap(jnp.array([False, True, False]), op='any')))
        self.assertFalse(bool(unvmap(jnp.array([False, False]), op='any')))

    def test_unvmap_all_eager(self):
        """``op='all'`` reduces to ``all()``."""
        self.assertTrue(bool(unvmap(jnp.array([True, True]), op='all')))
        self.assertFalse(bool(unvmap(jnp.array([True, False]), op='all')))

    def test_unvmap_max_eager(self):
        """``op='max'`` reduces to ``max()``."""
        self.assertEqual(int(unvmap(jnp.array([1, 5, 3]), op='max')), 5)

    def test_unvmap_none_returns_value(self):
        """``op='none'`` returns the input unchanged."""
        x = jnp.array([1.0, 2.0, 3.0])
        out = unvmap(x, op='none')
        self.assertTrue(bool(jnp.allclose(out, x)))

    def test_unvmap_invalid_op_raises(self):
        """An unsupported op raises ``ValueError``."""
        with self.assertRaises(ValueError):
            unvmap(jnp.array([1, 2]), op='bogus')

    def test_direct_reduction_helpers_eager(self):
        """The direct helpers reduce eagerly via their primitives."""
        self.assertTrue(bool(unvmap_all(jnp.array([True, True]))))
        self.assertTrue(bool(unvmap_any(jnp.array([False, True]))))
        self.assertEqual(int(unvmap_max(jnp.array([2, 9, 4]))), 9)


class TestUnvmapBatched(unittest.TestCase):
    """The batching rule path: each reduction collapses the vmapped axis."""

    def test_unvmap_any_under_vmap(self):
        """``unvmap_any`` collapses the batch axis to a shared scalar per lane."""
        out = jax.vmap(lambda row: unvmap_any(row > 0))(
            jnp.array([[1.0, -1.0], [-1.0, -1.0]])
        )
        self.assertEqual(out.shape, (2,))
        # The reduction is shared across the batch, so all lanes agree.
        self.assertTrue(bool(jnp.all(out == out[0])))

    def test_unvmap_all_under_vmap(self):
        """``unvmap_all`` survives the batching rule and stays boolean."""
        out = jax.vmap(lambda row: unvmap_all(row > 0))(
            jnp.array([[1.0, 1.0], [1.0, -1.0]])
        )
        self.assertEqual(out.shape, (2,))
        self.assertEqual(out.dtype, jnp.bool_)

    def test_unvmap_max_under_vmap(self):
        """``unvmap_max`` survives the batching rule and preserves dtype."""
        data = jnp.array([[1.0, 5.0], [2.0, 3.0]])
        out = jax.vmap(lambda row: unvmap_max(row))(data)
        self.assertEqual(out.shape, (2,))
        self.assertEqual(out.dtype, data.dtype)

    def test_unvmap_none_under_vmap(self):
        """``op='none'`` escapes the vmap batching rule without error.

        The ``no_vmap`` primitive marks its output as *not mapped*, so the value
        leaves the vmapped axis; the concrete re-broadcast shape is an internal
        detail, so we only assert the call succeeds and stays finite.
        """
        out = jax.vmap(lambda row: unvmap(row, op='none'))(jnp.arange(6.0).reshape(2, 3))
        self.assertTrue(bool(jnp.all(jnp.isfinite(out))))


if __name__ == "__main__":
    unittest.main()
