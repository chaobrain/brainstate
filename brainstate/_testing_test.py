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

"""Tests for the shared test-helper module."""

import unittest

import brainunit as u
import jax.numpy as jnp

from brainstate import _testing


class TestAssertAllclose(unittest.TestCase):
    """Validate the unit-aware closeness assertion."""

    def test_equal_arrays_pass(self):
        """Identical arrays pass."""
        _testing.assert_allclose(jnp.ones((2, 3)), jnp.ones((2, 3)))

    def test_shape_mismatch_raises(self):
        """Different shapes raise AssertionError."""
        with self.assertRaises(AssertionError):
            _testing.assert_allclose(jnp.ones((2, 3)), jnp.ones((2, 4)))

    def test_value_mismatch_raises(self):
        """Different values raise AssertionError."""
        with self.assertRaises(AssertionError):
            _testing.assert_allclose(jnp.zeros((2,)), jnp.ones((2,)))

    def test_unit_aware(self):
        """Quantities with equal magnitude+unit pass."""
        a = u.Quantity(jnp.ones((3,)), unit=u.mV)
        _testing.assert_allclose(a, a)

    def test_dtype_check(self):
        """check_dtype=True catches dtype mismatch."""
        with self.assertRaises(AssertionError):
            _testing.assert_allclose(
                jnp.ones((2,), dtype=jnp.float32),
                jnp.ones((2,), dtype=jnp.int32),
                check_dtype=True,
            )


class TestTransformHelpers(unittest.TestCase):
    """Validate jit/grad/vmap helpers."""

    def test_jit_equal(self):
        """jit(fn) matches eager fn."""
        _testing.assert_jit_equal(lambda x: x * 2.0, jnp.ones((4, 8)))

    def test_grad_finite(self):
        """grad of a scalar fn is finite."""
        _testing.assert_grad_finite(lambda z: jnp.sum(z ** 2), jnp.ones((4, 8)))

    def test_vmap_equal(self):
        """vmap(fn) over axis 0 matches a Python loop."""
        _testing.assert_vmap_equal(lambda r: r.sum(), jnp.arange(12.0).reshape(4, 3))

    def test_transform_compatible_default_jit(self):
        """Umbrella helper runs jit by default."""
        _testing.assert_transform_compatible(lambda x: x + 1.0, jnp.ones((4, 8)))


class TestPytreeRoundtrip(unittest.TestCase):
    """Validate pytree flatten/unflatten roundtrip helper."""

    def test_dict_roundtrip(self):
        """A nested dict roundtrips structurally and by value."""
        obj = {"a": jnp.ones((2,)), "b": {"c": jnp.zeros((3,))}}
        _testing.assert_pytree_roundtrip(obj)


class TestConstants(unittest.TestCase):
    """Size constants exist and are small."""

    def test_sizes_small(self):
        """Constants stay tiny per the performance convention."""
        self.assertLessEqual(_testing.SMALL_BATCH, 8)
        self.assertLessEqual(_testing.SMALL_DIM, 32)
        self.assertLessEqual(_testing.SMALL_SEQ, 8)


if __name__ == "__main__":
    unittest.main()
