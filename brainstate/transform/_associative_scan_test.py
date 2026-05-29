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
import brainunit as u

import brainstate


class TestExports(unittest.TestCase):
    def test_symbols_exist(self):
        self.assertTrue(callable(brainstate.transform.associative_scan))
        self.assertTrue(callable(brainstate.transform.linear_recurrence))


class TestAssociativeScan(unittest.TestCase):
    def test_cumsum_matches_jax(self):
        xs = jnp.arange(1.0, 6.0)
        out = brainstate.transform.associative_scan(lambda a, b: a + b, xs)
        ref = jax.lax.associative_scan(lambda a, b: a + b, xs)
        self.assertTrue(jnp.allclose(out, ref))
        self.assertTrue(jnp.allclose(out, jnp.cumsum(xs)))

    def test_reverse(self):
        xs = jnp.arange(1.0, 6.0)
        out = brainstate.transform.associative_scan(lambda a, b: a + b, xs, reverse=True)
        ref = jax.lax.associative_scan(lambda a, b: a + b, xs, reverse=True)
        self.assertTrue(jnp.allclose(out, ref))

    def test_axis(self):
        xs = jnp.arange(12.0).reshape(3, 4)
        out = brainstate.transform.associative_scan(lambda a, b: a + b, xs, axis=1)
        ref = jax.lax.associative_scan(lambda a, b: a + b, xs, axis=1)
        self.assertTrue(jnp.allclose(out, ref))

    def test_pytree_tuple(self):
        a = jnp.arange(1.0, 5.0)
        b = jnp.arange(5.0, 9.0)
        out = brainstate.transform.associative_scan(
            lambda l, r: (l[0] + r[0], l[1] + r[1]), (a, b)
        )
        self.assertTrue(jnp.allclose(out[0], jnp.cumsum(a)))
        self.assertTrue(jnp.allclose(out[1], jnp.cumsum(b)))

    def test_quantity_units_preserved(self):
        q = jnp.arange(1.0, 6.0) * u.mV
        out = brainstate.transform.associative_scan(lambda a, b: a + b, q)
        self.assertTrue(isinstance(out, u.Quantity))
        self.assertEqual(u.get_unit(out), u.get_unit(q))
        self.assertTrue(jnp.allclose(out.to_decimal(u.mV), jnp.cumsum(jnp.arange(1.0, 6.0))))


def _sequential_linrec(decay, drive):
    # Reference implementation: h_t = decay_t * h_{t-1} + drive_t, h_0 = 0.
    h = jnp.zeros_like(drive[0])
    out = []
    for t in range(drive.shape[0]):
        h = decay[t] * h + drive[t]
        out.append(h)
    return jnp.stack(out)


class TestLinearRecurrence(unittest.TestCase):
    def test_matches_sequential_scalar_series(self):
        decay = jnp.array([0.9, 0.8, 0.7, 0.5])
        drive = jnp.array([1.0, 2.0, 3.0, 4.0])
        out = brainstate.transform.linear_recurrence(decay, drive)
        ref = _sequential_linrec(decay, drive)
        self.assertTrue(jnp.allclose(out, ref))

    def test_matches_sequential_vector_series(self):
        # Time axis = 0, feature dim = 3.
        decay = jnp.array([[0.9, 0.5, 0.1]] * 5)
        drive = jnp.arange(15.0).reshape(5, 3)
        out = brainstate.transform.linear_recurrence(decay, drive)
        ref = _sequential_linrec(decay, drive)
        self.assertTrue(jnp.allclose(out, ref))

    def test_under_jit(self):
        decay = jnp.array([0.9, 0.8, 0.7, 0.5])
        drive = jnp.array([1.0, 2.0, 3.0, 4.0])
        out = jax.jit(brainstate.transform.linear_recurrence)(decay, drive)
        ref = _sequential_linrec(decay, drive)
        self.assertTrue(jnp.allclose(out, ref))

    def test_units(self):
        decay = jnp.array([0.9, 0.8, 0.7])             # dimensionless multiplier
        drive = jnp.array([1.0, 2.0, 3.0]) * u.mV       # carries voltage units
        out = brainstate.transform.linear_recurrence(decay, drive)
        self.assertTrue(isinstance(out, u.Quantity))
        self.assertEqual(u.get_unit(out), u.mV)
        ref = _sequential_linrec(decay, drive.to_decimal(u.mV))
        self.assertTrue(jnp.allclose(out.to_decimal(u.mV), ref))


if __name__ == '__main__':
    unittest.main()
