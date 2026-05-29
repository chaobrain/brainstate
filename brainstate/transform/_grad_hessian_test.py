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

"""Tests for :func:`brainstate.transform.hessian`."""

import unittest

import jax
import jax.numpy as jnp

import brainstate


class TestHessian(unittest.TestCase):
    """Validate ``hessian`` against analytic second derivatives."""

    def test_quadratic_hessian_is_constant(self):
        """For ``f(x) = x . x`` the Hessian equals ``2 I``."""

        def f(x):
            return jnp.sum(x ** 2)

        H = brainstate.transform.hessian(f)(jnp.ones((3,)))
        self.assertEqual(H.shape, (3, 3))
        self.assertTrue(bool(jnp.allclose(H, 2.0 * jnp.eye(3))))

    def test_hessian_return_value(self):
        """``return_value=True`` also yields the function value."""

        def f(x):
            return jnp.sum(x ** 2)

        H, val = brainstate.transform.hessian(f, return_value=True)(jnp.ones((2,)))
        self.assertEqual(H.shape, (2, 2))
        self.assertTrue(bool(jnp.allclose(val, 2.0)))

    def test_hessian_has_aux(self):
        """``has_aux=True`` returns ``(hessian, aux)``."""

        def f(x):
            return jnp.sum(x ** 2), {"n": x.shape[0]}

        H, aux = brainstate.transform.hessian(f, has_aux=True)(jnp.ones((2,)))
        self.assertEqual(H.shape, (2, 2))
        self.assertEqual(aux["n"], 2)

    def test_hessian_over_param_state(self):
        """``hessian`` w.r.t. a ParamState returns a dense block."""
        p = brainstate.ParamState(jnp.ones((2,)))

        def loss():
            return jnp.sum(p.value ** 2)

        H = brainstate.transform.hessian(loss, grad_states=p)()
        block = jax.tree.leaves(H)[0]
        self.assertEqual(block.shape, (2, 2))
        self.assertTrue(bool(jnp.allclose(block, 2.0 * jnp.eye(2))))


if __name__ == "__main__":
    unittest.main()
