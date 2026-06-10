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


class TestHessianStateStructure(unittest.TestCase):
    """The Hessian w.r.t. ``grad_states`` must mirror the user's structure,
    not expose internal raw-``id()``-keyed dicts (audit M3)."""

    def test_single_state_returns_dense_block(self):
        w = brainstate.State(jnp.asarray(1.0))

        def loss():
            return w.value ** 3

        h = brainstate.transform.hessian(loss, grad_states=w)()
        self.assertNotIsInstance(h, dict)
        self.assertEqual(float(h), 6.0)  # d2(w^3)/dw2 = 6w

    def test_single_state_with_return_value(self):
        w = brainstate.State(jnp.asarray(1.0))

        def loss():
            return w.value ** 3

        h, val = brainstate.transform.hessian(loss, grad_states=w, return_value=True)()
        self.assertNotIsInstance(h, dict)
        self.assertEqual(float(h), 6.0)
        self.assertEqual(float(val), 1.0)

    def test_list_of_states_structure(self):
        w1 = brainstate.State(jnp.asarray(1.0))
        w2 = brainstate.State(jnp.asarray(2.0))

        def loss():
            return w1.value ** 2 * w2.value

        h = brainstate.transform.hessian(loss, grad_states=[w1, w2])()
        self.assertIsInstance(h, list)
        self.assertEqual(len(h), 2)
        self.assertIsInstance(h[0], list)
        # d2L/dw1dw1 = 2*w2 = 4, d2L/dw1dw2 = 2*w1 = 2, d2L/dw2dw2 = 0
        self.assertEqual(float(h[0][0]), 4.0)
        self.assertEqual(float(h[0][1]), 2.0)
        self.assertEqual(float(h[1][0]), 2.0)
        self.assertEqual(float(h[1][1]), 0.0)

    def test_dict_of_states_structure(self):
        ws = {
            'a': brainstate.State(jnp.asarray(1.0)),
            'b': brainstate.State(jnp.asarray(2.0)),
        }

        def loss():
            return ws['a'].value ** 2 * ws['b'].value

        h = brainstate.transform.hessian(loss, grad_states=ws)()
        self.assertIsInstance(h, dict)
        self.assertEqual(set(h.keys()), {'a', 'b'})
        self.assertEqual(set(h['a'].keys()), {'a', 'b'})
        self.assertEqual(float(h['a']['a']), 4.0)
        self.assertEqual(float(h['a']['b']), 2.0)
        self.assertEqual(float(h['b']['a']), 2.0)
        self.assertEqual(float(h['b']['b']), 0.0)

    def test_states_and_argnums_not_supported(self):
        w = brainstate.State(jnp.asarray(1.0))

        def loss(x):
            return w.value * x ** 2

        with self.assertRaises(NotImplementedError):
            brainstate.transform.hessian(loss, grad_states=w, argnums=0)


if __name__ == "__main__":
    unittest.main()
