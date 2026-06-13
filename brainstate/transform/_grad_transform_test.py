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

"""Tests for the :class:`GradientTransform` engine shared by grad/vector_grad/hessian.

These tests target the engine directly (construction, validation, repr, the
``check_states`` policy, and the ``debug_nan`` path) rather than the public
``grad``/``vector_grad`` wrappers, which are covered in ``_grad_grad_test.py``.
"""

import unittest

import jax
import jax.numpy as jnp

import brainstate
from brainstate.transform import GradientTransform


class TestGradientTransformConstruction(unittest.TestCase):
    """Construction-time validation and the public ``grad`` factory."""

    def test_grad_returns_gradient_transform(self):
        """``grad(fn)`` is a :class:`GradientTransform` instance."""
        gt = brainstate.transform.grad(lambda x: jnp.sum(x ** 2))
        self.assertIsInstance(gt, GradientTransform)

    def test_non_state_grad_states_raises_type_error(self):
        """Passing a non-:class:`State` in ``grad_states`` raises ``TypeError``."""
        with self.assertRaises(TypeError):
            GradientTransform(
                target=lambda: jnp.array(0.0),
                transform=jax.grad,
                grad_states=[object()],
            )

    def test_repr_lists_configuration(self):
        """``repr`` reflects the configured target, grad_states, and flags."""
        p = brainstate.ParamState(jnp.ones((2,)))

        def loss():
            return jnp.sum(p.value ** 2)

        gt = GradientTransform(target=loss, transform=jax.grad, grad_states=p)
        text = repr(gt)
        self.assertIn("GradientTransform", text)
        self.assertIn("grad_states", text)
        self.assertIn("return_value", text)


class TestGradientTransformBehavior(unittest.TestCase):
    """Output-shape behavior across has_aux / return_value / argnums settings."""

    def test_has_aux_returns_pair(self):
        """``has_aux=True`` returns ``(grads, aux)``."""

        def f(x):
            return jnp.sum(x ** 2), {"norm": jnp.sqrt(jnp.sum(x ** 2))}

        grads, aux = brainstate.transform.grad(f, has_aux=True)(jnp.ones((3,)))
        self.assertTrue(bool(jnp.allclose(grads, 2.0 * jnp.ones((3,)))))
        self.assertIn("norm", aux)

    def test_return_value(self):
        """``return_value=True`` returns ``(grads, value)``."""
        grads, val = brainstate.transform.grad(
            lambda x: jnp.sum(x ** 2), return_value=True
        )(jnp.ones((2,)))
        self.assertTrue(bool(jnp.allclose(val, 2.0)))
        self.assertTrue(bool(jnp.allclose(grads, 2.0 * jnp.ones((2,)))))

    def test_argnums_sequence(self):
        """``argnums`` as a sequence differentiates multiple positional args."""

        def f(a, b):
            return jnp.sum(a * b)

        ga, gb = brainstate.transform.grad(f, argnums=(0, 1))(
            jnp.ones((2,)), 2.0 * jnp.ones((2,))
        )
        self.assertTrue(bool(jnp.allclose(ga, 2.0 * jnp.ones((2,)))))
        self.assertTrue(bool(jnp.allclose(gb, jnp.ones((2,)))))

    def test_grad_states_and_argnums_returns_var_and_arg_grads(self):
        """With both grad_states and argnums, returns ``(var_grads, arg_grads)``."""
        p = brainstate.ParamState(jnp.array([2.0, 3.0]))

        def loss(x):
            return jnp.sum(p.value * x)

        var_grads, arg_grads = brainstate.transform.grad(
            loss, grad_states=p, argnums=0
        )(jnp.array([1.0, 1.0]))
        self.assertTrue(bool(jnp.allclose(var_grads, jnp.array([1.0, 1.0]))))
        self.assertTrue(bool(jnp.allclose(arg_grads, jnp.array([2.0, 3.0]))))

    def test_grad_states_yields_analytic_gradient(self):
        """grad w.r.t. a ParamState yields the analytic gradient."""
        p = brainstate.ParamState(jnp.array([3.0, 4.0]))

        def loss():
            return jnp.sum(p.value ** 2)

        grads = brainstate.transform.grad(loss, grad_states=p)()
        self.assertTrue(bool(jnp.allclose(grads, jnp.array([6.0, 8.0]))))


class TestGradientTransformCheckStates(unittest.TestCase):
    """The ``check_states`` policy for grad_states absent from the trace."""

    def test_missing_state_raises_by_default(self):
        """A grad_state never used in the function raises ``ValueError``."""
        used = brainstate.ParamState(jnp.ones((2,)))
        unused = brainstate.ParamState(jnp.ones((2,)))

        def loss():
            return jnp.sum(used.value ** 2)

        gt = brainstate.transform.grad(loss, grad_states=[used, unused])
        with self.assertRaises(ValueError):
            gt()

    def test_missing_state_tolerated_when_check_disabled(self):
        """``check_states=False`` tolerates an unused grad_state (grad ~ 0)."""
        used = brainstate.ParamState(jnp.ones((2,)))
        unused = brainstate.ParamState(jnp.ones((2,)))

        def loss():
            return jnp.sum(used.value ** 2)

        grads = brainstate.transform.grad(
            loss, grad_states=[used, unused], check_states=False
        )()
        # grads is a list aligned with [used, unused]
        self.assertTrue(bool(jnp.allclose(grads[0], 2.0 * jnp.ones((2,)))))
        self.assertTrue(bool(jnp.all(jnp.isfinite(grads[1]))))
        self.assertTrue(bool(jnp.allclose(grads[1], jnp.zeros((2,)))))


class TestGradientTransformDebugNaN(unittest.TestCase):
    """The ``debug_nan`` evaluation path."""

    def test_debug_nan_clean_function_returns_gradient(self):
        """With ``debug_nan=True`` a NaN-free function returns the gradient."""
        p = brainstate.ParamState(jnp.array([3.0, 4.0]))

        def loss():
            return jnp.sum(p.value ** 2)

        gt = GradientTransform(
            target=loss, transform=jax.grad, grad_states=p, debug_nan=True
        )
        grads = gt()
        self.assertTrue(bool(jnp.allclose(grads, jnp.array([6.0, 8.0]))))

    def test_nan_check_detects_complex(self):
        """The JIT-compatible NaN check must flag NaN in complex gradients."""
        from brainstate.transform._grad_transform import _check_nan_jit_compatible
        clean = jnp.array([1.0 + 2.0j], dtype=jnp.complex64)
        nan_real = jnp.array([complex(float('nan'), 1.0)], dtype=jnp.complex64)
        nan_imag = jnp.array([complex(1.0, float('nan'))], dtype=jnp.complex64)
        self.assertFalse(bool(_check_nan_jit_compatible(clean)))
        self.assertTrue(bool(_check_nan_jit_compatible(nan_real)))
        self.assertTrue(bool(_check_nan_jit_compatible(nan_imag)))


class TestHasAuxValidation(unittest.TestCase):
    """``has_aux=True`` requires the function to return a (loss, aux) pair."""

    def test_non_tuple_aux_return_raises_type_error(self):
        """A non-(tuple/list) return under has_aux=True raises a clear TypeError,
        not a bare assertion stripped under ``python -O``."""

        def f(x):
            return jnp.sum(x ** 2)  # forgot to return aux

        with self.assertRaises(TypeError) as ctx:
            brainstate.transform.grad(f, has_aux=True)(jnp.ones((3,)))
        self.assertIn('has_aux', str(ctx.exception))

    def test_tuple_aux_return_ok(self):
        """A proper (loss, aux) pair works under has_aux=True."""

        def f(x):
            return jnp.sum(x ** 2), {'n': x.shape[0]}

        grads, aux = brainstate.transform.grad(f, has_aux=True)(jnp.ones((2,)))
        self.assertTrue(bool(jnp.allclose(grads, 2.0 * jnp.ones((2,)))))
        self.assertEqual(aux['n'], 2)


if __name__ == "__main__":
    unittest.main()
