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

import brainstate


class TestExports(unittest.TestCase):
    def test_symbols_exist(self):
        self.assertTrue(callable(brainstate.transform.vjp))
        self.assertTrue(callable(brainstate.transform.jvp))


class TestVjp(unittest.TestCase):
    def test_args_only_matches_jax(self):
        # vjp with no states should match jax.vjp on a pure function.
        def f(x):
            return jnp.sum(x ** 2)

        x = jnp.array([1.0, 2.0, 3.0])
        out, vjp_fn = brainstate.transform.vjp(f, x)
        jout, jvjp = jax.vjp(f, x)

        self.assertTrue(jnp.allclose(out, jout))
        # single int argnums -> cotangent returned unwrapped (brainstate grad convention)
        self.assertTrue(jnp.allclose(vjp_fn(1.0), jvjp(1.0)[0]))

    def test_sequence_argnums(self):
        def f(x, y):
            return jnp.sum(x * y)

        x = jnp.array([1.0, 2.0])
        y = jnp.array([3.0, 4.0])
        out, vjp_fn = brainstate.transform.vjp(f, x, y, argnums=(0, 1))
        gx, gy = vjp_fn(1.0)
        self.assertTrue(jnp.allclose(gx, y))   # d/dx sum(x*y) = y
        self.assertTrue(jnp.allclose(gy, x))   # d/dy sum(x*y) = x

    def test_grad_states(self):
        w = brainstate.State(jnp.array([2.0, 3.0]))

        def f(x):
            return jnp.sum(w.value * x)

        x = jnp.array([5.0, 7.0])
        out, vjp_fn = brainstate.transform.vjp(f, x, grad_states=w)
        state_ct, arg_ct = vjp_fn(1.0)
        self.assertTrue(jnp.allclose(out, jnp.sum(w.value * x)))
        self.assertTrue(jnp.allclose(state_ct, x))         # d/dw sum(w*x) = x
        self.assertTrue(jnp.allclose(arg_ct, w.value))     # d/dx sum(w*x) = w

    def test_dict_grad_states(self):
        w = brainstate.State(jnp.array([2.0, 3.0]))

        def f(x):
            return jnp.sum(w.value * x)

        x = jnp.array([5.0, 7.0])
        out, vjp_fn = brainstate.transform.vjp(f, x, grad_states={'w': w})
        state_ct, arg_ct = vjp_fn(1.0)
        self.assertIn('w', state_ct)
        self.assertTrue(jnp.allclose(state_ct['w'], x))

    def test_has_aux_and_state_writeback(self):
        counter = brainstate.State(jnp.array(0.0))

        def f(x):
            counter.value = counter.value + 1.0   # written, non-grad state
            return jnp.sum(x ** 2), {'mean': jnp.mean(x)}

        x = jnp.array([1.0, 2.0])
        out, vjp_fn, aux = brainstate.transform.vjp(f, x, has_aux=True)
        self.assertTrue(jnp.allclose(out, jnp.sum(x ** 2)))
        self.assertTrue(jnp.allclose(aux['mean'], jnp.mean(x)))
        self.assertTrue(jnp.allclose(counter.value, 1.0))   # state write re-threaded
        self.assertTrue(jnp.allclose(vjp_fn(1.0), 2.0 * x))


class TestJvp(unittest.TestCase):
    def test_args_only_matches_jax(self):
        def f(x):
            return jnp.sum(x ** 2)

        x = jnp.array([1.0, 2.0, 3.0])
        v = jnp.array([1.0, 1.0, 1.0])
        out, tangent_out = brainstate.transform.jvp(f, (x,), (v,))
        jout, jtan = jax.jvp(f, (x,), (v,))
        self.assertTrue(jnp.allclose(out, jout))
        self.assertTrue(jnp.allclose(tangent_out, jtan))

    def test_multiple_args(self):
        def f(x, y):
            return x * y

        out, tan = brainstate.transform.jvp(f, (2.0, 3.0), (1.0, 0.0))
        # d(xy) with dx=1, dy=0 -> y = 3.0
        self.assertTrue(jnp.allclose(out, 6.0))
        self.assertTrue(jnp.allclose(tan, 3.0))

    def test_state_passthrough_and_writeback(self):
        w = brainstate.State(jnp.array([2.0, 3.0]))
        counter = brainstate.State(jnp.array(0.0))

        def f(x):
            counter.value = counter.value + 1.0
            return jnp.sum(w.value * x)

        x = jnp.array([5.0, 7.0])
        v = jnp.array([1.0, 1.0])
        out, tan = brainstate.transform.jvp(f, (x,), (v,))
        self.assertTrue(jnp.allclose(out, jnp.sum(w.value * x)))
        self.assertTrue(jnp.allclose(tan, jnp.sum(w.value * v)))   # = w . v
        self.assertTrue(jnp.allclose(counter.value, 1.0))          # write re-threaded

    def test_has_aux(self):
        def f(x):
            return jnp.sum(x ** 2), {'n': x.shape[0]}

        x = jnp.array([1.0, 2.0])
        v = jnp.array([1.0, 1.0])
        out, tan, aux = brainstate.transform.jvp(f, (x,), (v,), has_aux=True)
        self.assertTrue(jnp.allclose(out, jnp.sum(x ** 2)))
        self.assertTrue(jnp.allclose(tan, jnp.sum(2.0 * x * v)))
        self.assertEqual(aux['n'], 2)

    def test_bad_primals_type_raises(self):
        with self.assertRaises(TypeError):
            brainstate.transform.jvp(lambda x: x, 1.0, (1.0,))


class TestComposition(unittest.TestCase):
    def test_vjp_under_jit(self):
        w = brainstate.State(jnp.array([2.0, 3.0]))

        def loss_and_grad(x):
            out, vjp_fn = brainstate.transform.vjp(
                lambda z: jnp.sum(w.value * z), x, grad_states=w
            )
            state_ct, arg_ct = vjp_fn(1.0)
            return out, state_ct, arg_ct

        x = jnp.array([5.0, 7.0])
        out, state_ct, arg_ct = jax.jit(loss_and_grad)(x)
        self.assertTrue(jnp.allclose(out, jnp.sum(w.value * x)))
        self.assertTrue(jnp.allclose(state_ct, x))
        self.assertTrue(jnp.allclose(arg_ct, w.value))

    def test_jvp_under_jit(self):
        def f(x):
            return jnp.sum(x ** 2)

        @jax.jit
        def run(x, v):
            return brainstate.transform.jvp(f, (x,), (v,))

        x = jnp.array([1.0, 2.0, 3.0])
        v = jnp.array([1.0, 1.0, 1.0])
        out, tan = run(x, v)
        jout, jtan = jax.jvp(f, (x,), (v,))
        self.assertTrue(jnp.allclose(out, jout))
        self.assertTrue(jnp.allclose(tan, jtan))

    def test_vjp_matches_grad(self):
        # The pullback at cotangent 1.0 equals brainstate.transform.grad.
        w = brainstate.State(jnp.array([2.0, 3.0]))

        def loss(x):
            return jnp.sum(w.value * x ** 2)

        x = jnp.array([5.0, 7.0])
        _, vjp_fn = brainstate.transform.vjp(loss, x, grad_states=w)
        state_ct, arg_ct = vjp_fn(1.0)

        ref = brainstate.transform.grad(loss, grad_states=w, argnums=0, return_value=False)
        ref_state_grad, ref_arg_grad = ref(x)
        self.assertTrue(jnp.allclose(state_ct, ref_state_grad))
        self.assertTrue(jnp.allclose(arg_ct, ref_arg_grad))


if __name__ == '__main__':
    unittest.main()
