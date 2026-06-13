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


class TestVjpErrors(unittest.TestCase):
    """Cover error paths in vjp (lines 31, 119)."""

    def test_non_state_grad_states_raises_type_error(self):
        """_flatten_grad_states raises TypeError when given non-State values (line 31)."""
        with self.assertRaises(TypeError):
            brainstate.transform.vjp(lambda x: x, jnp.array(1.0), grad_states=[42])

    def test_unused_grad_state_raises_value_error(self):
        """vjp raises ValueError when a grad_state is not used inside the function (line 119)."""
        w = brainstate.State(jnp.array(1.0))
        unused = brainstate.State(jnp.array(2.0))

        def f(x):
            return w.value * x  # only reads w, not unused

        with self.assertRaises(ValueError):
            brainstate.transform.vjp(f, jnp.array(1.0), grad_states=unused)


class TestJvpErrors(unittest.TestCase):
    """Cover error paths in jvp (line 226)."""

    def test_bad_tangents_type_raises(self):
        """jvp raises TypeError when tangents is not a tuple or list (line 226)."""
        with self.assertRaises(TypeError):
            brainstate.transform.jvp(lambda x: x, (jnp.array(1.0),), 1.0)


class TestVjpStateOnly(unittest.TestCase):
    """State-only differentiation (no differentiable positional argument).

    Regression tests for the bug where ``vjp(loss, grad_states=...)`` with no
    positional primals crashed with ``IndexError`` because ``argnums`` defaulted
    to ``0`` and unconditionally indexed ``primals[0]``.
    """

    def test_no_primals_single_state(self):
        # The canonical neural-network pattern: loss closes over a parameter.
        w = brainstate.State(jnp.array([2.0, 3.0]))

        def loss():
            return jnp.sum(w.value ** 2)

        out, vjp_fn = brainstate.transform.vjp(loss, grad_states=w)
        state_ct = vjp_fn(1.0)
        # Pullback returns ONLY the state cotangent (no arg cotangent).
        self.assertTrue(jnp.allclose(out, jnp.sum(w.value ** 2)))
        self.assertTrue(jnp.allclose(state_ct, 2.0 * w.value))

    def test_no_primals_matches_grad(self):
        # State-only vjp pullback at 1.0 must equal brainstate.transform.grad.
        weight = brainstate.State(jnp.array([[1.0, 2.0], [3.0, 4.0]]))
        bias = brainstate.State(jnp.array([0.5, -0.5]))
        x = jnp.array([1.0, 1.0])

        def loss():
            return jnp.sum((x @ weight.value + bias.value) ** 2)

        out, vjp_fn = brainstate.transform.vjp(loss, grad_states=[weight, bias])
        gw, gb = vjp_fn(1.0)

        ref = brainstate.transform.grad(loss, grad_states=[weight, bias])
        ref_gw, ref_gb = ref()
        self.assertTrue(jnp.allclose(gw, ref_gw))
        self.assertTrue(jnp.allclose(gb, ref_gb))

    def test_argnums_none_with_primals_present(self):
        # argnums=None disables arg differentiation even if primals are given.
        w = brainstate.State(jnp.array([2.0, 3.0]))

        def f(x):
            return jnp.sum(w.value * x)

        x = jnp.array([5.0, 7.0])
        out, vjp_fn = brainstate.transform.vjp(f, x, grad_states=w, argnums=None)
        state_ct = vjp_fn(1.0)
        # Only state cotangent, returned unwrapped (single State).
        self.assertTrue(jnp.allclose(state_ct, x))
        self.assertNotIsInstance(state_ct, tuple)

    def test_no_primals_dict_states(self):
        w = brainstate.State(jnp.array([2.0, 3.0]))

        def loss():
            return jnp.sum(w.value ** 2)

        out, vjp_fn = brainstate.transform.vjp(loss, grad_states={'w': w})
        state_ct = vjp_fn(1.0)
        self.assertIn('w', state_ct)
        self.assertTrue(jnp.allclose(state_ct['w'], 2.0 * w.value))


class TestVjpArgnumsSemantics(unittest.TestCase):
    """argnums validation and the three return-structure regimes."""

    def test_out_of_range_argnums_raises_value_error(self):
        with self.assertRaises(ValueError) as cm:
            brainstate.transform.vjp(lambda x: jnp.sum(x), jnp.array([1.0]), argnums=5)
        self.assertIn('out of range', str(cm.exception))

    def test_negative_out_of_range_argnums_raises(self):
        with self.assertRaises(ValueError):
            brainstate.transform.vjp(lambda x: jnp.sum(x), jnp.array([1.0]), argnums=-3)

    def test_nothing_to_differentiate_raises(self):
        # No positional primals and no grad_states -> nothing to differentiate.
        with self.assertRaises(ValueError) as cm:
            brainstate.transform.vjp(lambda: jnp.array(5.0))
        self.assertIn('nothing to differentiate', str(cm.exception))

    def test_negative_argnums(self):
        def f(x, y):
            return jnp.sum(x * y ** 2)

        x, y = jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])
        out, vjp_fn = brainstate.transform.vjp(f, x, y, argnums=-1)
        g = vjp_fn(1.0)
        self.assertTrue(jnp.allclose(g, 2.0 * x * y))   # d/dy sum(x*y**2)

    def test_duplicate_argnums_raises(self):
        """Aliasing/duplicate argnums must fail loudly (mirroring ``grad``),
        not silently return a zeroed cotangent for the discarded input
        (audit M42)."""
        def f(x, y):
            return jnp.sum(x * 10 + y * 100)

        x = jnp.array([1.0, 1.0])
        y = jnp.array([1.0, 1.0])

        # literal duplicate
        with self.assertRaises(ValueError):
            brainstate.transform.vjp(f, x, y, argnums=(0, 0))
        # positive/negative alias of the same argument (index 0 == -2)
        with self.assertRaises(ValueError):
            brainstate.transform.vjp(f, x, y, argnums=(0, -2))
        # positive/negative alias of the same argument (index 1 == -1)
        with self.assertRaises(ValueError):
            brainstate.transform.vjp(f, x, y, argnums=(1, -1))
        # duplicate embedded among distinct indices
        with self.assertRaises(ValueError):
            brainstate.transform.vjp(f, x, y, argnums=(0, 1, 0))

    def test_distinct_argnums_still_correct(self):
        """The duplicate guard must not disturb genuinely distinct argnums:
        cotangents stay correct (no spurious zeroing)."""
        def f(x, y):
            return jnp.sum(x * 10 + y * 100)

        x = jnp.array([1.0, 1.0])
        y = jnp.array([1.0, 1.0])
        out, vjp_fn = brainstate.transform.vjp(f, x, y, argnums=(0, 1))
        gx, gy = vjp_fn(1.0)
        self.assertTrue(jnp.allclose(gx, jnp.array([10.0, 10.0])))   # d/dx
        self.assertTrue(jnp.allclose(gy, jnp.array([100.0, 100.0]))) # d/dy

        # negative index normalizes to a distinct slot and stays correct.
        out2, vjp_fn2 = brainstate.transform.vjp(f, x, y, argnums=(0, -1))
        gx2, gy2 = vjp_fn2(1.0)
        self.assertTrue(jnp.allclose(gx2, jnp.array([10.0, 10.0])))
        self.assertTrue(jnp.allclose(gy2, jnp.array([100.0, 100.0])))


class TestVjpComprehensive(unittest.TestCase):
    """Cover the documented use cases end to end."""

    def test_vector_output_matches_jax(self):
        def f(x):
            return x ** 2

        x = jnp.array([1.0, 2.0, 3.0])
        out, vjp_fn = brainstate.transform.vjp(f, x)
        jout, jvjp = jax.vjp(f, x)
        ct = jnp.array([1.0, 1.0, 1.0])
        self.assertTrue(jnp.allclose(out, jout))
        self.assertTrue(jnp.allclose(vjp_fn(ct), jvjp(ct)[0]))

    def test_pytree_output_matches_jax(self):
        def f(x):
            return {'a': jnp.sum(x), 'b': jnp.sum(x ** 2)}

        x = jnp.array([1.0, 2.0, 3.0])
        out, vjp_fn = brainstate.transform.vjp(f, x)
        jout, jvjp = jax.vjp(f, x)
        ct = {'a': 1.0, 'b': 1.0}
        self.assertTrue(jnp.allclose(vjp_fn(ct), jvjp(ct)[0]))

    def test_pytree_input(self):
        def f(d):
            return jnp.sum(d['a'] ** 2) + jnp.sum(d['b'])

        d = {'a': jnp.array([1.0, 2.0]), 'b': jnp.array([3.0, 4.0])}
        out, vjp_fn = brainstate.transform.vjp(f, d)
        r = vjp_fn(1.0)
        self.assertTrue(jnp.allclose(r['a'], 2.0 * d['a']))
        self.assertTrue(jnp.allclose(r['b'], jnp.ones_like(d['b'])))

    def test_full_jacobian_via_vmap_pullback(self):
        def f(x):
            return jnp.array([jnp.sum(x), jnp.sum(x ** 2), jnp.prod(x)])

        x = jnp.array([1.0, 2.0, 3.0])
        out, vjp_fn = brainstate.transform.vjp(f, x)
        jac = jax.vmap(vjp_fn)(jnp.eye(3))
        self.assertTrue(jnp.allclose(jac, jax.jacrev(f)(x)))

    def test_pullback_reusable_and_linear(self):
        def f(x):
            return x ** 2

        x = jnp.array([1.0, 2.0, 3.0])
        out, vjp_fn = brainstate.transform.vjp(f, x)
        r1 = vjp_fn(jnp.ones(3))
        r2 = vjp_fn(2.0 * jnp.ones(3))
        self.assertTrue(jnp.allclose(r2, 2.0 * r1))   # pullback is linear

    def test_hessian_vector_product_via_nested_vjp(self):
        def f(x):
            return jnp.sum(x ** 3)

        x = jnp.array([1.0, 2.0, 3.0])
        v = jnp.array([1.0, 1.0, 1.0])

        def grad_f(x):
            _, vjp_fn = brainstate.transform.vjp(f, x)
            return vjp_fn(1.0)

        _, hvp_fn = brainstate.transform.vjp(grad_f, x)
        hvp = hvp_fn(v)
        # Hessian of sum(x**3) is diag(6x); H @ v = 6x.
        self.assertTrue(jnp.allclose(hvp, 6.0 * x))

    def test_multiple_states_and_args(self):
        w1 = brainstate.State(jnp.array([2.0, 3.0]))
        w2 = brainstate.State(jnp.array([0.5, 1.5]))

        def f(x, y):
            return jnp.sum(w1.value * x ** 2 + w2.value * y)

        x, y = jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])
        out, vjp_fn = brainstate.transform.vjp(
            f, x, y, grad_states=[w1, w2], argnums=(0, 1)
        )
        state_ct, arg_ct = vjp_fn(1.0)
        self.assertTrue(jnp.allclose(state_ct[0], x ** 2))           # d/dw1
        self.assertTrue(jnp.allclose(state_ct[1], y))                # d/dw2
        self.assertTrue(jnp.allclose(arg_ct[0], 2.0 * w1.value * x)) # d/dx
        self.assertTrue(jnp.allclose(arg_ct[1], w2.value))           # d/dy

    def test_nested_dict_grad_states_structure(self):
        w = brainstate.State(jnp.array([2.0, 3.0]))
        b = brainstate.State(jnp.array([1.0]))

        def f(x):
            return jnp.sum(w.value * x) + jnp.sum(b.value)

        x = jnp.array([5.0, 7.0])
        out, vjp_fn = brainstate.transform.vjp(
            f, x, grad_states={'layer': {'w': w, 'b': b}}
        )
        state_ct, arg_ct = vjp_fn(1.0)
        self.assertTrue(jnp.allclose(state_ct['layer']['w'], x))
        self.assertTrue(jnp.allclose(state_ct['layer']['b'], jnp.array([1.0])))

    def test_readonly_nongrad_state_is_constant(self):
        const = brainstate.State(jnp.array([10.0, 20.0]))
        w = brainstate.State(jnp.array([2.0, 3.0]))

        def f(x):
            return jnp.sum(const.value * w.value * x)

        x = jnp.array([1.0, 1.0])
        out, vjp_fn = brainstate.transform.vjp(f, x, grad_states=w)
        state_ct, arg_ct = vjp_fn(1.0)
        self.assertTrue(jnp.allclose(state_ct, const.value * x))
        self.assertTrue(jnp.allclose(arg_ct, const.value * w.value))
        # The non-grad read-only state is unchanged.
        self.assertTrue(jnp.allclose(const.value, jnp.array([10.0, 20.0])))

    def test_grad_state_read_then_written(self):
        # A grad_state that is read (for the output) and also written.
        w = brainstate.State(jnp.array([2.0, 3.0]))

        def f(x):
            old = w.value
            w.value = w.value + 1.0          # written
            return jnp.sum(old * x)          # output depends on the OLD value

        x = jnp.array([5.0, 7.0])
        out, vjp_fn = brainstate.transform.vjp(f, x, grad_states=w)
        state_ct, arg_ct = vjp_fn(1.0)
        self.assertTrue(jnp.allclose(state_ct, x))                 # d/dw_in
        self.assertTrue(jnp.allclose(arg_ct, jnp.array([2.0, 3.0])))  # d/dx = w_in
        self.assertTrue(jnp.allclose(w.value, jnp.array([3.0, 4.0])))  # write-back

    def test_has_aux_with_grad_states(self):
        w = brainstate.State(jnp.array([2.0, 3.0]))

        def f(x):
            return jnp.sum(w.value * x), {'norm': jnp.sum(x ** 2)}

        x = jnp.array([5.0, 7.0])
        out, vjp_fn, aux = brainstate.transform.vjp(
            f, x, grad_states=w, has_aux=True
        )
        state_ct, arg_ct = vjp_fn(1.0)
        self.assertTrue(jnp.allclose(out, jnp.sum(w.value * x)))
        self.assertTrue(jnp.allclose(aux['norm'], jnp.sum(x ** 2)))
        self.assertTrue(jnp.allclose(state_ct, x))
        self.assertTrue(jnp.allclose(arg_ct, w.value))

    def test_state_only_vjp_under_jit(self):
        w = brainstate.State(jnp.array([2.0, 3.0]))

        @jax.jit
        def run():
            out, vjp_fn = brainstate.transform.vjp(
                lambda: jnp.sum(w.value ** 2), grad_states=w
            )
            return out, vjp_fn(1.0)

        out, state_ct = run()
        self.assertTrue(jnp.allclose(state_ct, 2.0 * w.value))


if __name__ == '__main__':
    unittest.main()
