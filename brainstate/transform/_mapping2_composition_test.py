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

"""
Composition and integration tests for the modern mapping transforms.

Covers the composition matrix from the design spec:

- A: autodiff (grad through vmap2, vmap2-of-scan-of-grad)
- B: control flow (scan / while_loop inside vmap2)
- C: conditionals (cond with batched predicate)
- D: jit (jit∘vmap2, vmap2 under jit)
- E: remat
- F: pmap2(vmap2)
- G: deep grad(vmap2(scan(rnn_step)))

plus integration with neural-network modules, RNG, kwargs, output axis
placement, and the ``unexpected_out_state_mapping`` policies.
"""

import os

# Simulate 8 devices so pmap2 paths exercise real multi-device behaviour.
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import unittest
import warnings

import jax
import jax.numpy as jnp

import brainstate
import brainstate.random
from brainstate._error import BatchAxisError
from brainstate.transform import (
    vmap2, pmap2, vmap2_new_states, pmap2_new_states, map,
    grad, jit, scan, cond, while_loop, remat,
)
from brainstate.util import filter


class TestAutodiffComposition(unittest.TestCase):
    """Matrix A: autodiff composed with vmap2."""

    def test_grad_through_vmap2_stateless(self):
        w = brainstate.ParamState(jnp.array(3.0))

        def model(xs):
            ys = vmap2(lambda x: x * w.value)(xs)
            return jnp.sum(ys)

        g = grad(model, grad_states=w)(jnp.asarray([1., 2., 4.]))
        self.assertTrue(jnp.allclose(g, 7.0))  # sum(x)

    def test_vmap2_of_per_lane_grad(self):
        w = brainstate.ParamState(jnp.array(2.0))

        def loss(x):
            return (x ** 2) * w.value

        def per_lane(x):
            return grad(loss, grad_states=w)(x)  # x**2

        out = vmap2(per_lane)(jnp.asarray([1., 2., 3.]))
        self.assertTrue(jnp.allclose(out, jnp.asarray([1., 4., 9.])))

    def test_deep_grad_vmap2_scan_rnn(self):
        # Matrix G: grad(vmap2(scan(rnn_step)))
        W = brainstate.ParamState(jnp.array(0.5))

        def rnn(xs_lane):
            def step(c, x):
                c2 = c + W.value * x
                return c2, c2
            last, _ = scan(step, jnp.array(0.0), xs_lane)
            return last

        def loss(data):
            return jnp.sum(vmap2(rnn)(data))

        g = grad(loss, grad_states=W)(jnp.ones((2, 3)))
        self.assertTrue(jnp.allclose(g, 6.0))  # 2 lanes * (x0+x1+x2)=3 each


class TestControlFlowComposition(unittest.TestCase):
    """Matrix B/C: control flow and conditionals inside vmap2."""

    def test_scan_inside_vmap2(self):
        def f(xs_lane):
            def step(c, x):
                return c + x, c + x
            _, ys = scan(step, jnp.array(0.0), xs_lane)
            return ys

        data = jnp.arange(6.).reshape(2, 3)
        out = vmap2(f)(data)
        self.assertTrue(jnp.allclose(out, jnp.cumsum(data, axis=1)))

    def test_cond_batched_predicate(self):
        def f(x):
            return cond(x > 0, lambda: x * 10.0, lambda: x * -1.0)

        out = vmap2(f)(jnp.asarray([1.0, -2.0, 3.0]))
        self.assertTrue(jnp.allclose(out, jnp.asarray([10.0, 2.0, 30.0])))

    def test_while_loop_inside_vmap2(self):
        # count up to a per-lane target
        def f(n):
            def cond_fn(carry):
                i, acc = carry
                return i < n
            def body_fn(carry):
                i, acc = carry
                return i + 1, acc + i
            _, acc = while_loop(cond_fn, body_fn, (jnp.array(0), jnp.array(0)))
            return acc

        out = vmap2(f)(jnp.asarray([1, 2, 3]))
        # sum(range(n)): n=1->0, n=2->1, n=3->3
        self.assertTrue(jnp.allclose(out, jnp.asarray([0, 1, 3])))


class TestJitComposition(unittest.TestCase):
    """Matrix D: jit composed with vmap2."""

    def test_jit_of_vmap2_stateful(self):
        counter = brainstate.ShortTermState(jnp.zeros(3))

        @jit
        @vmap2(in_axes=0, state_in_axes=counter, state_out_axes=counter)
        def f(x):
            counter.value = counter.value + x
            return counter.value

        out = f(jnp.asarray([1., 2., 3.]))
        self.assertTrue(jnp.allclose(out, jnp.asarray([1., 2., 3.])))
        self.assertTrue(jnp.allclose(counter.value, jnp.asarray([1., 2., 3.])))

    def test_jit_of_vmap2_stateless(self):
        @jit
        def g(xs):
            return vmap2(lambda x: x * 2.0)(xs)

        self.assertTrue(jnp.allclose(g(jnp.arange(3.)), jnp.arange(3.) * 2))


class TestRematComposition(unittest.TestCase):
    """Matrix E: remat composed with vmap2."""

    def test_remat_of_vmap2(self):
        counter = brainstate.ShortTermState(jnp.zeros(3))

        def f(x):
            counter.value = counter.value + x
            return counter.value

        out = remat(vmap2(f, state_in_axes=counter, state_out_axes=counter))(
            jnp.asarray([1., 2., 3.])
        )
        self.assertTrue(jnp.allclose(out, jnp.asarray([1., 2., 3.])))


class TestPmapComposition(unittest.TestCase):
    """Matrix F + pmap integration (8 simulated devices)."""

    def setUp(self):
        self.n = jax.local_device_count()
        if self.n < 2:
            self.skipTest("Requires at least 2 devices")

    def test_pmap2_stateful_broadcast_in_batched_out(self):
        param = brainstate.ParamState(jnp.ones((4,)))

        @pmap2(in_axes=0, axis_name='d', state_out_axes={0: filter.OfType(brainstate.ParamState)})
        def update(delta):
            param.value = param.value + delta
            return param.value

        deltas = jnp.arange(self.n * 4., dtype=param.value.dtype).reshape(self.n, 4)
        out = update(deltas)
        self.assertEqual(out.shape, (self.n, 4))
        self.assertEqual(param.value.shape, (self.n, 4))

    def test_pmap2_of_vmap2(self):
        res = pmap2(lambda row: vmap2(lambda x: x * 2.0)(row), in_axes=0)(
            jnp.ones((self.n, 3))
        )
        self.assertEqual(res.shape, (self.n, 3))
        self.assertTrue(jnp.allclose(res, 2.0))

    def test_pmap2_new_states_distinct(self):
        class Ens(brainstate.nn.Module):
            def init_state(self, k):
                self.w = brainstate.ParamState(brainstate.random.randn(k))

        m = Ens()
        pmap2_new_states(m, init_kwargs={'k': 4}, axis_size=self.n)
        self.assertEqual(m.w.value.shape, (self.n, 4))
        self.assertFalse(jnp.allclose(m.w.value[0], m.w.value[1]))


class TestIntegration(unittest.TestCase):
    """General integration scenarios."""

    def test_kwargs_broadcast(self):
        @vmap2(in_axes=0)
        def f(x, *, scale):
            return x * scale

        out = f(jnp.arange(3.), scale=2.0)
        self.assertTrue(jnp.allclose(out, jnp.arange(3.) * 2.0))

    def test_rng_distinct_per_lane(self):
        rng = brainstate.random.RandomState(0)

        def f(x):
            return x + rng.randn(2)

        out = vmap2(f)(jnp.zeros((4, 2)))
        self.assertEqual(out.shape, (4, 2))
        self.assertFalse(jnp.allclose(out[0], out[1]))

    def test_nn_linear_ensemble(self):
        class MLP(brainstate.nn.Module):
            def init_state(self, din, dout):
                self.lin = brainstate.nn.Linear(din, dout)

            def update(self, x):
                return self.lin(x)

        m = brainstate.nn.Map(MLP(), init_map_size=4, behavior='vmap')
        m.init_all_states(din=3, dout=2)
        out = m.update(jnp.ones((4, 3)))
        self.assertEqual(out.shape, (4, 2))

    def test_multiple_states_different_axes(self):
        a = brainstate.ShortTermState(jnp.zeros(5))   # axis 0
        b = brainstate.ShortTermState(jnp.zeros(5))   # axis 0 too

        def f(x):
            a.value = a.value + x
            b.value = b.value + 2 * x
            return a.value + b.value

        out = vmap2(f, state_in_axes={0: [a, b]}, state_out_axes={0: [a, b]})(
            jnp.arange(5.)
        )
        self.assertTrue(jnp.allclose(out, 3 * jnp.arange(5.)))
        self.assertTrue(jnp.allclose(a.value, jnp.arange(5.)))
        self.assertTrue(jnp.allclose(b.value, 2 * jnp.arange(5.)))

    def test_out_axes_nonzero(self):
        def f(x):
            return jnp.stack([x, x * 2])  # (2,) per lane

        out = vmap2(f, in_axes=0, out_axes=1)(jnp.arange(3.))
        self.assertEqual(out.shape, (2, 3))

    def test_map_matches_vmap2(self):
        def f(x):
            return x * x + 1.0

        xs = jnp.arange(6.)
        self.assertTrue(jnp.allclose(map(f, xs), jax.vmap(f)(xs)))
        self.assertTrue(jnp.allclose(map(f, xs, batch_size=4), jax.vmap(f)(xs)))

    def test_map_rejects_bad_batch_size(self):
        with self.assertRaises(ValueError):
            map(lambda x: x, jnp.arange(4.), batch_size=0)

    def test_map_rejects_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            map(lambda a, b: a + b, jnp.arange(4.), jnp.arange(3.))


class TestUnexpectedStatePolicies(unittest.TestCase):
    """The ``unexpected_out_state_mapping`` policy matrix."""

    def test_auto_scatters(self):
        w = brainstate.ShortTermState(jnp.zeros(3))

        def f(x):
            w.value = w.value + x
            return w.value

        out = vmap2(f, unexpected_out_state_mapping='auto')(jnp.asarray([1., 2., 3.]))
        self.assertEqual(out.shape, (3, 3))
        self.assertEqual(w.value.shape, (3, 3))

    def test_raise_policy(self):
        w = brainstate.ShortTermState(jnp.zeros(3))

        def f(x):
            w.value = w.value + x
            return w.value

        with self.assertRaises(BatchAxisError):
            vmap2(f, unexpected_out_state_mapping='raise')(jnp.asarray([1., 2., 3.]))

    def test_warn_policy(self):
        w = brainstate.ShortTermState(jnp.zeros(3))

        def f(x):
            w.value = w.value + x
            return w.value

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = vmap2(f, unexpected_out_state_mapping='warn')(jnp.asarray([1., 2., 3.]))
        self.assertEqual(out.shape, (3, 3))
        self.assertTrue(any(issubclass(w_.category, UserWarning) for w_ in caught))

    def test_ignore_policy(self):
        w = brainstate.ShortTermState(jnp.zeros(3))

        def f(x):
            w.value = w.value + x
            return w.value

        out = vmap2(f, unexpected_out_state_mapping='ignore')(jnp.asarray([1., 2., 3.]))
        self.assertEqual(out.shape, (3, 3))


if __name__ == "__main__":
    unittest.main()
