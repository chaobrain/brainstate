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
import numpy as np

import brainstate.random
from brainstate._error import BatchAxisError
from brainstate.transform import (
    grad, jit, scan, cond, while_loop, remat,
)

import brainstate
import brainstate.random
from brainstate._state import NonBatchState
from brainstate.transform import StatefulMapping, vmap2, vmap_new_states, pmap2, map
from brainstate.transform import vmap2_new_states, pmap2_new_states
from brainstate.transform._mapping2 import (
    _ensure_tuple, _batch_and_remainder, _validate_leading_lengths,
    _build_new_state_resolver,
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

    def test_kwargs_mapped_over_axis0(self):
        # B1 fix: dynamic kwargs are mapped over axis 0, matching jax.vmap
        # (previously they were silently broadcast, diverging from jax).
        def f(x, *, scale):
            return x * scale

        x = jnp.arange(3.)
        scale = jnp.full(3, 2.0)
        out = vmap2(f, in_axes=0)(x, scale=scale)
        expected = jax.vmap(f, in_axes=0)(x, scale=scale)
        self.assertTrue(jnp.allclose(out, expected))
        self.assertTrue(jnp.allclose(out, jnp.arange(3.) * 2.0))

    def test_kwargs_static_broadcast(self):
        # A scalar kwarg can still be broadcast by declaring it static via
        # StatefulMapping(static_argnames=...) -- the documented escape hatch.
        from brainstate.transform import StatefulMapping

        def f(x, *, scale):
            return x * scale

        mapped = StatefulMapping(f, in_axes=0, static_argnames=('scale',))
        out = mapped(jnp.arange(3.), scale=2.0)
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
        a = brainstate.ShortTermState(jnp.zeros(5))  # axis 0
        b = brainstate.ShortTermState(jnp.zeros(5))  # axis 0 too

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

    def test_auto_promotes_read_modify_write(self):
        # B1: an undeclared read-modify-write state whose prior size along the
        # detected axis equals the batch size is promoted to a per-lane
        # input+output under 'auto' (stays (3,) instead of scattering to (3, 3)).
        w = brainstate.ShortTermState(jnp.zeros(3))

        def f(x):
            w.value = w.value + x
            return w.value

        out = vmap2(f, unexpected_out_state_mapping='auto')(jnp.asarray([1., 2., 3.]))
        self.assertEqual(out.shape, (3,))
        self.assertEqual(w.value.shape, (3,))
        self.assertTrue(jnp.allclose(w.value, jnp.asarray([1., 2., 3.])))

    def test_auto_scatters_pure_output(self):
        # A pure-output write (does not read its own prior value, no shape match)
        # is still scattered by 'auto': () -> (3,).
        w = brainstate.ShortTermState(jnp.zeros(()))

        def f(x):
            w.value = x * 2.0
            return x

        out = vmap2(f, unexpected_out_state_mapping='auto')(jnp.asarray([1., 2., 3.]))
        self.assertEqual(out.shape, (3,))
        self.assertEqual(w.value.shape, (3,))
        self.assertTrue(jnp.allclose(w.value, jnp.asarray([2., 4., 6.])))

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


class TestEnsureTuple(unittest.TestCase):
    """Tests for the _ensure_tuple helper (lines 56-60)."""

    def test_ensure_tuple_none_returns_empty(self):
        """_ensure_tuple(None) should return an empty tuple."""
        self.assertEqual(_ensure_tuple(None), ())

    def test_ensure_tuple_int_wraps_in_tuple(self):
        """_ensure_tuple(int) should wrap the integer in a single-element tuple."""
        self.assertEqual(_ensure_tuple(3), (3,))

    def test_ensure_tuple_iterable_converts(self):
        """_ensure_tuple([1,2]) should produce (1, 2)."""
        self.assertEqual(_ensure_tuple([1, 2]), (1, 2))


class TestStatefulMappingInit(unittest.TestCase):
    """Tests for StatefulMapping construction with non-default static args."""

    def test_static_argnums_int(self):
        """M41: StatefulMapping rejects a non-empty static_argnums (the engine
        never excludes static positional slots from the axis mapping, so it
        would crash/mismap). The int form is normalized before the check."""
        with self.assertRaises(NotImplementedError):
            StatefulMapping(lambda x: x, static_argnums=2)

    def test_static_argnames_list(self):
        """StatefulMapping accepts a list for static_argnames and converts to tuple."""
        sm = StatefulMapping(lambda x: x, static_argnames=['mode', 'flag'])
        self.assertEqual(sm.static_argnames, ('mode', 'flag'))

    def test_static_argnums_none(self):
        """StatefulMapping with static_argnums=None stores an empty tuple."""
        sm = StatefulMapping(lambda x: x, static_argnums=None)
        self.assertEqual(sm.static_argnums, ())


class TestMap(unittest.TestCase):
    def test_map_matches_vectorized(self):
        xs = jnp.arange(6.0).reshape(6, 1)

        def fn(x):
            return x + 1.0

        expected = jax.vmap(fn)(xs)
        result = map(fn, xs)
        self.assertTrue(jnp.allclose(result, expected))

    def test_map_multiple_inputs_and_batch_size(self):
        xs = jnp.arange(5.0)
        ys = jnp.ones_like(xs) * 2.0

        def fn(a, b):
            return a * a + b

        expected = jax.vmap(fn)(xs, ys)
        result = map(fn, xs, ys, batch_size=2)
        self.assertTrue(jnp.allclose(result, expected))


class TestVmapIntegration(unittest.TestCase):
    def test_decorator_batched_stateful_function(self):
        counter = brainstate.ShortTermState(jnp.zeros(3))

        @vmap2(
            in_axes=0,
            out_axes=0,
            state_in_axes={0: filter.OfType(brainstate.ShortTermState)},
            state_out_axes={0: filter.OfType(brainstate.ShortTermState)},
        )
        def accumulate(x):
            counter.value = counter.value + x
            return counter.value

        xs = jnp.asarray([1.0, 2.0, 3.0])
        result = accumulate(xs)
        self.assertTrue(jnp.allclose(result, xs))
        self.assertTrue(jnp.allclose(counter.value, xs))

    def test_vmap_partial_returns_stateful_mapping(self):
        builder = vmap2(in_axes=0, out_axes=0)

        def fn(x):
            return x * 2.0

        mapped = builder(fn)
        self.assertIsInstance(mapped, StatefulMapping)
        xs = jnp.arange(3.0)
        self.assertTrue(jnp.allclose(mapped(xs), xs * 2.0))

    def test_vmap_rand(self):
        rng1 = brainstate.random.RandomState(42)
        rng2 = brainstate.random.RandomState(43)

        def f(x):
            a = brainstate.random.rand(2)
            b = rng1.randn(2)
            c = rng2.random(2)
            return a + x, b, c

        r = brainstate.transform.StatefulMapping(f)(jnp.asarray([1.0, 2.0]))
        print()
        print(r[0])
        print(r[1])
        print(r[2])


class TestVmapNewStates(unittest.TestCase):
    def test_new_states_are_vectorized(self):
        @vmap_new_states(in_axes=0, out_axes=0)
        def build(x):
            scratch = brainstate.ShortTermState(jnp.array(0.0), tag='scratch')
            scratch.value = scratch.value + x
            return scratch.value

        xs = jnp.arange(4.0)
        result_first = build(xs)
        result_second = build(xs)
        self.assertTrue(jnp.allclose(result_first, xs))
        self.assertTrue(jnp.allclose(result_second, xs))


class TestPmapIntegration(unittest.TestCase):
    @unittest.skipIf(jax.local_device_count() < 2, "Requires at least 2 devices")
    def test_pmap_stateful_execution(self):
        param = brainstate.ParamState(jnp.ones((4,)))

        # ``param`` is replicated across devices (broadcast input) and each device
        # updates its own copy, so it is scattered along axis 0 on output only.
        @pmap2(
            in_axes=0,
            out_axes=0,
            axis_name='devices',
            state_out_axes={0: filter.OfType(brainstate.ParamState)},
        )
        def update(delta):
            param.value = param.value + delta
            return param.value

        device_count = jax.local_device_count()
        deltas = jnp.arange(device_count * 4.0, dtype=param.value.dtype).reshape(device_count, 4)
        updated = update(deltas)
        self.assertEqual(updated.shape, (device_count, 4))
        self.assertTrue(jnp.all(updated >= 1.0))


class TestMapValidation(unittest.TestCase):
    """Tests for map() input-validation branches."""

    def test_map_no_inputs_raises(self):
        """map called with no array xs should raise ValueError."""
        with self.assertRaises(ValueError):
            map(lambda: None)

    def test_map_mismatched_lengths_raises(self):
        """map with xs of different leading lengths should raise ValueError."""
        xs_a = jnp.arange(3.0)
        xs_b = jnp.arange(5.0)
        with self.assertRaises(ValueError):
            map(lambda a, b: a + b, xs_a, xs_b)

    def test_map_invalid_batch_size_zero_raises(self):
        """map with batch_size=0 should raise ValueError."""
        xs = jnp.arange(4.0)
        with self.assertRaises(ValueError):
            map(lambda x: x, xs, batch_size=0)

    def test_map_invalid_batch_size_negative_raises(self):
        """map with batch_size=-1 should raise ValueError."""
        xs = jnp.arange(4.0)
        with self.assertRaises(ValueError):
            map(lambda x: x, xs, batch_size=-1)

    def test_map_batch_size_exact_no_remainder(self):
        """map with batch_size dividing length evenly takes the no-remainder path."""
        xs = jnp.arange(4.0)

        def fn(x):
            return x * 2.0

        result = map(fn, xs, batch_size=2)
        self.assertTrue(jnp.allclose(result, xs * 2.0))

    def test_map_batch_size_with_remainder(self):
        """map with batch_size not dividing length handles remainder correctly."""
        xs = jnp.arange(5.0)

        def fn(x):
            return x * 3.0

        result = map(fn, xs, batch_size=2)
        self.assertTrue(jnp.allclose(result, xs * 3.0))


class TestMapBatchedRNG(unittest.TestCase):
    """B5: batched map (``batch_size=...``) tolerates random-number use.

    A RandomState key-split registers as a State write, but RNG threads
    correctly through the batched path: vmap2 splits a distinct key per lane and
    the surrounding state-aware scan advances the global key across batches.
    Only *non-RNG* writes remain rejected.
    """

    def test_batched_map_allows_rng(self):
        """A function drawing randomness runs under batch_size and draws distinctly."""
        brainstate.random.seed(0)

        def rand_fn(x):
            return x + brainstate.random.normal()

        xs = jnp.zeros(6)
        out = map(rand_fn, xs, batch_size=3)
        self.assertEqual(out.shape, (6,))
        # every lane (within and across batches) draws a distinct value
        vals = np.asarray(out).round(6).tolist()
        self.assertEqual(len(set(vals)), 6)

    def test_batched_map_rng_deterministic(self):
        """Same seed -> identical batched-map randomness; different seed -> different."""

        def rand_fn(x):
            return x + brainstate.random.normal()

        xs = jnp.zeros(6)
        brainstate.random.seed(0)
        out_a = map(rand_fn, xs, batch_size=3)
        brainstate.random.seed(0)
        out_b = map(rand_fn, xs, batch_size=3)
        self.assertTrue(jnp.allclose(out_a, out_b))
        brainstate.random.seed(123)
        out_c = map(rand_fn, xs, batch_size=3)
        self.assertFalse(jnp.allclose(out_a, out_c))

    def test_batched_map_rng_remainder_path(self):
        """RNG threads through both the scan batches and the trailing remainder."""
        brainstate.random.seed(0)

        def rand_fn(x):
            return x + brainstate.random.normal()

        xs = jnp.zeros(5)  # 5 = 2*2 + 1 -> exercises the remainder branch
        out = map(rand_fn, xs, batch_size=2)
        self.assertEqual(out.shape, (5,))
        vals = np.asarray(out).round(6).tolist()
        self.assertEqual(len(set(vals)), 5)

    def test_batched_map_still_rejects_nonrng_write(self):
        """A genuine (non-RNG) State write under batch_size is still rejected."""
        counter = brainstate.ShortTermState(jnp.zeros(()))

        def writes_state(x):
            counter.value = counter.value + x
            return x

        xs = jnp.arange(4.0)
        with self.assertRaises(ValueError):
            map(writes_state, xs, batch_size=2)


class TestBatchAndRemainder(unittest.TestCase):
    """Tests for the _batch_and_remainder helper."""

    def test_no_leaves_raises(self):
        """_batch_and_remainder with an empty pytree should raise ValueError."""
        with self.assertRaises(ValueError):
            _batch_and_remainder((), 2)

    def test_mismatched_leaf_lengths_raises(self):
        """_batch_and_remainder with leaves of different lengths raises ValueError."""
        a = jnp.arange(3.0)
        b = jnp.arange(4.0)
        with self.assertRaises(ValueError):
            _batch_and_remainder({'a': a, 'b': b}, 2)

    def test_exact_division_returns_none_remainder(self):
        """_batch_and_remainder with length divisible by batch_size returns None as remainder."""
        xs = jnp.arange(6.0)
        scan_tree, remainder = _batch_and_remainder((xs,), 3)
        self.assertIsNone(remainder)

    def test_with_remainder_returns_remainder_pytree(self):
        """_batch_and_remainder with non-zero remainder returns the leftover slice."""
        xs = jnp.arange(5.0)
        scan_tree, remainder = _batch_and_remainder((xs,), 2)
        self.assertIsNotNone(remainder)
        # remainder should contain 1 element
        self.assertEqual(remainder[0].shape[0], 1)


class TestValidateLeadingLengths(unittest.TestCase):
    """Tests for the _validate_leading_lengths helper."""

    def test_empty_xs_raises(self):
        """_validate_leading_lengths with no array leaves should raise ValueError."""
        with self.assertRaises(ValueError):
            _validate_leading_lengths(())

    def test_mismatched_lengths_raises(self):
        """_validate_leading_lengths with inconsistent leading sizes should raise ValueError."""
        xs_a = jnp.arange(3.0)
        xs_b = jnp.arange(4.0)
        with self.assertRaises(ValueError):
            _validate_leading_lengths((xs_a, xs_b))

    def test_matching_lengths_returns_length(self):
        """_validate_leading_lengths with matching sizes returns the common length."""
        xs = jnp.arange(5.0)
        length = _validate_leading_lengths((xs, xs))
        self.assertEqual(length, 5)


class TestPmap2Decorator(unittest.TestCase):
    """Tests for pmap2() as a decorator (Missing fn path)."""

    def test_pmap2_returns_partial_when_fn_missing(self):
        """pmap2 called without fn returns a callable partial."""
        import functools
        result = pmap2(in_axes=0, out_axes=0)
        self.assertIsInstance(result, functools.partial)

    def test_pmap2_partial_creates_stateful_mapping(self):
        """Partial from pmap2 applied to a function produces a StatefulMapping."""
        decorator = pmap2(in_axes=0, out_axes=0)

        def fn(x):
            return x + 1.0

        mapped = decorator(fn)
        self.assertIsInstance(mapped, StatefulMapping)


class TestPmap2UnsupportedArgnums(unittest.TestCase):
    """B2: pmap2 must reject static_broadcasted_argnums / donate_argnums.

    The state-aware engine bundles the user's arguments into a single tuple and
    appends RNG keys + grouped state values before handing them to ``jax.pmap``.
    Any positional index the user supplies therefore addresses the *wrapper's*
    parameters, not their own -- silently broadcasting/donating the wrong thing.
    Rather than mis-apply them, pmap2 rejects both with a clear error.
    """

    def _fn(self, x):
        return x + 1.0

    def test_static_broadcasted_argnums_rejected_direct(self):
        with self.assertRaises(NotImplementedError):
            pmap2(self._fn, static_broadcasted_argnums=0)

    def test_static_broadcasted_argnums_rejected_iterable(self):
        with self.assertRaises(NotImplementedError):
            pmap2(self._fn, static_broadcasted_argnums=(0, 1))

    def test_donate_argnums_rejected_direct(self):
        with self.assertRaises(NotImplementedError):
            pmap2(self._fn, donate_argnums=0)

    def test_donate_argnums_rejected_iterable(self):
        with self.assertRaises(NotImplementedError):
            pmap2(self._fn, donate_argnums=(0,))

    def test_rejected_on_decorator_application(self):
        # decorator form: fn is Missing -> partial; the guard must fire when the
        # partial is finally applied to the function.
        decorator = pmap2(static_broadcasted_argnums=(0,))
        with self.assertRaises(NotImplementedError):
            decorator(self._fn)

    def test_empty_argnums_allowed(self):
        # positive control: the defaults () must NOT raise.
        mapped = pmap2(self._fn, in_axes=0, static_broadcasted_argnums=(), donate_argnums=())
        self.assertIsInstance(mapped, StatefulMapping)


class TestBuildNewStateResolver(unittest.TestCase):
    """Tests for _build_new_state_resolver covering all branches."""

    def test_none_key_in_state_out_axes(self):
        """state_out_axes with None key triggers the user_none branch."""
        ordered, axes_order = _build_new_state_resolver(
            {None: filter.OfType(NonBatchState)}
        )
        # axes_order should start with None
        self.assertIn(None, axes_order)

    def test_non_zero_axis_in_state_out_axes(self):
        """state_out_axes with axis=1 triggers the non-zero non-None axis branch."""
        ordered, axes_order = _build_new_state_resolver(
            {1: filter.OfType(brainstate.ShortTermState)}
        )
        self.assertIn(1, axes_order)

    def test_zero_axis_in_state_out_axes(self):
        """state_out_axes with axis=0 triggers the user[0] branch instead of catch-all."""
        ordered, axes_order = _build_new_state_resolver(
            {0: filter.OfType(brainstate.ShortTermState)}
        )
        self.assertIn(0, axes_order)

    def test_non_dict_state_out_axes_converted(self):
        """Non-dict state_out_axes is promoted to {0: ...}."""
        ordered, axes_order = _build_new_state_resolver(
            filter.OfType(brainstate.ShortTermState)
        )
        self.assertIn(0, axes_order)

    def test_none_state_out_axes(self):
        """None state_out_axes uses the default catch-all resolver."""
        ordered, axes_order = _build_new_state_resolver(None)
        self.assertIn(0, axes_order)
        self.assertIn(None, axes_order)


class TestMapNewStatesInternal(unittest.TestCase):
    """Tests for the internal _map_new_states helper."""

    def test_invalid_behavior_raises(self):
        """_map_new_states with an unrecognized behavior string raises ValueError."""
        from brainstate.transform._mapping2 import _map_new_states

        class M(brainstate.nn.Module):
            def init_all_states(self, **kw):
                self.x = brainstate.ShortTermState(jnp.zeros(()))

        with self.assertRaises(ValueError):
            _map_new_states('neither_vmap_nor_pmap', M(), {}, axis_size=2)

    def test_non_random_state_probe_hook_path(self):
        """vmap2_new_states with a pre-existing state read during init hits probe_hook's non-RandomState branch."""
        from brainstate.transform._mapping2 import _map_new_states

        # A pre-existing state that is read during init_all_states triggers probe_hook for
        # a non-RandomState, hitting the `return state._value` branch on line 735.
        shared = brainstate.ShortTermState(jnp.ones((2,)))

        class M(brainstate.nn.Module):
            def init_all_states(self, **kw):
                val = shared.value * 2.0  # reads shared (non-RandomState) during probe
                self.x = brainstate.ShortTermState(val)

        m = M()
        _map_new_states('vmap', m, {}, axis_size=3)
        self.assertEqual(m.x.value.shape[0], 3)


class TestVmap2NewStates(unittest.TestCase):
    """Tests for vmap2_new_states()."""

    def test_vmap2_new_states_no_axis_size_raises(self):
        """vmap2_new_states without axis_size raises ValueError."""

        class M(brainstate.nn.Module):
            def init_all_states(self, **kw):
                self.x = brainstate.ShortTermState(jnp.zeros(()))

        with self.assertRaises(ValueError):
            vmap2_new_states(M(), {})

    def test_vmap2_new_states_vectorizes_state(self):
        """vmap2_new_states expands state along axis 0 with the given axis_size."""

        class M(brainstate.nn.Module):
            def init_all_states(self, **kw):
                self.x = brainstate.ShortTermState(jnp.zeros(()))

        m = M()
        vmap2_new_states(m, {}, axis_size=4)
        self.assertEqual(m.x.value.shape[0], 4)

    def test_vmap2_new_states_with_random_state(self):
        """vmap2_new_states correctly splits random keys for random initializers."""

        class M(brainstate.nn.Module):
            def init_all_states(self, **kw):
                self.w = brainstate.ParamState(brainstate.random.randn(3))

        m = M()
        vmap2_new_states(m, {}, axis_size=5)
        self.assertEqual(m.w.value.shape[0], 5)

    def test_vmap2_new_states_with_none_key_state_out_axes(self):
        """vmap2_new_states with None key in state_out_axes replicates NonBatchState."""

        class M(brainstate.nn.Module):
            def init_all_states(self, **kw):
                self.x = brainstate.ShortTermState(jnp.zeros(()))

        m = M()
        result = vmap2_new_states(
            m, {}, axis_size=3,
            state_out_axes={None: filter.OfType(NonBatchState)}
        )
        # ShortTermState is not NonBatchState, so it should still be batched at axis 0
        self.assertIsInstance(result, dict)


class TestPmap2NewStates(unittest.TestCase):
    """Tests for pmap2_new_states()."""

    def test_pmap2_new_states_with_random_state(self):
        """pmap2_new_states initializes state across devices with random initialization."""

        class M(brainstate.nn.Module):
            def init_all_states(self, **kw):
                self.w = brainstate.ParamState(brainstate.random.randn(3))

        m = M()
        result = pmap2_new_states(m, {}, axis_size=jax.local_device_count())
        self.assertIsInstance(result, dict)
        self.assertEqual(m.w.value.shape[0], jax.local_device_count())

    def test_pmap2_new_states_default_axis_size(self):
        """pmap2_new_states without explicit axis_size uses local_device_count."""

        class M(brainstate.nn.Module):
            def init_all_states(self, **kw):
                self.w = brainstate.ParamState(brainstate.random.randn(2))

        m = M()
        result = pmap2_new_states(m, {})
        self.assertIsInstance(result, dict)
        self.assertEqual(m.w.value.shape[0], jax.local_device_count())

    @unittest.skipIf(jax.local_device_count() < 2, "Requires at least 2 devices")
    def test_pmap2_new_states_deterministic_param(self):
        """H19: pmap2_new_states must not crash when init draws no randomness.

        ``jax.pmap`` (unlike ``jax.vmap``) requires at least one mapped
        argument; with deterministic state init there are no RNG keys to map,
        so the no-rng pmap branch must supply a throwaway mapped argument.
        Pre-fix this raised ``ValueError: pmap requires at least one argument
        with a mapped axis``.
        """

        class M(brainstate.nn.Module):
            def init_all_states(self, **kw):
                self.w = brainstate.ParamState(jnp.ones(3))

        m = M()
        n = jax.local_device_count()
        result = pmap2_new_states(m, {}, axis_size=n)
        self.assertIsInstance(result, dict)
        # batched at axis 0 with size == number of devices
        self.assertEqual(m.w.value.shape, (n, 3))
        # each device receives the same deterministic init
        self.assertTrue(jnp.allclose(m.w.value, jnp.ones((n, 3))))

    @unittest.skipIf(jax.local_device_count() < 2, "Requires at least 2 devices")
    def test_pmap2_new_states_deterministic_default_axis_size(self):
        """H19: deterministic init with default axis_size (local_device_count)."""

        class M(brainstate.nn.Module):
            def init_all_states(self, **kw):
                self.count = brainstate.ShortTermState(jnp.zeros(()))

        m = M()
        result = pmap2_new_states(m, {})
        self.assertIsInstance(result, dict)
        self.assertEqual(m.count.value.shape, (jax.local_device_count(),))


class TestVmap2PytreeInAxes(unittest.TestCase):
    """B3: per-leaf (pytree-prefix) in_axes within a single positional arg."""

    def test_dict_arg_pytree_prefix_in_axes(self):
        def f(d):
            return d['a'] + d['b']

        arg = {'a': jnp.arange(3.), 'b': jnp.arange(3.) * 10}
        in_axes = ({'a': 0, 'b': None},)
        got = vmap2(f, in_axes=in_axes)(arg)
        expected = jax.vmap(f, in_axes=in_axes)(arg)
        self.assertTrue(jnp.allclose(got, expected))


class TestVmap2Collectives(unittest.TestCase):
    """B2: axis_name collectives (psum, axis_index) match jax.vmap."""

    def test_psum_matches_jax(self):
        def f(x):
            return x / jax.lax.psum(x, 'i')

        xs = jnp.arange(1., 5.)
        got = vmap2(f, in_axes=0, axis_name='i')(xs)
        expected = jax.vmap(f, in_axes=0, axis_name='i')(xs)
        self.assertTrue(jnp.allclose(got, expected))

    def test_axis_index_matches_jax(self):
        def f(x):
            return x + jax.lax.axis_index('i')

        xs = jnp.zeros(4)
        got = vmap2(f, in_axes=0, axis_name='i')(xs)
        expected = jax.vmap(f, in_axes=0, axis_name='i')(xs)
        self.assertTrue(jnp.allclose(got, expected))

    def test_psum_with_batched_state_write(self):
        s = brainstate.ShortTermState(jnp.zeros(4))

        def f(x):
            s.value = x / jax.lax.psum(x, 'i')
            return s.value

        out = vmap2(f, in_axes=0, axis_name='i', state_out_axes=s)(jnp.arange(1., 5.))
        self.assertTrue(jnp.allclose(out, jnp.array([0.1, 0.2, 0.3, 0.4])))
        self.assertTrue(jnp.allclose(s.value, jnp.array([0.1, 0.2, 0.3, 0.4])))


class TestVmap2DiscoverySkip(unittest.TestCase):
    """B5: a stateless cold call skips the discovery vmap (probe + execution only)."""

    def test_stateless_cold_call_skips_discovery(self):
        # Stateless function: probe (eager) + execution = 2 body executions on a
        # cold call. Before the fix it was 3 (probe + discovery + execution).
        count = {'n': 0}

        def f(x):
            count['n'] += 1
            return x * 2

        out = vmap2(f, in_axes=0)(jnp.arange(4.))
        self.assertTrue(jnp.allclose(out, jnp.arange(4.) * 2))
        self.assertEqual(count['n'], 2)


class TestVmap2Kwargs(unittest.TestCase):
    """B1: dynamic kwargs are mapped over axis 0 (jax.vmap parity)."""

    def test_kwarg_mapped_over_axis0_matches_jax(self):
        def f(x, y):
            return x + y

        x = jnp.arange(3.)
        y = jnp.arange(3.) * 10
        got = vmap2(f, in_axes=0)(x, y=y)
        expected = jax.vmap(f, in_axes=0)(x, y=y)
        self.assertEqual(got.shape, (3,))
        self.assertTrue(jnp.allclose(got, expected))
        self.assertTrue(jnp.allclose(got, jnp.array([0., 11., 22.])))

    def test_kwargs_only_no_positional(self):
        f = lambda *, y: y * 2
        got = vmap2(f, in_axes=0)(y=jnp.arange(3.))
        self.assertTrue(jnp.allclose(got, jnp.arange(3.) * 2))

    def test_state_write_with_mapped_kwarg(self):
        s = brainstate.ShortTermState(jnp.zeros(3))

        def f(x, scale):
            s.value = x * scale
            return s.value

        out = vmap2(f, in_axes=0, state_out_axes=s)(jnp.arange(3.), scale=jnp.ones(3) * 5)
        self.assertTrue(jnp.allclose(out, jnp.arange(3.) * 5))
        self.assertTrue(jnp.allclose(s.value, jnp.arange(3.) * 5))


class TestMapState(unittest.TestCase):
    """B4: sequential map handles state; batched map rejects stateful f clearly."""

    def test_sequential_map_accumulates_state(self):
        acc = brainstate.ShortTermState(jnp.zeros(()))

        def f(x):
            acc.value = acc.value + x
            return x * 2

        out = map(f, jnp.arange(6.))
        self.assertTrue(jnp.allclose(out, jnp.arange(6.) * 2))
        self.assertTrue(jnp.allclose(acc.value, 15.0))

    def test_stateless_batched_map_works(self):
        out = map(lambda x: x * x, jnp.arange(6.), batch_size=4)
        self.assertTrue(jnp.allclose(out, jnp.arange(6.) ** 2))

    def test_stateful_batched_map_raises_clear_error(self):
        acc = brainstate.ShortTermState(jnp.zeros(()))

        def f(x):
            acc.value = acc.value + x
            return x * 2

        with self.assertRaises(ValueError) as ctx:
            map(f, jnp.arange(6.), batch_size=3)
        self.assertIn("batch_size", str(ctx.exception))
        self.assertIn("State", str(ctx.exception))


class TestVmap2JaxParitySweep(unittest.TestCase):
    """vmap2 must match jax.vmap on stateless functions across feature axes."""

    def _assert_parity(self, f, *args, **kw):
        got = vmap2(f, **kw)(*args)
        expected = jax.vmap(f, **kw)(*args)
        self.assertTrue(jnp.allclose(got, expected), f"mismatch: {got} vs {expected}")

    def test_basic(self):
        self._assert_parity(lambda x: x * 2, jnp.arange(4.), in_axes=0)

    def test_in_axes_nonleading(self):
        self._assert_parity(lambda v: v.sum(), jnp.arange(6.).reshape(2, 3), in_axes=1)

    def test_out_axes(self):
        self._assert_parity(lambda v: v * 2, jnp.arange(6.).reshape(3, 2), in_axes=0, out_axes=1)

    def test_tuple_in_axes_with_none(self):
        self._assert_parity(lambda a, b: a + b, jnp.arange(3.), jnp.array(10.), in_axes=(0, None))

    def test_pytree_prefix_in_axes(self):
        f = lambda d: d['a'] + d['b']
        self._assert_parity(f, {'a': jnp.arange(3.), 'b': jnp.arange(3.) * 10},
                            in_axes=({'a': 0, 'b': None},))

    def test_kwargs_axis0(self):
        x, y = jnp.arange(3.), jnp.arange(3.) * 10
        got = vmap2(lambda a, y: a + y, in_axes=0)(x, y=y)
        expected = jax.vmap(lambda a, y: a + y, in_axes=0)(x, y=y)
        self.assertTrue(jnp.allclose(got, expected))

    def test_collective_psum(self):
        self._assert_parity(lambda x: x / jax.lax.psum(x, 'i'), jnp.arange(1., 5.),
                            in_axes=0, axis_name='i')

    def test_nested_vmap(self):
        x = jnp.arange(6.).reshape(2, 3)
        got = vmap2(vmap2(lambda v: v.sum(), in_axes=0), in_axes=0)(x)
        expected = jax.vmap(jax.vmap(lambda v: v.sum(), in_axes=0), in_axes=0)(x)
        self.assertTrue(jnp.allclose(got, expected))


if __name__ == "__main__":
    unittest.main()


class TestFailedNewStatesMappingRestoresRng(unittest.TestCase):
    """A failure inside the mapped init pass must not leave key tracers in
    the global random state (audit M5)."""

    def test_vmap2_new_states_failure_restores_rng(self):
        brainstate.random.seed(0)

        class _Stub:
            def __init__(self):
                self.calls = 0

            def init_all_states(self):
                self.calls += 1
                if self.calls >= 2:  # first call: probe; second: mapped pass
                    raise RuntimeError('boom')
                self.s = brainstate.ShortTermState(brainstate.random.randn(3))

        stub = _Stub()
        with self.assertRaises(RuntimeError):
            brainstate.transform.vmap2_new_states(stub, {}, axis_size=4)
        self.assertFalse(isinstance(brainstate.random.DEFAULT.value, jax.core.Tracer))
