import unittest

import jax
import jax.numpy as jnp

import brainstate
import brainstate.random
from brainstate.transform import StatefulMapping, vmap2, vmap_new_states, pmap2, map
from brainstate.util import filter


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

        @pmap2(
            in_axes=0,
            out_axes=0,
            axis_name='devices',
            state_in_axes={0: filter.OfType(brainstate.ParamState)},
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


if __name__ == "__main__":
    unittest.main()
