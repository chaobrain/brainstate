import unittest

import jax
import jax.numpy as jnp

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
        """StatefulMapping accepts an int for static_argnums and wraps it in a tuple."""
        sm = StatefulMapping(lambda x: x, static_argnums=2)
        self.assertEqual(sm.static_argnums, (2,))

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
