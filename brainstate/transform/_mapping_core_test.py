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

"""Unit tests for the shared mapping engine in ``_mapping_core``."""

import unittest

import jax
import jax.numpy as jnp

import brainstate
import brainstate.random
from brainstate._error import BatchAxisError
from brainstate.transform._mapping_core import (
    _flatten_in_out_states,
    _remove_axis,
    _get_batch_size,
    _format_state_axes,
    normalize_state_axes,
    make_identity_predicate,
    coerce_axis_value_to_predicate,
    split_rng_keys,
    unwind_new_state_levels,
    leaf_batch_dim,
    state_map_transform,
)
from brainstate.util import filter


class TestHelpersMoved(unittest.TestCase):
    """The Phase 1 helpers must keep their original behaviour after the move."""

    def test_flatten_none(self):
        a, b = _flatten_in_out_states(None)
        self.assertEqual(a, {})
        self.assertEqual(b, {})

    def test_flatten_list_of_states(self):
        s1 = brainstate.ShortTermState(jnp.zeros(2))
        s2 = brainstate.ShortTermState(jnp.zeros(2))
        axis_to_states, state_to_axis = _flatten_in_out_states([s1, s2])
        self.assertEqual(state_to_axis[s1], 0)
        self.assertEqual(state_to_axis[s2], 0)

    def test_remove_axis(self):
        x = jnp.arange(6).reshape(2, 3)
        self.assertEqual(_remove_axis(x, 0).shape, (3,))
        self.assertEqual(_remove_axis(x, 1).shape, (2,))
        self.assertEqual(_remove_axis(x, -1).shape, (2,))

    def test_get_batch_size(self):
        args = (jnp.zeros((4, 3)),)
        self.assertEqual(_get_batch_size(args, 0, {}), 4)

    def test_get_batch_size_axis_size_fallback(self):
        self.assertEqual(_get_batch_size((), 0, {}, axis_size=7), 7)

    def test_format_state_axes_adds_in_as_out(self):
        s = brainstate.ShortTermState(jnp.zeros(2))
        _, in_to_axis, axis_to_out, out_to_axis = _format_state_axes([s], None)
        self.assertEqual(out_to_axis[s], 0)


class TestNormalizeStateAxes(unittest.TestCase):
    def test_none(self):
        self.assertEqual(normalize_state_axes(None), {})

    def test_bare_filter_goes_to_axis0(self):
        spec = normalize_state_axes(filter.OfType(brainstate.ShortTermState))
        self.assertIn(0, spec)

    def test_dict_passthrough(self):
        spec = normalize_state_axes({1: filter.OfType(brainstate.ParamState)})
        self.assertIn(1, spec)

    def test_state_instance(self):
        s = brainstate.ShortTermState(jnp.zeros(2))
        spec = normalize_state_axes(s)
        self.assertTrue(spec[0](tuple(), s))
        other = brainstate.ShortTermState(jnp.zeros(2))
        self.assertFalse(spec[0](tuple(), other))

    def test_identity_predicate(self):
        s = brainstate.ShortTermState(jnp.zeros(2))
        pred = make_identity_predicate(s)
        self.assertTrue(pred(tuple(), s))

    def test_coerce_iterable_of_states(self):
        s1 = brainstate.ShortTermState(jnp.zeros(2))
        s2 = brainstate.ShortTermState(jnp.zeros(2))
        pred = coerce_axis_value_to_predicate([s1, s2])
        self.assertTrue(pred(tuple(), s1))
        self.assertTrue(pred(tuple(), s2))


class TestRngAndStackLevel(unittest.TestCase):
    def test_split_rng_keys_shapes(self):
        rng = brainstate.random.RandomState(0)
        keys, backups = split_rng_keys([rng], 5)
        self.assertEqual(keys[0].shape[0], 5)
        self.assertEqual(len(backups), 1)

    def test_unwind_new_state_levels(self):
        s = brainstate.ShortTermState(jnp.zeros(2))
        s.stack_level = 3
        unwind_new_state_levels([s], base_level=1)
        self.assertEqual(s.stack_level, 1)

    def test_unwind_no_negative(self):
        s = brainstate.ShortTermState(jnp.zeros(2))
        s.stack_level = 1
        unwind_new_state_levels([s], base_level=5)
        self.assertEqual(s.stack_level, 1)


class TestLeafBatchDim(unittest.TestCase):
    def test_unbatched(self):
        self.assertIsNone(leaf_batch_dim(jnp.zeros(3)))


class TestStateMapEngine(unittest.TestCase):
    def test_basic_counter(self):
        counter = brainstate.ShortTermState(jnp.zeros(3))

        def accumulate(x):
            counter.value = counter.value + x
            return counter.value

        mapped = state_map_transform(
            accumulate, in_axes=0, out_axes=0,
            state_in_axes={0: filter.OfType(brainstate.ShortTermState)},
            state_out_axes={0: filter.OfType(brainstate.ShortTermState)},
        )
        xs = jnp.asarray([1., 2., 3.])
        result = mapped(xs)
        self.assertTrue(jnp.allclose(result, xs))
        self.assertTrue(jnp.allclose(counter.value, xs))

    def test_auto_dim_undeclared_write(self):
        w = brainstate.ShortTermState(jnp.zeros(3))

        def f(x):
            w.value = w.value + x
            return w.value

        mapped = state_map_transform(f, in_axes=0, out_axes=0)
        out = mapped(jnp.asarray([10., 20., 30.]))
        self.assertEqual(out.shape, (3, 3))
        self.assertEqual(w.value.shape, (3, 3))

    def test_raise_policy(self):
        w = brainstate.ShortTermState(jnp.zeros(3))

        def f(x):
            w.value = w.value + x
            return w.value

        mapped = state_map_transform(
            f, in_axes=0, out_axes=0, unexpected_out_state_mapping='raise'
        )
        with self.assertRaises(BatchAxisError):
            mapped(jnp.asarray([1., 2., 3.]))

    def test_rng_consumed_once(self):
        rng = brainstate.random.RandomState(0)

        def f(x):
            return x + rng.randn(2)

        mapped = state_map_transform(f, in_axes=0, out_axes=0)
        out = mapped(jnp.zeros((4, 2)))
        self.assertEqual(out.shape, (4, 2))
        self.assertFalse(jnp.allclose(out[0], out[1]))

    def test_readonly_param_broadcast(self):
        p = brainstate.ParamState(jnp.ones((2,)))

        def f(x):
            return x @ p.value

        mapped = state_map_transform(f, in_axes=0, out_axes=0)
        out = mapped(jnp.ones((5, 2)))
        self.assertEqual(out.shape, (5,))
        self.assertEqual(p.value.shape, (2,))

    def test_warm_cache_reuse(self):
        counter = brainstate.ShortTermState(jnp.zeros(3))

        def accumulate(x):
            counter.value = counter.value + x
            return counter.value

        mapped = state_map_transform(
            accumulate, in_axes=0, out_axes=0,
            state_in_axes=counter, state_out_axes=counter,
        )
        xs = jnp.asarray([1., 1., 1.])
        mapped(xs)
        mapped(xs)
        self.assertTrue(jnp.allclose(counter.value, jnp.full((3,), 2.0)))

    def test_grad_inside_vmap(self):
        from brainstate.transform import grad

        w = brainstate.ParamState(jnp.array(2.0))

        def loss(x):
            return (x ** 2) * w.value

        def per_lane(x):
            return grad(loss, grad_states=w)(x)

        mapped = state_map_transform(per_lane, in_axes=0, out_axes=0)
        xs = jnp.asarray([1., 2., 3.])
        self.assertTrue(jnp.allclose(mapped(xs), xs ** 2))

    def test_cond_inside_vmap(self):
        from brainstate.transform import cond

        def f(x):
            return cond(x > 0, lambda: x * 10.0, lambda: x * -1.0)

        mapped = state_map_transform(f, in_axes=0, out_axes=0)
        out = mapped(jnp.asarray([1.0, -2.0, 3.0]))
        self.assertTrue(jnp.allclose(out, jnp.asarray([10.0, 2.0, 30.0])))

    def test_scan_inside_vmap(self):
        from brainstate.transform import scan

        def step(carry, x):
            return carry + x, carry + x

        def f(xs_lane):
            _, ys = scan(step, jnp.array(0.0), xs_lane)
            return ys

        mapped = state_map_transform(f, in_axes=0, out_axes=0)
        data = jnp.arange(6.).reshape(2, 3)
        self.assertTrue(jnp.allclose(mapped(data), jnp.cumsum(data, axis=1)))

    def test_nested_vmap(self):
        inner = state_map_transform(lambda x: x + 1.0, in_axes=0, out_axes=0)
        outer = state_map_transform(lambda row: inner(row), in_axes=0, out_axes=0)
        out = outer(jnp.ones((2, 3)))
        self.assertTrue(jnp.allclose(out, jnp.ones((2, 3)) * 2.0))

    def test_out_axis_placement(self):
        # The function return is placed at axis 0, but the state is stored with
        # its batch axis moved to position 1 via state_out_axes.
        s = brainstate.ShortTermState(jnp.zeros(5))

        def f(x):
            s.value = s.value + x
            return s.value

        mapped = state_map_transform(
            f, in_axes=0, out_axes=0, state_out_axes={1: s},
        )
        xs = jnp.ones((4, 5))  # mapped (lane) axis is leading, size 4
        out = mapped(xs)
        self.assertEqual(out.shape, (4, 5))      # return: batch at axis 0
        self.assertEqual(s.value.shape, (5, 4))  # state: batch moved to axis 1


if __name__ == "__main__":
    unittest.main()
