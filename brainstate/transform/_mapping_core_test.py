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


class TestGetBatchSizeNoneAxis(unittest.TestCase):
    """Cover the axis is None branch in _get_batch_size (line 200)."""

    def test_skip_none_axis_in_states(self):
        """States with axis=None are skipped; batch size falls through to axis_size."""
        s = brainstate.ShortTermState(jnp.zeros((3, 4)))
        # axis=None means broadcast; no batch size can be inferred from it
        batch = _get_batch_size((), 0, {None: [s]}, axis_size=5)
        self.assertEqual(batch, 5)


class TestLeafBatchDimEdgeCases(unittest.TestCase):
    """Cover empty-leaves (line 396) and inconsistent-dims (line 398) paths."""

    def test_empty_value_returns_none(self):
        """leaf_batch_dim returns None when the pytree has no leaves."""
        # An empty tuple is a valid pytree with no leaves.
        result = leaf_batch_dim(())
        self.assertIsNone(result)

    def test_inconsistent_batch_dims_raises(self):
        """leaf_batch_dim raises BatchAxisError when leaves disagree on batch dim."""
        from brainstate._compatible_import import BatchTracer

        class FakeBatchTracer:
            """Minimal stand-in that looks like a BatchTracer to leaf_batch_dim."""
            def __init__(self, batch_dim):
                self.batch_dim = batch_dim

        # Monkey-patch the type check so our fake objects are recognised.
        import brainstate.transform._mapping_core as mc
        original_bt = mc.BatchTracer
        mc.BatchTracer = FakeBatchTracer
        try:
            t0 = FakeBatchTracer(0)
            t1 = FakeBatchTracer(1)
            from brainstate._error import BatchAxisError as BAE
            with self.assertRaises(BAE):
                leaf_batch_dim([t0, t1])
        finally:
            mc.BatchTracer = original_bt


class TestProbeStatesBroadcastInput(unittest.TestCase):
    """Cover axis=None (broadcast) input predicate path (line 460)."""

    def test_broadcast_state_not_stripped(self):
        """A state matched with axis=None is left unchanged during the probe."""
        param = brainstate.ParamState(jnp.ones(3))

        def f(x):
            return x + param.value

        # axis=None means broadcast; predicate matches param but strips nothing
        mapped = state_map_transform(
            f,
            in_axes=0, out_axes=0,
            state_in_axes={None: filter.OfType(brainstate.ParamState)},
        )
        out = mapped(jnp.ones((4, 3)))
        self.assertEqual(out.shape, (4, 3))
        # param should still be shape (3,)
        self.assertEqual(param.value.shape, (3,))


class TestBuildPlanInOutAxisMismatch(unittest.TestCase):
    """Cover error when in_state and out_state axes disagree (line 578)."""

    def test_in_out_axis_mismatch_raises(self):
        """state_map_transform raises BatchAxisError when in/out axes disagree."""
        s = brainstate.ShortTermState(jnp.zeros((4, 3)))

        def f(x):
            s.value = s.value + x
            return s.value

        from brainstate._error import BatchAxisError
        # in_axis=0 but out_axis=1 for the same state -> error
        mapped = state_map_transform(
            f,
            in_axes=0, out_axes=0,
            state_in_axes={0: s},
            state_out_axes={1: s},  # mismatch
        )
        with self.assertRaises(BatchAxisError):
            mapped(jnp.ones((4, 3)))


class TestBuildPlanMatchedBroadcastOut(unittest.TestCase):
    """Cover matched out_axis=None write path (lines 590-591, 595-596, 678)."""

    def test_matched_broadcast_out_state(self):
        """Declared out_state with axis=None is restored from a single lane (broadcast)."""
        shared = brainstate.ShortTermState(jnp.array(0.0))

        def f(x):
            # Write a constant scalar across all lanes — not a batched value
            shared.value = jnp.array(7.0)
            return x

        # state_out_axes={None: ...} means "broadcast restore from lane 0"
        mapped = state_map_transform(
            f,
            in_axes=0, out_axes=0,
            state_out_axes={None: shared},
        )
        out = mapped(jnp.ones((3, 2)))
        self.assertEqual(out.shape, (3, 2))
        # shared is restored from a single lane; value should be 7.0
        self.assertEqual(float(shared.value), 7.0)

    def test_matched_out_state_no_detected_dim(self):
        """Declared out_state with axis=1 but no batched write -> broadcast restore."""
        shared = brainstate.ShortTermState(jnp.array(0.0))

        def f(x):
            # Write a broadcast (non-batched) value
            shared.value = jnp.array(42.0)
            return x

        # Declare with axis=1 but the actual write is broadcast (det=None)
        mapped = state_map_transform(
            f,
            in_axes=0, out_axes=0,
            state_out_axes={1: shared},
        )
        out = mapped(jnp.ones((4, 3)))
        self.assertEqual(out.shape, (4, 3))


class TestBuildPlanWarnPolicy(unittest.TestCase):
    """Cover 'warn' unexpected_out_state_mapping policy (lines 610-615)."""

    def test_warn_policy_issues_warning(self):
        """unexpected_out_state_mapping='warn' issues a UserWarning."""
        w = brainstate.ShortTermState(jnp.zeros(3))

        def f(x):
            w.value = w.value + x
            return w.value

        mapped = state_map_transform(
            f, in_axes=0, out_axes=0,
            unexpected_out_state_mapping='warn',
        )
        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = mapped(jnp.asarray([1., 2., 3.]))
        self.assertTrue(any(issubclass(c.category, UserWarning) for c in caught))
        self.assertEqual(out.shape, (3, 3))


class TestBuildPlanIgnorePolicy(unittest.TestCase):
    """Cover 'ignore' unexpected_out_state_mapping policy (line 616-617)."""

    def test_ignore_policy_silently_scatters(self):
        """unexpected_out_state_mapping='ignore' silently scatters batched writes."""
        w = brainstate.ShortTermState(jnp.zeros(3))

        def f(x):
            w.value = w.value + x
            return w.value

        mapped = state_map_transform(
            f, in_axes=0, out_axes=0,
            unexpected_out_state_mapping='ignore',
        )
        out = mapped(jnp.asarray([1., 2., 3.]))
        self.assertEqual(out.shape, (3, 3))


class TestBuildPlanInvalidPolicy(unittest.TestCase):
    """Cover invalid unexpected_out_state_mapping value (lines 619)."""

    def test_invalid_policy_raises_value_error(self):
        """An invalid unexpected_out_state_mapping raises ValueError."""
        w = brainstate.ShortTermState(jnp.zeros(3))

        def f(x):
            w.value = w.value + x
            return w.value

        mapped = state_map_transform(
            f, in_axes=0, out_axes=0,
            unexpected_out_state_mapping='invalid_policy',
        )
        with self.assertRaises(ValueError, msg="Invalid value for unexpected_out_state_mapping"):
            mapped(jnp.asarray([1., 2., 3.]))


class TestBuildPlanBroadcastWrite(unittest.TestCase):
    """Cover broadcast write -> oth_out_states path (line 625)."""

    def test_broadcast_write_restored_from_single_lane(self):
        """Undeclared broadcast write (det=None) is restored from one lane."""
        shared = brainstate.ShortTermState(jnp.array(0.0))

        def f(x):
            # Write a non-batched value (same across lanes -> det=None)
            shared.value = jnp.array(99.0)
            return x

        mapped = state_map_transform(
            f, in_axes=0, out_axes=0,
        )
        out = mapped(jnp.ones((3,)))
        # shared should have the broadcast value restored
        self.assertEqual(float(shared.value), 99.0)


class TestStateMapTransformListInAxes(unittest.TestCase):
    """Cover list in_axes conversion (line 744)."""

    def test_list_in_axes_converted(self):
        """state_map_transform accepts list in_axes and converts to tuple."""
        def f(x, y):
            return x + y

        mapped = state_map_transform(
            f,
            in_axes=[0, 0], out_axes=0,
        )
        # Two args each shape (3,): 3 lanes
        xs = jnp.ones(3)
        ys = jnp.ones(3) * 2
        out = mapped(xs, ys)
        self.assertEqual(out.shape, (3,))
        self.assertTrue(jnp.allclose(out, jnp.full(3, 3.0)))


class TestStateMapTransformMappingKwargs(unittest.TestCase):
    """Cover non-None mapping_kwargs path (line 740)."""

    def test_mapping_kwargs_forwarded(self):
        """mapping_kwargs dict is forwarded to the underlying mapping primitive."""
        s = brainstate.ShortTermState(jnp.zeros(3))

        def f(x):
            s.value = s.value + x
            return s.value

        import functools
        # Pass spmd_axis_name=None through mapping_kwargs (benign but exercises path)
        mapped = state_map_transform(
            f, in_axes=0, out_axes=0,
            state_in_axes=s, state_out_axes=s,
            mapping_fn=functools.partial(jax.vmap, spmd_axis_name=None),
            mapping_kwargs={},
        )
        out = mapped(jnp.ones(3))
        self.assertEqual(out.shape, (3,))


class TestStateMapTransformInAxesTupleMismatch(unittest.TestCase):
    """Cover in_axes tuple length mismatch error (lines 750-755)."""

    def test_in_axes_tuple_length_mismatch_raises(self):
        """tuple in_axes with wrong length raises ValueError."""
        import pytest

        def f(x):
            return x

        mapped = state_map_transform(f, in_axes=(0, 0), out_axes=0)
        with pytest.raises(ValueError):
            mapped(jnp.ones(3))


class TestCompileStatefulFunction(unittest.TestCase):
    """Cover _compile_stateful_function (lines 147-175)."""

    def test_compile_stateful_function_no_states(self):
        """_compile_stateful_function with empty state_vals and int in_axes."""
        from brainstate.transform._mapping_core import _compile_stateful_function
        from brainstate.transform._make_jaxpr import StatefulFunction

        # The function receives (state_vals, args) where args is a tuple.
        # With no states and int in_axes, only the args axis-stripping path runs.
        def inner(state_vals, args):
            (x,) = args
            return x * 2.0

        sf = StatefulFunction(inner)
        x_batched = jnp.ones((4, 3))  # shape (4, 3): 4 lanes each width 3

        cache_key = _compile_stateful_function(
            sf,
            in_axes=([], 0),          # in_axes_st=[], in_axes=0
            args=([], (x_batched,)),  # state_vals=[], args=(x_batched,)
        )
        self.assertIsNotNone(cache_key)

    def test_compile_stateful_function_with_state_vals(self):
        """_compile_stateful_function strips state axis when state_vals provided."""
        from brainstate.transform._mapping_core import _compile_stateful_function
        from brainstate.transform._make_jaxpr import StatefulFunction

        def inner(state_vals, args):
            (sv,) = state_vals
            (x,) = args
            return sv + x

        sf = StatefulFunction(inner)
        # Batched state: shape (4, 2) with axis 0 -> strip to (2,)
        state_vals_batched = [jnp.ones((4, 2))]
        x_batched = jnp.ones((4, 2))

        cache_key = _compile_stateful_function(
            sf,
            in_axes=([0], 0),
            args=(state_vals_batched, (x_batched,)),
        )
        self.assertIsNotNone(cache_key)

    def test_compile_stateful_function_tuple_in_axes(self):
        """_compile_stateful_function handles tuple in_axes with None entry."""
        from brainstate.transform._mapping_core import _compile_stateful_function
        from brainstate.transform._make_jaxpr import StatefulFunction

        def inner(state_vals, args):
            x, y = args
            return x + y

        sf = StatefulFunction(inner)
        x = jnp.ones((4, 2))  # batched on axis 0
        y = jnp.ones(2)        # broadcast (axis=None -> not stripped)

        cache_key = _compile_stateful_function(
            sf,
            in_axes=([], (0, None)),
            args=([], (x, y)),
        )
        self.assertIsNotNone(cache_key)

    def test_compile_stateful_function_tuple_mismatch_raises(self):
        """_compile_stateful_function raises when tuple in_axes length mismatches args."""
        import pytest
        from brainstate.transform._mapping_core import _compile_stateful_function
        from brainstate.transform._make_jaxpr import StatefulFunction

        def inner(state_vals, args):
            (x,) = args
            return x

        sf = StatefulFunction(inner)
        x = jnp.ones((4, 3))

        with pytest.raises(ValueError):
            _compile_stateful_function(
                sf,
                in_axes=([], (0, 0)),  # 2-entry tuple but only 1 arg
                args=([], (x,)),
            )

    def test_compile_stateful_function_none_in_axes(self):
        """_compile_stateful_function handles in_axes=None (no stripping of args)."""
        from brainstate.transform._mapping_core import _compile_stateful_function
        from brainstate.transform._make_jaxpr import StatefulFunction

        def inner(state_vals, args):
            (x,) = args
            return x * 3.0

        sf = StatefulFunction(inner)
        x = jnp.ones((4, 3))  # not stripped since in_axes=None

        cache_key = _compile_stateful_function(
            sf,
            in_axes=([], None),  # None -> no stripping
            args=([], (x,)),
        )
        self.assertIsNotNone(cache_key)


class TestRemoveAxisTree(unittest.TestCase):
    """Cover _remove_axis_tree (369-371) and _strip_args None path (line 377)."""

    def test_remove_axis_tree_direct(self):
        """_remove_axis_tree strips axis from every leaf of a pytree."""
        from brainstate.transform._mapping_core import _remove_axis_tree
        value = {'a': jnp.ones((3, 4)), 'b': jnp.ones((3, 5))}
        result = _remove_axis_tree(value, axis=0)
        self.assertEqual(result['a'].shape, (4,))
        self.assertEqual(result['b'].shape, (5,))

    def test_strip_args_none_in_axes(self):
        """_strip_args with in_axes=None returns args unchanged (line 377)."""
        from brainstate.transform._mapping_core import _strip_args
        args = (jnp.ones((4, 3)), jnp.ones((2, 5)))
        result = _strip_args(args, in_axes=None)
        self.assertIs(result, args)  # same object returned

    def test_remove_axis_tree_via_probe_states(self):
        """_remove_axis_tree is called for batched input states in the probe."""
        # A state with a pytree value (dict) that has its axis stripped in probe
        s = brainstate.ShortTermState({'a': jnp.zeros((3, 2)), 'b': jnp.zeros((3, 5))})

        def f(x):
            return x + s.value['a']

        mapped = state_map_transform(
            f,
            in_axes=0, out_axes=0,
            state_in_axes={0: s},
            state_out_axes={0: s},
        )
        out = mapped(jnp.ones((3, 2)))
        self.assertEqual(out.shape, (3, 2))


class TestGetBatchSizeBranchCoverage(unittest.TestCase):
    """Cover remaining branch paths in _get_batch_size."""

    def test_in_axes_none_falls_through(self):
        """in_axes=None skips the arg-scan loop entirely."""
        # With in_axes=None, no batch sizes from args; use axis_size
        batch = _get_batch_size(
            (jnp.ones((4, 2)),),
            in_axes=None,
            in_states={},
            axis_size=7,
        )
        self.assertEqual(batch, 7)

    def test_in_axes_tuple_with_none_entry(self):
        """Tuple in_axes with None entries skips those args."""
        # Only the non-None axis contributes a batch size
        args = (jnp.ones((4, 2)), jnp.ones((3, 5)))
        batch = _get_batch_size(args, in_axes=(0, None), in_states={})
        self.assertEqual(batch, 4)

    def test_empty_arg_leaves_skipped(self):
        """Args with no leaves (empty pytree) don't contribute to batch_sizes."""
        # An empty tuple has no leaves
        args = ((), jnp.ones((5,)))
        batch = _get_batch_size(args, in_axes=(0, 0), in_states={})
        self.assertEqual(batch, 5)

    def test_in_states_empty_leaves_skipped(self):
        """States with no leaves in their value don't contribute to batch_sizes."""
        # A state whose value has no leaves (empty dict pytree) should be skipped.
        # Cover the `if len(state_leaves):` False branch (203->201).
        class _EmptyState:
            value = {}  # empty pytree -> no leaves

        in_states = {0: [_EmptyState()]}
        # No batch size from states, and no args -> need axis_size fallback
        batch = _get_batch_size((), 0, in_states, axis_size=6)
        self.assertEqual(batch, 6)

    def test_in_states_with_none_axis_and_valid_arg(self):
        """None axis in in_states is skipped; batch size comes from args."""
        s = brainstate.ShortTermState(jnp.zeros((4,)))
        in_states = {None: [s]}  # axis=None -> skipped
        args = (jnp.ones((4,)),)
        batch = _get_batch_size(args, 0, in_states)
        self.assertEqual(batch, 4)

    def test_in_states_none_uses_args(self):
        """in_states=None is handled by skipping the state loop (197->206 False branch)."""
        # When in_states is None, the `if in_states is not None:` branch is False.
        args = (jnp.ones((5, 2)),)
        batch = _get_batch_size(args, in_axes=0, in_states=None)
        self.assertEqual(batch, 5)


class TestMatchedStateNotWrite(unittest.TestCase):
    """Cover matched/no-write branches (590->597, 595->597)."""

    def test_matched_out_state_read_only_null_axis(self):
        """Declared out_state axis=None that is read-only: matched+not-write (590->597)."""
        ro_state = brainstate.ShortTermState(jnp.array(5.0))

        def f(x):
            # Only read ro_state, do not write it
            return x + ro_state.value

        # Declare ro_state as an out_state with axis=None; but it's never written.
        # Path: matched=True, out_axis=None, is_write=False -> skip (590->597).
        mapped = state_map_transform(
            f,
            in_axes=0, out_axes=0,
            state_out_axes={None: ro_state},
        )
        out = mapped(jnp.ones(3))
        self.assertEqual(out.shape, (3,))
        # ro_state unchanged since it was not written
        self.assertEqual(float(ro_state.value), 5.0)

    def test_matched_out_state_read_only_with_axis(self):
        """Declared out_state axis=0 that is read-only: matched+not-write (595->597)."""
        ro_state = brainstate.ShortTermState(jnp.array(5.0))

        def f(x):
            # Only read ro_state (scalar), do not write it
            return x * ro_state.value

        # Declare ro_state as out_state axis=0, but it is never written.
        # Path: matched=True, out_axis=0, det=None (never written), is_write=False.
        # Since det is None and is_write is False, this exercises the `elif is_write:`
        # False branch (595->597) -- state is simply skipped.
        mapped = state_map_transform(
            f,
            in_axes=0, out_axes=0,
            state_out_axes={0: ro_state},
        )
        out = mapped(jnp.ones(3))
        self.assertEqual(out.shape, (3,))
        # ro_state value should be unchanged (scalar 5.0)
        self.assertEqual(float(ro_state.value), 5.0)

    def test_matched_out_state_axis_not_detected(self):
        """Declared out_state axis=1 but write is broadcast (det=None) -> oth path."""
        shared = brainstate.ShortTermState(jnp.array(0.0))

        def f(x):
            shared.value = jnp.array(55.0)  # broadcast write
            return x

        mapped = state_map_transform(
            f,
            in_axes=0, out_axes=0,
            state_out_axes={1: shared},
        )
        mapped(jnp.ones((4,)))
        # Should have restored the broadcast value
        self.assertEqual(float(shared.value), 55.0)


class TestRemoveAxisErrors(unittest.TestCase):
    """B8: _remove_axis must raise real exceptions (not assert) so they survive -O."""

    def test_non_int_axis_raises_typeerror(self):
        from brainstate.transform._mapping_core import _remove_axis
        with self.assertRaises(TypeError):
            _remove_axis(jnp.zeros((3, 2)), 1.5)  # float axis

    def test_out_of_bounds_axis_raises_valueerror(self):
        from brainstate.transform._mapping_core import _remove_axis
        with self.assertRaises(ValueError):
            _remove_axis(jnp.zeros((3,)), 5)


class TestBatchSizeErrors(unittest.TestCase):
    """B8: indeterminate batch size raises ValueError (not AssertionError)."""

    def test_indeterminate_batch_size_raises_valueerror(self):
        from brainstate.transform._mapping_core import _get_batch_size
        with self.assertRaises(ValueError):
            _get_batch_size((jnp.array(3.),), in_axes=None, in_states={}, axis_size=None)


if __name__ == "__main__":
    unittest.main()
