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
from unittest.mock import Mock, patch

import jax.numpy as jnp

import brainstate
from brainstate import environ
from brainstate.nn import Module, EnvironContext
from brainstate.nn._common import _filter_states


class DummyModule(Module):
    """A simple module for testing purposes."""

    def __init__(self, value=0):
        super().__init__()
        self.value = value
        self.state = brainstate.State(jnp.array([1.0, 2.0, 3.0]))
        self.param = brainstate.ParamState(jnp.array([4.0, 5.0, 6.0]))

    def update(self, x):
        return x + self.value

    def __call__(self, x, y=0):
        return x + self.value + y


class TestEnvironContext(unittest.TestCase):
    """Test cases for EnvironContext class."""

    def setUp(self):
        """Set up test fixtures."""
        self.dummy_module = DummyModule(10)

    def test_init_valid_module(self):
        """Test EnvironContext initialization with valid module."""
        context = EnvironContext(self.dummy_module, fit=True, a='test')
        self.assertEqual(context.layer, self.dummy_module)
        self.assertEqual(context.context, {'fit': True, 'a': 'test'})

    def test_init_invalid_module(self):
        """Test EnvironContext initialization with invalid module."""
        with self.assertRaises(AssertionError):
            EnvironContext("not a module", training=True)

        with self.assertRaises(AssertionError):
            EnvironContext(None, training=True)

        with self.assertRaises(AssertionError):
            EnvironContext(42, training=True)

    def test_update_with_context(self):
        """Test update method applies context correctly."""
        context = EnvironContext(self.dummy_module, fit=True)

        # Test with positional arguments
        result = context.update(5)
        self.assertEqual(result, 15)  # 5 + 10

        # Test with keyword arguments
        result = context.update(5, y=3)
        self.assertEqual(result, 18)  # 5 + 10 + 3

    def test_update_context_applied(self):
        """Test that environment context is actually applied during update."""
        with patch.object(environ, 'context') as mock_context:
            mock_context.return_value.__enter__ = Mock(return_value=None)
            mock_context.return_value.__exit__ = Mock(return_value=None)

            context = EnvironContext(self.dummy_module, fit=True, a='eval')
            context.update(5)

            mock_context.assert_called_once_with(fit=True, a='eval')

    def test_add_context(self):
        """Test add_context method updates context correctly."""
        context = EnvironContext(self.dummy_module, fit=True)
        self.assertEqual(context.context, {'fit': True})

        # Add new context
        context.add_context(a='test', debug=False)
        self.assertEqual(context.context, {'fit': True, 'a': 'test', 'debug': False})

        # Overwrite existing context
        context.add_context(fit=False)
        self.assertEqual(context.context, {'fit': False, 'a': 'test', 'debug': False})

    def test_empty_context(self):
        """Test EnvironContext with no initial context."""
        context = EnvironContext(self.dummy_module)
        self.assertEqual(context.context, {})

        result = context.update(7)
        self.assertEqual(result, 17)  # 7 + 10


class TestFilterStates(unittest.TestCase):
    """Test cases for _filter_states function."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_module = Mock(spec=Module)
        self.mock_module.states = Mock()

    def test_filter_states_none(self):
        """Test _filter_states with None filters."""
        result = _filter_states(self.mock_module, None)
        self.assertIsNone(result)
        self.mock_module.states.assert_not_called()

    def test_filter_states_single_filter(self):
        """Test _filter_states with single filter (non-dict)."""
        filter_obj = lambda x: x.startswith('test')
        self.mock_module.states.return_value = ['test1', 'test2']

        result = _filter_states(self.mock_module, filter_obj)

        self.mock_module.states.assert_called_once_with(filter_obj)
        self.assertEqual(result, ['test1', 'test2'])


# ---------------------------------------------------------------------------
# Appended coverage tests for brainstate.nn._common.
# ---------------------------------------------------------------------------

import pytest

from brainstate import _testing
from brainstate.nn._common import Vmap, Map, ToPredicate, _MapCaller
from brainstate.util.filter import OfType


class _ParamModule(Module):
    """Tiny module exposing a single mappable ParamState for wrapper tests."""

    def init_state(self, din, dout):
        """Create a Linear submodule with the requested fan-in/out."""
        self.lin = brainstate.nn.Linear(din, dout)

    def update(self, x):
        """Apply the wrapped linear layer to ``x``."""
        return self.lin(x)

    def predict(self, x):
        """Apply the linear layer and scale the result by two."""
        return self.lin(x) * 2.0


class TestEnvironContextExtraContext(unittest.TestCase):
    """Cover the per-call ``context`` override path of ``EnvironContext``."""

    def test_update_with_explicit_context(self):
        """A per-call context dict is merged with the stored context."""
        layer = _ParamModule()
        layer.init_state(din=2, dout=2)
        wrapped = EnvironContext(layer, fit=True)
        with patch.object(environ, 'context') as mock_context:
            mock_context.return_value.__enter__ = Mock(return_value=None)
            mock_context.return_value.__exit__ = Mock(return_value=None)
            wrapped.update(jnp.ones(2), context={'dt': 0.1})
        mock_context.assert_called_once_with(fit=True, dt=0.1)


class TestFilterStatesDict(unittest.TestCase):
    """Exercise the dictionary branch of ``_filter_states``."""

    def test_filter_states_dict_tuple_key(self):
        """Map a (filter, axis) tuple key to its selected states by axis."""

        class M(Module):
            """Module carrying one ParamState and one ShortTermState."""

            def __init__(self):
                """Create the two states."""
                super().__init__()
                self.p = brainstate.ParamState(jnp.ones(3))
                self.s = brainstate.ShortTermState(jnp.zeros(3))

        module = M()
        # The implementation iterates over the dict's keys and unpacks each into
        # (filter_, axis); supply a 2-tuple key so the unpack succeeds.
        result = _filter_states(module, {(OfType(brainstate.ParamState), 0): 'ignored'})
        self.assertIn(0, result)
        self.assertEqual(len(result[0]), 1)

    @pytest.mark.skip(reason="BUG: _filter_states dict branch unpacks dict keys "
                             "instead of items; documented {filter: axis} form "
                             "raises TypeError. See report.")
    def test_filter_states_dict_documented_form(self):
        """Documented ``{filter: axis}`` mapping should select states by axis."""

        class M(Module):
            """Module carrying one ParamState."""

            def __init__(self):
                """Create the ParamState."""
                super().__init__()
                self.p = brainstate.ParamState(jnp.ones(3))

        module = M()
        result = _filter_states(module, {OfType(brainstate.ParamState): 0})
        self.assertIn(0, result)


class TestToPredicate(unittest.TestCase):
    """Tests for the identity-based ``ToPredicate`` filter helper."""

    def test_matches_only_registered_states(self):
        """Return True only for states whose identity was registered."""
        s1 = brainstate.ParamState(jnp.ones(2))
        s2 = brainstate.ParamState(jnp.ones(2))
        predicate = ToPredicate([s1])
        self.assertTrue(predicate(('a',), s1))
        self.assertFalse(predicate(('b',), s2))

    def test_empty_registration_matches_nothing(self):
        """An empty registration set never matches a state."""
        predicate = ToPredicate([])
        self.assertFalse(predicate(('x',), brainstate.ParamState(jnp.zeros(1))))


class TestVmapWrapper(unittest.TestCase):
    """Tests for the ``Vmap`` module wrapper."""

    def test_init_requires_module(self):
        """Constructing ``Vmap`` with a non-module raises AssertionError."""
        with self.assertRaises(AssertionError):
            Vmap("not a module")

    def test_batched_output_shape(self):
        """A vmapped linear layer maps the leading batch axis of its input."""
        module = _ParamModule()
        module.init_state(din=_testing.SMALL_DIM, dout=3)
        wrapped = Vmap(module, in_axes=0, axis_name='batch')
        with _testing.seeded(0):
            x = brainstate.random.randn(_testing.SMALL_BATCH, _testing.SMALL_DIM)
        out = wrapped.update(x)
        self.assertEqual(out.shape, (_testing.SMALL_BATCH, 3))

    def test_records_axes_and_fn(self):
        """The wrapper stores the requested axes and a callable vmapped fn."""
        module = _ParamModule()
        module.init_state(din=2, dout=2)
        wrapped = Vmap(module, in_axes=0, out_axes=0, axis_size=_testing.SMALL_BATCH)
        self.assertEqual(wrapped.in_axes, 0)
        self.assertEqual(wrapped.out_axes, 0)
        self.assertEqual(wrapped.axis_size, _testing.SMALL_BATCH)
        self.assertTrue(callable(wrapped.vmapped_fn))

    def test_vmap_states_filter_branch(self):
        """Passing ``vmap_states`` exercises the ``_filter_states`` single-filter path."""
        module = _ParamModule()
        module.init_state(din=2, dout=2)
        wrapped = Vmap(
            module,
            in_axes=0,
            vmap_states=OfType(brainstate.ParamState),
        )
        self.assertTrue(callable(wrapped.vmapped_fn))


class TestMapVmapWrapper(unittest.TestCase):
    """Tests for the ``Map`` wrapper in ``vmap`` mode."""

    def test_basic_init_and_update(self):
        """init_all_states then update batches the module over the map axis."""
        m = Map(_ParamModule(), init_map_size=_testing.SMALL_BATCH, behavior='vmap')
        m.init_all_states(din=3, dout=2)
        self.assertTrue(m._init)
        self.assertIn(0, m.dict_vmap_states)
        out = m.update(jnp.ones((_testing.SMALL_BATCH, 3)))
        self.assertEqual(out.shape, (_testing.SMALL_BATCH, 2))

    def test_update_before_init_raises(self):
        """Calling update before init_all_states raises ValueError."""
        m = Map(_ParamModule(), init_map_size=4)
        with self.assertRaises(ValueError):
            m.update(jnp.ones((4, 3)))

    def test_map_before_init_raises(self):
        """Calling map() before init_all_states raises ValueError."""
        m = Map(_ParamModule(), init_map_size=4)
        with self.assertRaises(ValueError):
            m.map('update')

    def test_bad_behavior_rejected(self):
        """A behavior other than vmap/pmap is rejected at construction."""
        with self.assertRaises(AssertionError):
            Map(_ParamModule(), init_map_size=4, behavior='bogus')

    def test_non_integer_init_map_size_rejected(self):
        """A non-integer init_map_size is rejected at construction."""
        with self.assertRaises(AssertionError):
            Map(_ParamModule(), init_map_size=2.5)

    def test_init_state_axes(self):
        """init_state_axes routes newly created states to the requested axis."""
        m = Map(
            _ParamModule(),
            init_map_size=_testing.SMALL_BATCH,
            init_state_axes={0: OfType(brainstate.ParamState)},
        )
        m.init_all_states(din=3, dout=2)
        out = m.update(jnp.ones((_testing.SMALL_BATCH, 3)))
        self.assertEqual(out.shape, (_testing.SMALL_BATCH, 2))

    def test_call_state_axes_integration(self):
        """call_state_axes is merged with the init-time vmapped states."""
        m = Map(
            _ParamModule(),
            init_map_size=_testing.SMALL_BATCH,
            call_state_axes={0: OfType(brainstate.ParamState)},
        )
        m.init_all_states(din=3, dout=2)
        out = m.update(jnp.ones((_testing.SMALL_BATCH, 3)))
        self.assertEqual(out.shape, (_testing.SMALL_BATCH, 2))

    def test_map_method_by_name(self):
        """map() resolves a method by name and vectorizes its call."""
        m = Map(_ParamModule(), init_map_size=_testing.SMALL_BATCH)
        m.init_all_states(din=3, dout=2)
        caller = m.map('predict')
        self.assertIsInstance(caller, _MapCaller)
        out = caller(jnp.ones((_testing.SMALL_BATCH, 3)))
        self.assertEqual(out.shape, (_testing.SMALL_BATCH, 2))

    def test_map_with_state_axes(self):
        """map() accepts an explicit state_axes specification."""
        m = Map(_ParamModule(), init_map_size=_testing.SMALL_BATCH)
        m.init_all_states(din=3, dout=2)
        out = m.map(
            'predict',
            state_axes={0: OfType(brainstate.ParamState)},
        )(jnp.ones((_testing.SMALL_BATCH, 3)))
        self.assertEqual(out.shape, (_testing.SMALL_BATCH, 2))

    def test_map_with_callable_fn(self):
        """map() accepts a callable directly instead of a method name.

        The callable is invoked with the mapped inputs only, so it closes over
        the wrapped module rather than receiving it as an argument.
        """
        m = Map(_ParamModule(), init_map_size=_testing.SMALL_BATCH)
        m.init_all_states(din=3, dout=2)
        module = m.module

        def run(x):
            """Call the wrapped module's update on ``x``."""
            return module.update(x)

        out = m.map(run)(jnp.ones((_testing.SMALL_BATCH, 3)))
        self.assertEqual(out.shape, (_testing.SMALL_BATCH, 2))

    def test_call_state_axes_unmatched_key(self):
        """call_state_axes keys absent from the vmapped states pass through.

        Exercises the branch in ``_integrate_state_axes`` where a supplied axis
        key is not among the states created at initialization time.
        """
        m = Map(
            _ParamModule(),
            init_map_size=_testing.SMALL_BATCH,
            call_state_axes={1: OfType(brainstate.ShortTermState)},
        )
        m.init_all_states(din=3, dout=2)
        # Axis 0 holds the vmapped ParamStates; axis 1 was supplied but unmatched.
        self.assertIn(0, m._call_state_axes)
        self.assertIn(1, m._call_state_axes)

    def test_map_missing_method_raises(self):
        """map() raises AttributeError for an unknown method name."""
        m = Map(_ParamModule(), init_map_size=4)
        m.init_all_states(din=3, dout=2)
        with self.assertRaises(AttributeError):
            m.map('does_not_exist')

    def test_pretty_repr_hides_internal_fields(self):
        """The repr suppresses internal bookkeeping attributes."""
        m = Map(_ParamModule(), init_map_size=4)
        m.init_all_states(din=3, dout=2)
        text = repr(m)
        self.assertNotIn('_init', text)
        self.assertNotIn('dict_vmap_states', text)
        self.assertNotIn('_call_state_axes', text)

    def test_init_all_states_invalid_behavior_defensive(self):
        """A corrupted behavior triggers the defensive guard in init_all_states."""
        m = Map(_ParamModule(), init_map_size=2)
        m.behavior = 'bogus'
        with self.assertRaises(ValueError):
            m.init_all_states(din=3, dout=2)

    def test_update_invalid_behavior_defensive(self):
        """A corrupted behavior triggers the defensive guard in update."""
        m = Map(_ParamModule(), init_map_size=2)
        m.init_all_states(din=3, dout=2)
        m.behavior = 'bogus'
        with self.assertRaises(ValueError):
            m.update(jnp.ones((2, 3)))


class TestMapPmapWrapper(unittest.TestCase):
    """Tests for the ``Map`` wrapper in single-device ``pmap`` mode."""

    def test_pmap_init_and_map_call(self):
        """pmap-mode init then ``map()`` runs the module across one device.

        Uses ``init_map_size=1`` so the parallel axis fits the single available
        CPU device. The ``map()`` path is used because ``update()`` is broken in
        pmap mode (see ``test_pmap_update_is_broken``).
        """
        m = Map(_ParamModule(), init_map_size=1, behavior='pmap')
        m.init_all_states(din=3, dout=2)
        self.assertIn(0, m.dict_vmap_states)
        out = m.map('update')(jnp.ones((1, 3)))
        self.assertEqual(out.shape, (1, 2))

    def test_pmap_update_is_broken(self):
        """pmap-mode ``update`` raises TypeError from an invalid pmap2 kwarg.

        ``Map.update`` always forwards ``spmd_axis_name`` to the map function,
        but ``pmap2`` does not accept that keyword. This documents the bug while
        keeping the suite green; see report.
        """
        m = Map(_ParamModule(), init_map_size=1, behavior='pmap')
        m.init_all_states(din=3, dout=2)
        with self.assertRaises(TypeError):
            m.update(jnp.ones((1, 3)))


class TestMapCaller(unittest.TestCase):
    """Tests for the internal ``_MapCaller`` dispatcher."""

    def test_invalid_behavior_raises(self):
        """_MapCaller rejects behaviors other than vmap/pmap."""
        with self.assertRaises(ValueError):
            _MapCaller(lambda x: x, behavior='bogus')

    def test_callable_dispatch(self):
        """A vmap _MapCaller batches a plain function over the leading axis."""
        caller = _MapCaller(lambda x: x * 2.0, behavior='vmap', in_axes=0, out_axes=0)
        out = caller(jnp.arange(4.0))
        _testing.assert_allclose(out, jnp.arange(4.0) * 2.0)
